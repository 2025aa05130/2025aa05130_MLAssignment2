import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt


st.set_page_config(page_title="BITS ML Assignment 2", layout="wide")
st.title("BITS ML Assignment 2 — Multi-Model Classification App")


# -----------------------------
# Load meta.json (features)
# -----------------------------
META_PATH = "model/meta.json"
try:
    with open(META_PATH, "r") as f:
        meta = json.load(f)
except FileNotFoundError:
    st.error(
        "meta.json not found at model/meta.json.\n\n"
        "Fix: Ensure your GitHub repo has model/meta.json committed.\n"
        "Also ensure the folder name is exactly 'model'."
    )
    st.stop()

FEATURES = meta.get("feature_names", [])
TARGET_DEFAULT = meta.get("target_name", "target")
LABEL_MAP = meta.get("label_map", {})  # e.g. {"0":"Malignant","1":"Benign"}

if not FEATURES:
    st.error("meta.json does not contain 'feature_names'. Please regenerate meta.json correctly.")
    st.stop()


# -----------------------------
# Model registry
# -----------------------------
MODEL_FILES = {
    "Logistic Regression": "model/Logistic_Regression.pkl",
    "Decision Tree": "model/Decision_Tree.pkl",
    "KNN": "model/KNN.pkl",
    "Naive Bayes": "model/Naive_Bayes.pkl",
    "Random Forest": "model/Random_Forest.pkl",
    "XGBoost": "model/XGBoost.pkl",
}


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def positive_class_proba_if_available(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    return None


def plot_cm(cm: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    st.pyplot(fig)


def build_inverse_label_map(label_map: dict):
    inv = {}
    if isinstance(label_map, dict):
        for k, v in label_map.items():
            try:
                inv[str(v).strip().lower()] = int(str(k).strip())
            except Exception:
                pass
    return inv


def normalize_y_true(y_true_series: pd.Series, label_map: dict):
    # Drop blanks that look like "nan" strings
    y_clean = y_true_series.copy()

    # If already numeric
    if pd.api.types.is_numeric_dtype(y_clean):
        y = pd.to_numeric(y_clean, errors="coerce")
        if y.isna().any():
            return None, False, "Target column contains NaN or non-numeric values."
        return y.astype(int).to_numpy().ravel(), True, ""

    y_str = y_clean.astype(str).str.strip().str.lower()

    # remove empty strings
    y_str = y_str.replace({"": np.nan, "nan": np.nan, "none": np.nan})
    if y_str.isna().any():
        # caller already drops NaNs, but just in case:
        y_str = y_str.dropna()

    # numeric strings
    y_num = pd.to_numeric(y_str, errors="coerce")
    if not y_num.isna().any():
        return y_num.astype(int).to_numpy().ravel(), True, ""

    inv = build_inverse_label_map(label_map)
    inv.update({"malignant": 0, "m": 0, "benign": 1, "b": 1})

    y_mapped = y_str.map(inv)
    if y_mapped.isna().any():
        uniques = y_str.unique().tolist()
        return None, False, f"Unrecognized target labels: {uniques[:20]}. Use 0/1 or Benign/Malignant (or B/M)."

    return y_mapped.astype(int).to_numpy().ravel(), True, ""


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))
mode = st.sidebar.radio("Mode", ["Upload Test CSV", "Manual Single Prediction"])

try:
    model = load_model(MODEL_FILES[model_name])
except FileNotFoundError:
    st.error(
        f"Model file not found: {MODEL_FILES[model_name]}\n\n"
        "Fix: Make sure all .pkl files are inside the 'model/' folder in your GitHub repo."
    )
    st.stop()

st.subheader(f"Selected Model: {model_name}")


# -----------------------------
# Mode 1: Upload CSV
# -----------------------------
if mode == "Upload Test CSV":
    st.write(
        "Upload a **test CSV** with the same feature columns used for training.\n\n"
        "If your CSV also contains ground-truth labels (target), select the target column below to compute metrics."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV to proceed.")
        st.stop()

    df = pd.read_csv(uploaded)

    st.markdown("### Preview")
    st.dataframe(df.head())

    default_index = 0
    if TARGET_DEFAULT in df.columns:
        default_index = 1

    target_col = st.selectbox(
        "Target column (optional). Select 'None' for prediction-only CSV.",
        ["None"] + list(df.columns),
        index=default_index
    )

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(
            "Your CSV is missing required feature columns.\n\n"
            f"Missing columns: {missing}\n\n"
            "Fix: Use the same column names as meta.json (training features)."
        )
        st.stop()

    X_input = df[FEATURES].copy()
    y_pred = model.predict(X_input)

    out = pd.DataFrame({"prediction": y_pred})
    if LABEL_MAP:
        out["prediction_label"] = out["prediction"].astype(str).map(LABEL_MAP).fillna(out["prediction"].astype(str))

    st.markdown("### Predictions (first 50)")
    st.dataframe(out.head(50))

    if target_col == "None":
        st.stop()

    st.markdown("## Evaluation Results")

    # Align rows where target exists
    mask = df[target_col].notna()
    if mask.sum() == 0:
        st.error("Target column selected but all target values are empty/NaN.")
        st.stop()

    df_eval = df.loc[mask].copy()
    X_eval = df_eval[FEATURES].copy()
    y_pred_eval = np.array(y_pred)[mask.to_numpy()]

    y_true, ok, msg = normalize_y_true(df_eval[target_col], LABEL_MAP)
    if not ok:
        st.error(msg)
        st.stop()

    acc = accuracy_score(y_true, y_pred_eval)

    labels_present = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred_eval)]))
    single_class = (len(labels_present) < 2)

    if single_class:
        st.warning(
            f"Only one class found in uploaded data/predictions ({labels_present.tolist()}). "
            "Precision/Recall/F1 and AUC are not well-defined for a single-class set."
        )
        prec = "Not defined (single class)"
        rec = "Not defined (single class)"
        f1v = "Not defined (single class)"
        auc = "Not defined (single class)"
    else:
        # Auto-average to avoid multiclass/binary crash
        n_classes = len(labels_present)
        avg = "binary" if n_classes == 2 else "macro"
        if avg != "binary":
            st.warning(f"Detected {n_classes} classes {labels_present.tolist()} → using average='{avg}' for Precision/Recall/F1.")

        prec = float(precision_score(y_true, y_pred_eval, average=avg, zero_division=0))
        rec = float(recall_score(y_true, y_pred_eval, average=avg, zero_division=0))
        f1v = float(f1_score(y_true, y_pred_eval, average=avg, zero_division=0))

        y_prob = positive_class_proba_if_available(model, X_eval)
        if avg == "binary" and y_prob is not None:
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            auc = "Not available"

    mcc = float(matthews_corrcoef(y_true, y_pred_eval))

    metrics = {
        "Accuracy": float(acc),
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1v,
        "MCC": mcc,
    }

    st.markdown("### Metrics")
    st.json(metrics)

    st.markdown("### Confusion Matrix")
    # If your dataset is binary, keep labels [0,1]; otherwise let sklearn infer
    if len(np.unique(y_true)) <= 2:
        cm = confusion_matrix(y_true, y_pred_eval, labels=[0, 1])
    else:
        cm = confusion_matrix(y_true, y_pred_eval)
    plot_cm(cm)

    st.markdown("### Classification Report")
    if len(np.unique(y_true)) <= 2:
        st.text(classification_report(y_true, y_pred_eval, labels=[0, 1], zero_division=0))
    else:
        st.text(classification_report(y_true, y_pred_eval, zero_division=0))


# -----------------------------
# Mode 2: Manual Single Prediction
# -----------------------------
else:
    st.write("Enter feature values for a single prediction. Fields come from **model/meta.json**.")

    cols = st.columns(3)
    values = {}

    for i, feat in enumerate(FEATURES):
        with cols[i % 3]:
            values[feat] = st.number_input(feat, value=0.0)

    if st.button("Predict"):
        X_one = pd.DataFrame([values], columns=FEATURES)
        pred = model.predict(X_one)[0]

        pred_label = LABEL_MAP.get(str(pred), str(pred))
        st.success(f"Prediction: **{pred_label}** (raw={pred})")

        prob = positive_class_proba_if_available(model, X_one)
        if prob is not None:
            st.info(f"Positive-class probability: **{float(prob[0]):.4f}**")
