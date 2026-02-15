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


# -----------------------------
# Page config
# -----------------------------
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
# Model registry (must match your saved filenames)
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
    """Return positive-class probabilities if model supports predict_proba; else None."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    return None


def plot_cm(cm: np.ndarray):
    """Simple confusion matrix visualization with matplotlib (no seaborn)."""
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    st.pyplot(fig)


def build_inverse_label_map(label_map: dict):
    """
    Convert LABEL_MAP like {"0": "Malignant", "1": "Benign"}
    into {"malignant": 0, "benign": 1}
    """
    inv = {}
    if isinstance(label_map, dict):
        for k, v in label_map.items():
            try:
                inv[str(v).strip().lower()] = int(str(k).strip())
            except Exception:
                pass
    return inv


def normalize_y_true(y_true_series: pd.Series, label_map: dict):
    """
    Make y_true numeric (0/1) if possible, handling:
      - already numeric 0/1
      - strings "0"/"1"
      - "Benign"/"Malignant"
      - "B"/"M"
      - values matching LABEL_MAP descriptions
    Returns: (y_true_np, ok, message)
    """
    # If already numeric
    if pd.api.types.is_numeric_dtype(y_true_series):
        y = pd.to_numeric(y_true_series, errors="coerce")
        if y.isna().any():
            return None, False, "Target column contains NaN or non-numeric values."
        return y.astype(int).to_numpy().ravel(), True, ""

    # Convert to clean strings
    y_str = y_true_series.astype(str).str.strip().str.lower()

    # First try numeric conversion from strings "0"/"1"
    y_num = pd.to_numeric(y_str, errors="coerce")
    if not y_num.isna().any():
        return y_num.astype(int).to_numpy().ravel(), True, ""

    # Build mapping from label_map + common values
    inv = build_inverse_label_map(label_map)
    inv.update({
        "malignant": 0, "m": 0,
        "benign": 1, "b": 1
    })

    y_mapped = y_str.map(inv)
    if y_mapped.isna().any():
        uniques = y_str.unique().tolist()
        return None, False, f"Unrecognized target labels: {uniques[:20]}. Use 0/1 or Benign/Malignant (or B/M)."

    return y_mapped.astype(int).to_numpy().ravel(), True, ""


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))
mode = st.sidebar.radio("Mode", ["Upload Test CSV", "Manual Single Prediction"])

# Load selected model
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

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        st.markdown("### Preview")
        st.dataframe(df.head())

        # Choose target column (optional)
        default_index = 0
        if TARGET_DEFAULT in df.columns:
            default_index = 1

        target_col = st.selectbox(
            "Target column (optional). Select 'None' for prediction-only CSV.",
            ["None"] + list(df.columns),
            index=default_index
        )

        # Validate feature columns
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

        # Show predictions
        out = pd.DataFrame({"prediction": y_pred})
        if LABEL_MAP:
            out["prediction_label"] = out["prediction"].astype(str).map(LABEL_MAP).fillna(out["prediction"].astype(str))

        st.markdown("### Predictions (first 50)")
        st.dataframe(out.head(50))

        # If ground truth provided, compute metrics + CM + report
        if target_col != "None":
            st.markdown("## Evaluation Results")

            # Drop rows where target is missing and align y_pred
            mask = df[target_col].notna()
            if mask.sum() == 0:
                st.error("Target column selected but all target values are empty/NaN.")
                st.stop()

            df_eval = df.loc[mask].copy()
            X_eval = df_eval[FEATURES].copy()
            y_pred_eval = np.array(y_pred)[mask.to_numpy()]

            # Normalize y_true
            y_true, ok, msg = normalize_y_true(df_eval[target_col], LABEL_MAP)
            if not ok:
                st.error(msg)
                st.stop()

            # Metrics
            acc = accuracy_score(y_true, y_pred_eval)
            prec = precision_score(y_true, y_pred_eval, zero_division=0)
            rec = recall_score(y_true, y_pred_eval, zero_division=0)
            f1 = f1_score(y_true, y_pred_eval, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred_eval)

            # AUC only if proba exists
            y_prob = positive_class_proba_if_available(model, X_eval)
            auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

            metrics = {
                "Accuracy": float(acc),
                "AUC": float(auc) if auc is not None else "Not available (no predict_proba)",
                "Precision": float(prec),
                "Recall": float(rec),
                "F1": float(f1),
                "MCC": float(mcc),
            }

            st.markdown("### Metrics")
            st.json(metrics)

            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred_eval)
            plot_cm(cm)

            st.markdown("### Classification Report")
            st.text(classification_report(y_true, y_pred_eval, zero_division=0))

    else:
        st.info("Upload a CSV to proceed.")


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
