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


def normalize_y_true(y_true_raw: pd.Series, label_map: dict, y_pred: np.ndarray):
    """
    Convert y_true into the same style as y_pred.
    Handles:
      - numeric 0/1
      - strings "0"/"1"
      - "Benign"/"Malignant"
      - "B"/"M"
      - label_map-based mapping (inverse)
    Returns: (y_true_numeric, ok, error_message)
    """

    # 1) Make a clean string series for mapping attempts
    y_str = y_true_raw.astype(str).str.strip()

    # 2) If y_pred is numeric, we try to convert y_true to numeric
    y_pred_is_numeric = np.issubdtype(np.array(y_pred).dtype, np.number)

    # Helper: try direct numeric conversion
    def try_numeric(series):
        out = pd.to_numeric(series, errors="coerce")
        return out

    # Build an inverse label map (e.g. {"malignant":0, "benign":1})
    inv = {}
    if isinstance(label_map, dict) and len(label_map) > 0:
        for k, v in label_map.items():
            inv[str(v).strip().lower()] = int(str(k).strip())

    # Add common fallbacks (in case label_map is missing or different)
    inv.update({
        "malignant": 0, "m": 0, "0": 0,
        "benign": 1, "b": 1, "1": 1
    })

    # 3) Attempt conversions
    if y_pred_is_numeric:
        # Try numeric first
        y_num = try_numeric(y_str)
        if not y_num.isna().any():
            return y_num.astype(int).to_numpy().ravel(), True, ""

        # Try mapping strings like "Benign"/"Malignant"
        y_mapped = y_str.str.lower().map(inv)
        if not y_mapped.isna().any():
            return y_mapped.astype(int).to_numpy().ravel(), True, ""

        # If still failing, show user what values were found
        unique_vals = y_str.unique().tolist()[:20]
        return None, False, f"Could not convert target labels to numeric. Found values like: {unique_vals}"

    else:
        # If model predicts labels as strings, try to make y_true strings comparable
        # (rare in your setup, but safe)
        return y_str.to_numpy().ravel(), True, ""


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

        # Predict for all rows (even if target exists; we will align later)
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

            # Drop rows where target is missing, and align y_pred accordingly
            mask = df[target_col].notna()
            if mask.sum() == 0:
                st.error("Target column is selected but all values are empty/NaN. Please upload a CSV with target values.")
                st.stop()

            y_true_raw = df.loc[mask, target_col]
            y_pred_eval = np.array(y_pred)[mask.to_numpy()]

            # Normalize y_true to match y_pred style
            y_true, ok, msg = normalize_y_true(y_true_raw, LABEL_MAP, y_pred_eval)
            if not ok:
                st.error(msg)
                st.stop()

            # Decide pos_label automatically for binary metrics
            unique_classes = np.unique(y_true)
            if len(unique_classes) == 2:
                pos_label = int(np.max(unique_classes))
            else:
                pos_label = 1  # fallback

            # core metrics (binary classification expected)
            acc = accuracy_score(y_true, y_pred_eval)
            prec = precision_score(y_true, y_pred_eval, pos_label=pos_label, zero_division=0)
            rec = recall_score(y_true, y_pred_eval, pos_label=pos_label, zero_division=0)
            f1 = f1_score(y_true, y_pred_eval, pos_label=pos_label, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred_eval)

            # AUC only if proba exists
            y_prob = positive_class_proba_if_available(model, X_input.loc[mask])
            if y_prob is not None:
                auc = roc_auc_score(y_true, y_prob)
            else:
                auc = None

            metrics = {
                "Accuracy": float(acc),
                "AUC": (float(auc) if auc is not None else "Not available (no predict_proba)"),
                "Precision": float(
