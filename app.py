import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("Breast Cancer Prediction using XGBoost")
st.markdown("Enter tumor feature values below to predict whether the tumor is **Malignant** or **Benign**.")

st.subheader("Input Features")

# Create 3 columns for better layout
col1, col2, col3 = st.columns(3)

# Column 1
with col1:
    mean_radius = st.number_input("mean radius")
    mean_texture = st.number_input("mean texture")
    mean_perimeter = st.number_input("mean perimeter")
    mean_area = st.number_input("mean area")
    mean_smoothness = st.number_input("mean smoothness")
    mean_compactness = st.number_input("mean compactness")
    mean_concavity = st.number_input("mean concavity")
    mean_concave_points = st.number_input("mean concave points")
    mean_symmetry = st.number_input("mean symmetry")
    mean_fractal_dimension = st.number_input("mean fractal dimension")

# Column 2
with col2:
    radius_error = st.number_input("radius error")
    texture_error = st.number_input("texture error")
    perimeter_error = st.number_input("perimeter error")
    area_error = st.number_input("area error")
    smoothness_error = st.number_input("smoothness error")
    compactness_error = st.number_input("compactness error")
    concavity_error = st.number_input("concavity error")
    concave_points_error = st.number_input("concave points error")
    symmetry_error = st.number_input("symmetry error")
    fractal_dimension_error = st.number_input("fractal dimension error")

# Column 3
with col3:
    worst_radius = st.number_input("worst radius")
    worst_texture = st.number_input("worst texture")
    worst_perimeter = st.number_input("worst perimeter")
    worst_area = st.number_input("worst area")
    worst_smoothness = st.number_input("worst smoothness")
    worst_compactness = st.number_input("worst compactness")
    worst_concavity = st.number_input("worst concavity")
    worst_concave_points = st.number_input("worst concave points")
    worst_symmetry = st.number_input("worst symmetry")
    worst_fractal_dimension = st.number_input("worst fractal dimension")

# Combine all features in correct order
features = np.array([[
    mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
    mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension,
    radius_error, texture_error, perimeter_error, area_error, smoothness_error,
    compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
    worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
    worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension
]])

if st.button("Predict"):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    confidence = max(probability[0]) * 100

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠ Malignant Tumor Detected")
    else:
        st.success("✅ Benign Tumor Detected")

    st.write(f"Model Confidence: {confidence:.2f}%")