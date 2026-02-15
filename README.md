# Machine Learning Assignment 2  
BITS Pilani - Work Integrated Learning Programme  
M.Tech (AIML/DSE)  

---

## 1. Problem Statement

The objective of this assignment is to implement and compare multiple machine learning classification models on a real-world dataset. The task involves training six different classification algorithms, evaluating their performance using standard evaluation metrics, and deploying the models through an interactive Streamlit web application.

The goal is to analyze model performance and identify the best-performing classifier based on multiple evaluation criteria.

---

## 2. Dataset Description

Dataset Name: Breast Cancer Wisconsin Dataset  

Source: UCI Machine Learning Repository  

Number of Instances: 569  
Number of Features: 30  
Type of Problem: Binary Classification  

Target Variable:
- 0 = Malignant
- 1 = Benign

The dataset contains numerical features computed from digitized images of fine needle aspirates (FNA) of breast masses. These features describe characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

The objective is to classify tumors as malignant or benign based on these features.

---

## 3. Machine Learning Models Implemented

The following six classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (KNN)  
4. Naive Bayes (Gaussian Naive Bayes)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

All models were trained and evaluated on the same dataset using an 80-20 train-test split.

---

## 4. Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.96 | 0.99 | 0.96 | 0.96 | 0.96 | 0.91 |
| Decision Tree | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.86 |
| KNN | 0.95 | 0.98 | 0.95 | 0.95 | 0.95 | 0.89 |
| Naive Bayes | 0.94 | 0.98 | 0.94 | 0.94 | 0.94 | 0.88 |
| Random Forest | 0.97 | 0.99 | 0.97 | 0.97 | 0.97 | 0.93 |
| XGBoost | 0.98 | 0.99 | 0.98 | 0.98 | 0.98 | 0.95 |

---

## 5. Observations on Model Performance

| ML Model | Observation |
|------------|-------------|
| Logistic Regression | Performs very well due to strong linear separability in dataset. |
| Decision Tree | Slightly lower accuracy due to tendency to overfit on training data. |
| KNN | Good performance but sensitive to feature scaling and choice of K value. |
| Naive Bayes | Performs well despite independence assumption; computationally efficient. |
| Random Forest | Improved generalization compared to single decision tree due to ensemble averaging. |
| XGBoost | Achieved highest accuracy and MCC score due to boosting mechanism and better handling of feature interactions. |

Overall, ensemble models (Random Forest and XGBoost) outperformed individual models. XGBoost demonstrated the best overall performance across all evaluation metrics.

---

## 6. Evaluation Metrics Used

The following metrics were used to evaluate all models:

- Accuracy
- AUC Score (Area Under ROC Curve)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

These metrics provide a comprehensive understanding of classification performance, especially in binary classification problems.

---

## 7. Streamlit Application Features

The deployed Streamlit application includes:

- Dataset upload option (CSV file)
- Model selection dropdown (6 models)
- Automatic model loading from saved `.pkl` files
- Display of evaluation metrics
- Confusion matrix
- Classification report

The application allows users to test different models dynamically on uploaded test datasets.

---

## 8. Deployment Links

GitHub Repository:  
https://github.com/2025aa05130/2025aa05130_MLAssignment2

Live Streamlit App:  
https://2025aa05130mlassignment2-pgzkidhkvctmqffinnfjz2.streamlit.app/

---

## 9. Conclusion

This project demonstrates the implementation, evaluation, and deployment of multiple machine learning classification models. The results indicate that ensemble methods such as Random Forest and XGBoost provide superior performance on structured classification datasets.

The assignment successfully integrates model development, performance evaluation, web deployment, and reproducible project structuring as required.
