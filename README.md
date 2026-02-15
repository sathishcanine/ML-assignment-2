# 🌲 Forest Cover Type Classification

## 📌 Problem Statement

The objective of this project is to build and evaluate multiple machine
learning classification models to predict the **forest cover type**
based on cartographic and environmental features.

The task is a **multi-class classification problem** where each instance
must be classified into one of seven forest cover types using geographic
attributes such as elevation, slope, distance to water sources, soil
type, and wilderness area.

This project demonstrates the complete end-to-end machine learning
workflow including:

-   Data preprocessing
-   Model training
-   Performance evaluation
-   Model comparison
-   Web app deployment using Streamlit

------------------------------------------------------------------------

## 📊 Dataset Description

-   **Dataset Name:** Forest Cover Type Dataset
-   **Source:** UCI Machine Learning Repository
-   **Total Instances:** 581,012
-   **Features:** 54 input features
-   **Target Variable:** `Cover_Type` (7 classes)

### Feature Categories:

1.  Quantitative Features:
    -   Elevation
    -   Aspect
    -   Slope
    -   Horizontal/Vertical distance to hydrology
    -   Horizontal distance to roadways
    -   Hillshade measurements
    -   Horizontal distance to fire points
2.  Binary Features:
    -   Wilderness Area (4 binary columns)
    -   Soil Type (40 binary columns)

### Target Classes:

Forest cover types labeled from 1 to 7.

For model compatibility (especially XGBoost), target labels were
converted from:

    1–7  →  0–6

------------------------------------------------------------------------

## 🤖 Models Implemented

The following 6 classification models were implemented on the same
dataset:

1.  Logistic Regression
2.  Decision Tree Classifier
3.  K-Nearest Neighbors (KNN)
4.  Naive Bayes (Gaussian)
5.  Random Forest (Ensemble)
6.  XGBoost (Ensemble)

All models were evaluated using the following metrics:

-   Accuracy
-   AUC Score (One-vs-Rest for multiclass)
-   Precision (Weighted)
-   Recall (Weighted)
-   F1 Score (Weighted)
-   Matthews Correlation Coefficient (MCC)

------------------------------------------------------------------------

## 📈 Model Comparison Table

| ML Model Name        | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC    |
|----------------------|----------|---------|-----------|---------|----------|--------|
| Logistic Regression  | 0.7291   | 0.9374  | 0.7231    | 0.7291  | 0.7197   | 0.5574 |
| Decision Tree        | 0.8192   | 0.8473  | 0.8193    | 0.8192  | 0.8192   | 0.7106 |
| KNN                  | 0.8276   | 0.9393  | 0.8253    | 0.8276  | 0.8257   | 0.7218 |
| Naive Bayes          | 0.4555   | 0.8863  | 0.6552    | 0.4555  | 0.4113   | 0.3113 |
| Random Forest        | 0.8487   | 0.9810  | 0.8511    | 0.8487  | 0.8445   | 0.7543 |
| XGBoost              | 0.8269   | 0.9760  | 0.8273    | 0.8269  | 0.8245   | 0.7193 |


------------------------------------------------------------------------

## 🔎 Model Performance Observations

-----------------------------------------------------------------------
ML Model           Observation About Performance
------------------ ----------------------------------------------------
Logistic           Achieves moderate performance with reasonable AUC.
Regression         However, due to its linear decision boundary
                   assumption, it struggles to capture complex
                   non-linear relationships present in the dataset.

Decision Tree      Performs better than Logistic Regression by capturing
                   non-linear feature interactions. However, its
                   generalization ability is lower compared to ensemble
                   methods on the sampled dataset.

KNN                Provides competitive performance and handles
                   high-dimensional data well. Since it relies on
                   distance-based similarity, its performance is stable
                   but computationally more expensive as dataset size
                   increases.

Naive Bayes        Shows the lowest accuracy among all models. This is
                   expected because it assumes feature independence,
                   while many geographical and soil features in the
                   dataset are correlated.

Random Forest      Achieved the highest overall performance among all
                   models in the 50k sample setting. By combining
                   multiple decision trees, it reduces variance and
                   improves generalization capability.

XGBoost            Performs very competitively and achieves high AUC.
                   Although slightly below Random Forest in accuracy,
                   it demonstrates strong boosting-based learning and
                   effective handling of feature interactions.
-----------------------------------------------------------------------


### Key Insights:

-   Ensemble models outperform single models.
-   Random Forest achieved the highest overall performance.
-   Naive Bayes underperformed due to independence assumption.
-   Class imbalance was handled effectively using weighted evaluation
    metrics.
-   MCC provided a reliable performance measure under imbalance.

------------------------------------------------------------------------

## 🌐 Streamlit Application Features

The deployed Streamlit web application includes:

-   CSV dataset upload option
-   Model selection dropdown
-   Evaluation metric display
-   Confusion matrix visualization
-   Classification report display
-   Download button for sample test dataset

------------------------------------------------------------------------

## 📦 Project Structure

    project-folder/
    │-- app.py
    │-- requirements.txt
    │-- README.md
    │-- model/
    │   ├── Logistic_Regression.pkl
    │   ├── Decision_Tree.pkl
    │   ├── KNN.pkl
    │   ├── Naive_Bayes.pkl
    │   ├── Random_Forest.pkl
    │   ├── XGBoost.pkl
    │   ├── scaler.pkl
    │   └── model_results.csv


1. GitHub Repository Link:
https://github.com/sathishcanine/ML-assignment-2.git

2. Live Streamlit App Link:
https://sathish-2025aa05083.streamlit.app/