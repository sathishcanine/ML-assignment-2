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

  ------------------------------------------------------------------------------
  ML Model Name       Accuracy   AUC      Precision   Recall   F1 Score MCC
  ------------------- ---------- -------- ----------- -------- -------- --------
  Logistic Regression 0.7227     0.9363   0.7116      0.7227   0.7131   0.5459

  Decision Tree       0.9391     0.9421   0.9390      0.9391   0.9391   0.9022

  KNN                 0.9287     0.9830   0.9285      0.9287   0.9285   0.8853

  Naive Bayes         0.4604     0.8879   0.6481      0.4604   0.4185   0.3106

  Random Forest       0.9539     0.9978   0.9540      0.9539   0.9536   0.9258
    
  XGBoost             0.8699     0.9865   0.8701      0.8700   0.8693   0.7901

  ------------------------------------------------------------------------------

------------------------------------------------------------------------

## 🔎 Model Performance Observations

  -----------------------------------------------------------------------
  ML Model           Observation About Performance
  ------------------ ----------------------------------------------------
  Logistic           Performs moderately well but struggles due to
  Regression         non-linear relationships between features. It
                     assumes linear decision boundaries, which limits
                     performance.

  Decision Tree      Performs significantly better as it captures
                     non-linear feature interactions. However, it may
                     overfit if not controlled.

  KNN                Achieves strong performance due to similarity-based
                     classification in high-dimensional feature space.
                     However, computational cost increases with dataset
                     size.

  Naive Bayes        Performs poorly because it assumes feature
                     independence. Many features in this dataset are
                     correlated, violating this assumption.

  Random Forest      Best performing model. Ensemble of trees reduces
                     variance and improves generalization. Handles
                     non-linearity and feature interactions effectively.

  XGBoost            Strong ensemble model with high AUC. Slightly lower
                     accuracy than Random Forest in this implementation
                     but still performs robustly.
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
