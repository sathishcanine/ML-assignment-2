import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Forest Cover Type Classification", layout="wide")

st.title("🌲 Forest Cover Type Classification App")
st.write("Upload test dataset and evaluate selected model.")


st.subheader("📥 Download Sample Test Dataset")

st.markdown("""
    <style>
    div.stDownloadButton > button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1em;
    }
    div.stDownloadButton > button:hover {
        background-color: #218838;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

with open("test_data.csv", "rb") as file:
    st.download_button(
        label="Download test_data.csv",
        data=file,
        file_name="test_data.csv",
        mime="text/csv"
    )


@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/Logistic_Regression.pkl"),
        "Decision Tree": joblib.load("model/Decision_Tree.pkl"),
        "KNN": joblib.load("model/KNN.pkl"),
        "Naive Bayes": joblib.load("model/Naive_Bayes.pkl"),
        "Random Forest": joblib.load("model/Random_Forest.pkl"),
        "XGBoost": joblib.load("model/XGBoost.pkl")
    }
    return models

@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.pkl")

models = load_models()
scaler = load_scaler()


uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write("Dataset Shape:", df.shape)

    if "Cover_Type" not in df.columns:
        st.error("Uploaded file must contain 'Cover_Type' column.")
    else:
        X = df.drop("Cover_Type", axis=1)
        y = df["Cover_Type"] - 1   # convert 1-7 to 0-6

        
        selected_model_name = st.selectbox(
            "Select Model",
            list(models.keys())
        )

        model = models[selected_model_name]

        # Scale if required
        if selected_model_name in ["Logistic Regression", "KNN"]:
            X = scaler.transform(X)


        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)
            auc = roc_auc_score(y, y_prob, multi_class="ovr")
        else:
            auc = None

       
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="weighted")
        recall = recall_score(y, y_pred, average="weighted")
        f1 = f1_score(y, y_pred, average="weighted")
        mcc = matthews_corrcoef(y, y_pred)

        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", f"{f1:.4f}")
        col5.metric("AUC", f"{auc:.4f}" if auc is not None else "N/A")
        col6.metric("MCC", f"{mcc:.4f}")

        
        st.subheader("🔎 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        st.write(cm)

        
        st.subheader("📄 Classification Report")
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

else:
    st.info("Please upload test_data.csv to continue.")
