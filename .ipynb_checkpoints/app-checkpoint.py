import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# -------------------------------
# Load the trained Random Forest model
# -------------------------------
model = joblib.load("modelRf.pkl")   # make sure this file exists

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="CardioRisk-Lite", layout="wide")
st.title("💖 CardioRisk-Lite Dashboard")
st.write("Predicting heart disease risk using a Random Forest Classifier.")

# -------------------------------
# User Input Section
# -------------------------------
st.sidebar.header("Enter Patient Information")

def user_input():
    age = st.sidebar.slider("Age", 20, 80, 40)
    sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
    ca = st.sidebar.slider("Number of Major Vessels (0-4)", 0, 4, 0)
    thal = st.sidebar.selectbox("Thal (1 = Normal, 2 = Fixed defect, 3 = Reversable defect)", [1, 2, 3])

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# -------------------------------
# Prediction
# -------------------------------
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

st.subheader("Prediction")
if prediction == 1:
    st.error("⚠️ High Risk of Heart Disease")
else:
    st.success("✅ Low Risk of Heart Disease")

st.write("**Prediction Probability:**")
st.write(f"Low Risk: {prediction_proba[0]:.2f} | High Risk: {prediction_proba[1]:.2f}")

# -------------------------------
# Evaluation Section (Optional)
# -------------------------------
st.subheader("📊 Model Performance (on Test Data)")

# Load your test dataset (same preprocessing as training)
# Replace with your own test dataset
try:
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")

    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)

    # ROC Curvecls
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

except:
    st.warning("⚠️ No test dataset found. Upload X_test.pkl and y_test.pkl to show evaluation metrics.")
