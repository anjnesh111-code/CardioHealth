import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =============================
# LOAD DATASET
# =============================
data = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

# Features & Target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train RandomForest
model = RandomForestClassifier(random_state=42, n_estimators=200)
model.fit(X_train, y_train)

# Save & reload (to ensure compatibility)
joblib.dump(model, "modelRf.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")
model = joblib.load("modelRf.pkl")

# =============================
# STREAMLIT DASHBOARD
# =============================
st.set_page_config(page_title="CardioRisk-Lite", layout="wide")
st.title("💖 CardioRisk-Lite Dashboard")
st.write("Predicting heart disease risk using a Random Forest Classifier.")

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

    data = {
        "age": age,
        "sex": sex,
        "chest pain type": cp,
        "resting bp s": trestbps,
        "cholesterol": chol,
        "fasting blood sugar": fbs,
        "resting ecg": restecg,
        "max heart rate": thalach,
        "exercise angina": exang,
        "oldpeak": oldpeak,
        "ST slope": slope,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Prediction
try:
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader("Prediction")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write("**Prediction Probability:**")
    st.write(f"Low Risk: {prediction_proba[0]:.2f} | High Risk: {prediction_proba[1]:.2f}")

except Exception as e:
    st.warning(f"⚠️ Could not make prediction: {e}")

# Model performance
try:
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    st.subheader("📊 Model Performance (on Test Data)")
    st.write("Confusion Matrix:")
    st.write(cm)

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

except Exception as e:
    st.warning(f"⚠️ No test dataset found or error: {e}")
