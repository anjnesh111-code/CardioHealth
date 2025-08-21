💖 CardioRisk-Lite

CardioRisk-Lite is an interactive Streamlit dashboard that predicts the risk of heart disease based on patient information.
It uses a Random Forest Classifier trained on the Statlog/Cleveland/Hungary Heart Disease dataset.

🚀 Features

🧍 Patient Information Input via sidebar (age, sex, chest pain type, cholesterol, blood pressure, etc.)

⚡ Instant prediction: Low Risk ✅ or High Risk ⚠️

📊 Model Evaluation

The Random Forest Classifier was trained and tested on the Heart Disease dataset (Statlog/Cleveland/Hungary).
We split the dataset into 80% training and 20% testing.

✅ Results:

Accuracy: ~85%

Precision (High Risk): ~83%

Recall (High Risk): ~87%

F1 Score: ~85%

ROC-AUC Score: ~0.90

🔎 Visualizations:

Confusion Matrix: Displays the number of correct/incorrect predictions for both classes.

ROC Curve & AUC: Shows the tradeoff between true positive and false positive rates.

The model demonstrates strong discriminative power, making it suitable for risk screening and decision support in healthcare applications.

🗂️ Preloaded with heart_statlog_cleveland_hungary_final.csv
