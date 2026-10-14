# =========================
# Import Packages
# =========================

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

# =========================
# Load Model
# =========================

model = joblib.load("./model/fraud_detector.pkl")

print("Model Loaded Successfully!")

# =========================
# Load Test Data
# =========================

X_test = pd.read_csv("./data/X_test.csv")
y_test = pd.read_csv("./data/y_test.csv")

# Convert DataFrame to Series
y_test = y_test.squeeze()

print("Test Data Loaded Successfully!")

# =========================
# Prediction
# =========================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Prediction Completed!")

# =========================
# Accuracy Score
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print("ACCURACY SCORE")
print("="*50)

print(round(accuracy, 4))

# =========================
# Confusion Matrix
# =========================

print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)

print(confusion_matrix(y_test, y_pred))

# =========================
# Classification Report
# =========================

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)

print(classification_report(y_test, y_pred))

# =========================
# ROC-AUC Score
# =========================

roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*50)
print("ROC-AUC SCORE")
print("="*50)

print(round(roc_auc, 4))

# =========================
# PR-AUC Score
# =========================

pr_auc = average_precision_score(y_test, y_prob)

print("\n" + "="*50)
print("PR-AUC SCORE")
print("="*50)

print(round(pr_auc, 4))k