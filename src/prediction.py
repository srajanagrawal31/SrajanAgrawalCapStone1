import joblib
import pandas as pd


# =========================
# Load Model
# =========================

model = joblib.load("./model/fraud_detector.pkl")


def predict_fraud(data):

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    return prediction, probability


if __name__ == "__main__":

    X_test = pd.read_csv("./data/X_test.csv")

    prediction, probability = predict_fraud(
        X_test.iloc[[0]]
    )

    print("Prediction:", prediction)
    print("Fraud Probability:", probability)