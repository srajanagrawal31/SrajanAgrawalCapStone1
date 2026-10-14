#uvicorn api.main:app --reload
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="Fraud Detection API",
    version="1.0"
)


# =========================
# Load Model
# =========================

model = joblib.load("./model/fraud_detector.pkl")

print("Model Loaded Successfully!")


# =========================
# Input Schema
# =========================

class FraudInput(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


# =========================
# Home Route
# =========================

@app.get("/")
def home():

    return {
        "message": "Fraud Detection API"
    }


# =========================
# Prediction Route
# =========================

@app.post("/predict")
def predict(data: FraudInput):

    input_dict = data.model_dump()

    # =========================
    # Feature Engineering
    # =========================

    input_dict["orig_balance_diff"] = (
        input_dict["oldbalanceOrg"]
        - input_dict["newbalanceOrig"]
    )

    input_dict["dest_balance_diff"] = (
        input_dict["newbalanceDest"]
        - input_dict["oldbalanceDest"]
    )

    input_dict["orig_error"] = (
        input_dict["oldbalanceOrg"]
        - input_dict["amount"]
        - input_dict["newbalanceOrig"]
    )

    input_dict["dest_error"] = (
        input_dict["oldbalanceDest"]
        + input_dict["amount"]
        - input_dict["newbalanceDest"]
    )

    # =========================
    # Convert To DataFrame
    # =========================

    input_df = pd.DataFrame([input_dict])

    # =========================
    # Prediction
    # =========================

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": round(float(probability), 4)
    }