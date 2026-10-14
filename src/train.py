# =========================
# Import Packages
# =========================

import pandas as pd
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# =========================
# Read Data
# =========================

fraud_df = pd.read_csv("./data/onlinefraud.csv")

print("Dataset Loaded Successfully!")


# =========================
# Data Cleaning
# =========================

fraud_df.drop(
    columns=[
        "isFlaggedFraud",
        "nameOrig",
        "nameDest"
    ],
    inplace=True
)

print("Columns Dropped Successfully!")


# =========================
# Feature Engineering
# =========================

fraud_df["orig_balance_diff"] = (
    fraud_df["oldbalanceOrg"]
    - fraud_df["newbalanceOrig"]
)

fraud_df["dest_balance_diff"] = (
    fraud_df["newbalanceDest"]
    - fraud_df["oldbalanceDest"]
)

fraud_df["orig_error"] = (
    fraud_df["oldbalanceOrg"]
    - fraud_df["amount"]
    - fraud_df["newbalanceOrig"]
)

fraud_df["dest_error"] = (
    fraud_df["oldbalanceDest"]
    + fraud_df["amount"]
    - fraud_df["newbalanceDest"]
)

print("Feature Engineering Completed!")


# =========================
# Features & Target
# =========================

X = fraud_df.drop("isFraud", axis=1)
y = fraud_df["isFraud"]


# =========================
# Numerical & Categorical Columns
# =========================

num_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

cat_cols = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()


# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=10,
    stratify=y
)

print("Train Test Split Completed!")


# =========================
# Create 500K Stratified Sample
# =========================

X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train,
    y_train,
    train_size=500000,
    stratify=y_train,
    random_state=10
)

print("500K Stratified Sample Created!")


# =========================
# Save Train/Test Data
# =========================

X_train.to_csv("./data/X_train.csv", index=False)
X_test.to_csv("./data/X_test.csv", index=False)

y_train.to_csv("./data/y_train.csv", index=False)
y_test.to_csv("./data/y_test.csv", index=False)

print("Train/Test Files Saved Successfully!")


# =========================
# Preprocessing Pipeline
# =========================

num_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

cat_pipeline = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ]
)

print("Preprocessing Pipeline Created!")


# =========================
# Random Forest Pipeline
# =========================

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                class_weight="balanced",
                random_state=10,
                n_jobs=-1
            )
        )
    ]
)

print("Random Forest Pipeline Created!")


# =========================
# Model Training
# =========================

print("Training Model...")

rf_pipeline.fit(
    X_train_sample,
    y_train_sample
)

print("Model Training Completed!")


# =========================
# Save Model
# =========================

joblib.dump(
    rf_pipeline,
    "./model/fraud_detector.pkl"
)

print("Model Saved Successfully!")
print("Location: ./model/fraud_detector.pkl")