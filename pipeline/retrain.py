import pandas as pd
from model.train import train_model
from model.drift import detect_drift
from pipeline.fetch_data import fetch_data
import os

RAW_PATH = "data/raw/nyc_taxi.csv"
PROCESSED_PATH = "data/processed/processed.csv"
MODEL_PATH = "model/model.pkl"


def preprocess(df):
    df = df.dropna(subset=["trip_distance", "fare_amount"])
    df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)]
    return df[["trip_distance"]], df["fare_amount"]


def retrain_if_needed():
    new_df = fetch_data(RAW_PATH)
    X_new, y_new = preprocess(new_df)

    if os.path.exists(PROCESSED_PATH):
        old_df = pd.read_csv(PROCESSED_PATH)
        X_old, y_old = preprocess(old_df)
        if not detect_drift(X_old["trip_distance"], X_new["trip_distance"]):
            print("No significant drift detected. Skipping retrain.")
            return

    train_model(X_new, y_new, MODEL_PATH)
    new_df.to_csv(PROCESSED_PATH, index=False)
    print("Model retrained and data updated.")