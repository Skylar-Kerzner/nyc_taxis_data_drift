import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from model.predict import load_model
from pipeline.retrain import preprocess
from datetime import datetime
import os

METRICS_PATH = "data/metrics.csv"


def evaluate():
    df = pd.read_csv("data/raw/nyc_taxi.csv")
    X, y = preprocess(df)
    model = load_model()
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    today = datetime.today().strftime('%Y-%m-%d')

    new_row = pd.DataFrame([[today, mae, r2, len(df)]], columns=["date", "MAE", "R2", "n"])
    if os.path.exists(METRICS_PATH):
        old = pd.read_csv(METRICS_PATH)
        df_out = pd.concat([old, new_row], ignore_index=True)
    else:
        df_out = new_row

    df_out.to_csv(METRICS_PATH, index=False)
    print(f"Evaluation complete. MAE: {mae:.2f}, R2: {r2:.2f}")