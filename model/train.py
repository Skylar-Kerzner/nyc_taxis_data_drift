import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def train_model(X, y, model_path="model/model.pkl"):
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model