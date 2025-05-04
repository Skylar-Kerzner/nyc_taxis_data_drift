import joblib


def load_model(model_path="model/model.pkl"):
    return joblib.load(model_path)