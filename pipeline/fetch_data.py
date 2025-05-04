import pandas as pd

DATA_URL = "https://data.cityofnewyork.us/resource/t29m-gskq.csv?$limit=1000"


def fetch_data(output_path="data/raw/nyc_taxi.csv"):
    df = pd.read_csv(DATA_URL)
    df.to_csv(output_path, index=False)
    return df