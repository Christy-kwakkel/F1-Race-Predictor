import pandas as pd
import joblib
import numpy as np

def predict_race(year, location):
    model, scaler = joblib.load("../data/processed/GBR_model.pkl")
    df = pd.read_csv("../data/processed/predict.csv")
    race_df = df[(df["year"] == year) & (df["location"].str.lower() == location.lower())]

    features = ["driver_form", "constructor_form", "grid", "is_post2022"]
    preds = model.predict(scaler.transfrom(race_df[features]))
    race_df["predicted_position"] = np.round(preds)
    race_df = race_df.sort_values("predicted_position").reset_index(drop=True)
    print(race_df[["surname", "name_y", "predicted_position"]])


if __name__ == "__main__":
    predict_race(2025, "Monaco")
