import datetime as dt
import pandas as pd
from config import TRAIN_SEASONS, TEST_SEASONS, PREDICT_SEASONS, FEATURES_PATH
from .fastf1_loader import build_fastf1_base
from .kaggle_loader import load_kaggle_core
from .weather_api import fetch_race_weather
from ..features.feature_engineering import add_driver_form_features, add_team_form_features, finalize_features

def maybe_patch_weather_with_api(df):
    kaggle = load_kaggle_core()
    races = kaggle["races"]
    circuits = kaggle["circuits"]
    races = races.merge(circuits, on="circuitId", how="left", suffixes=("", "_circuit"))
    races["race_key"] = list(zip(races["year"], races["round"]))
    race_meta = races.set_index("race_key")

    patched = []
    for (season, rnd), grp in df.groupby(["season", "round"]):
        g = grp.copy()
        if g["air_temp_mean"].isna().all():
            key = (season, rnd)
            if key not in race_meta.index:
                patched.append(g)
                continue
            row = race_meta.loc[key]
            lat = row["lat"]
            lon = row["lng"]
            date = dt.datetime.strptime(row["date"], "%Y-%m-%d").date()
            feats = fetch_race_weather(lat, lon, date)
            if feats:
                g["air_temp_mean"] = feats["air_temp_mean_ext"]
                g["humidity_mean"] = feats["humidity_mean_ext"]
                g["pressure_mean"] = feats["pressure_mean_ext"]
                g["wind_speed_mean"] = feats["wind_speed_mean_ext"]
                g["rain_flag"] = float(feats["rain_flag_ext"])
        patched.append(g)
    return pd.concat(patched, ignore_index=True)


def attach_circuit_info(df: pd.DataFrame) -> pd.DataFrame:
    kaggle = load_kaggle_core()
    races = kaggle["races"]
    circuits = kaggle["circuits"]

    races = races.merge(
        circuits[["circuitId", "name", "location", "country"]],
        on="circuitId",
        how="left",
        suffixes=("", "_circuit")
    )    

    races["race_key"] = list(zip(races["year"], races["round"]))
    race_meta = races.set_index("race_key")

    df = df.copy()
    df["race_key"] = list(zip(df["season"], df["round"]))

    df = df.merge(
        race_meta[["circuitId", "name_circuit", "location", "country"]],
        on="race_key",
        how="left"
    )

    df = df.drop(columns=["race_key"])

    df["circuit_key"] = (
        df["name_circuit"]
        .fillna("")
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


def build_and_save_features():
    all_seasons = sorted(set(TRAIN_SEASONS + TEST_SEASONS + PREDICT_SEASONS))
    df = build_fastf1_base(all_seasons)      
    
    #df = attach_circuit_info(df)
    df = maybe_patch_weather_with_api(df)
    df = add_driver_form_features(df)
    df = add_team_form_features(df)
    X, y, meta = finalize_features(df)

    # Drop duplicate id columns from X; keep them only from meta
    X = X.drop(columns=["season", "round"], errors="ignore")

    out = pd.concat([meta, X, y.rename("target_pos")], axis=1)
    out.to_parquet(FEATURES_PATH, index=False)
    print(f"Saved features to {FEATURES_PATH}")

