import pandas as pd
import numpy as np

def add_driver_form_features(df):
    # Ensure sorted by driver, season, round
    df = df.sort_values(["driver_code", "season", "round"]).copy()

    # Convert finishing position to numeric, DNFs treated as 21
    df["pos_num"] = pd.to_numeric(df["position"], errors="coerce").fillna(21)
    df["pts_proxy"] = (21 - df["pos_num"]).clip(lower=0)

    # Only use historical seasons (up to last known race) for rolling avg 
    max_historical_season = df["season"].max()
    df_hist = df[df["season"] < max_historical_season]
    df_recent = df[df["season"] == max_historical_season]

    group = df.groupby("driver_code", group_keys=False)
    for window in [3, 5, 10]:
        df[f"drv_avg_pos_{window}"] = group["pos_num"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"drv_avg_pts_{window}"] = group["pts_proxy"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)

    return df


def add_team_form_features(df):
    df = df.sort_values(["team_name", "season", "round"]).copy()
    df["pos_num"] = pd.to_numeric(df["position"], errors="coerce").fillna(21)
    df["pts_proxy"] = (21 - df["pos_num"]).clip(lower=0)

    group = df.groupby("team_name", group_keys=False)
    for window in [3, 5, 10]:
        df[f"team_avg_pos_{window}"] = group["pos_num"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"team_avg_pts_{window}"] = group["pts_proxy"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)

    return df


def finalize_features(df):
    df["grid"] = df["grid"].fillna(20).clip(lower=1, upper=20)
    df["team_id_enc"] = df["team_name"].astype("category").cat.codes
    df["driver_id_enc"] = df["driver_code"].astype("category").cat.codes

    feature_cols = [
        "season", "round", "grid",
        "team_id_enc", "driver_id_enc",
        "air_temp_mean", "track_temp_mean",
        "wind_speed_mean", "humidity_mean", "pressure_mean", "rain_flag",
        "drv_avg_pos_3", "drv_avg_pos_5", "drv_avg_pos_10",
        "drv_avg_pts_3", "drv_avg_pts_5", "drv_avg_pts_10",
        "team_avg_pos_3", "team_avg_pos_5", "team_avg_pos_10",
        "team_avg_pts_3", "team_avg_pts_5", "team_avg_pts_10",
    ]

    df = df[df["pos_num"].notna()].copy()

    X = df[feature_cols].copy()
    y = df["pos_num"].copy()
    meta = df[["season", "round", "driver_code", "team_name"]].copy()
    return X, y, meta
