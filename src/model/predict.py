import pickle
import pandas as pd
from pathlib import Path
from config import FEATURES_PATH, MODEL_PATH, PREDICT_SEASONS


def _load_base():
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in ["season", "round", "driver_code", "team_name", "target_pos"]]
    
    # Load Linear Regression model
    model_path_pkl = Path(MODEL_PATH) / "model_linear.pkl"
    with open(model_path_pkl, 'rb') as f:
        model = pickle.load(f)
    
    return df, feature_cols, model


def predict_single_race(season: int, round_num: int) -> pd.DataFrame:
    df, feature_cols, model = _load_base()
    race_df = df[(df["season"] == season) & (df["round"] == round_num)].copy()
    if race_df.empty:
        raise ValueError(f"No features found for season {season}, round {round_num}")

    preds = model.predict(race_df[feature_cols])
    race_df["pred_pos"] = preds
    race_df = race_df.sort_values("pred_pos").copy()
    race_df["pred_rank"] = range(1, len(race_df) + 1)
    return race_df[["season", "round", "driver_code", "team_name", "pred_pos", "pred_rank"]]


def predict_single_season(season: int) -> pd.DataFrame:
    df, feature_cols, model = _load_base()
    race_df = df[(df["season"] == season)].copy()
    if race_df.empty:
        raise ValueError(f"No features found for season {season}")
    
    preds = model.predict(race_df[feature_cols])
    race_df["pred_pos"] = preds
    race_df = race_df.sort_values("pred_pos").copy()
    race_df["pred_rank"] = range(1, len(race_df) + 1)
    return race_df[["season", "round", "driver_code", "team_name", "pred_pos", "pred_rank"]]


def predict_all_for_seasons(seasons=None) -> pd.DataFrame:
    df, feature_cols, model = _load_base()
    if seasons is None:
        seasons = PREDICT_SEASONS
    pred_df = df[df["season"].isin(seasons)].copy()
    preds = model.predict(pred_df[feature_cols])
    pred_df["pred_pos"] = preds

    race_orders = []
    for (season, rnd), grp in pred_df.groupby(["season", "round"]):
        g = grp.sort_values("pred_pos").copy()
        g["pred_rank"] = range(1, len(g) + 1)
        race_orders.append(g)

    final = pd.concat(race_orders, ignore_index=True)
    return final[["season", "round", "driver_code", "team_name", "pred_pos", "pred_rank"]]
