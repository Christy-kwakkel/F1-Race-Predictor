import lightgbm as lgb
import pandas as pd
import unicodedata
from config import FEATURES_PATH, MODEL_PATH, PREDICT_SEASONS

def _load_base():
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in ["season", "round", "driver_code", "team_name", "target_pos"]]
    model = lgb.Booster(model_file=MODEL_PATH)
    return df, feature_cols, model

def _normalize(text: str) -> str:
    """Normalize text for fuzzy matching (accents, case, punctuation)."""
    text = str(text).lower()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return text.replace("-", " ").replace("_", " ").strip()

def resolve_circuit_query(query: str, season: int | None = None) -> pd.DataFrame:
    """Resolve 'monaco', 'yas marina', 'spa', etc. → matching races."""
    df = pd.read_parquet(FEATURES_PATH)
    
    # Filter by season if provided
    if season is not None:
        df = df[df["season"] == season]
    
    # Get unique races with circuit info
    races = (
        df[["season", "round", "name_circuit", "location", "country", "circuit_key"]]
        .drop_duplicates(subset=["season", "round"])
        .copy()
    )
    
    # Normalize all text fields
    races["norm_circuit"] = races["name_circuit"].fillna("").map(_normalize)
    races["norm_location"] = races["location"].fillna("").map(_normalize)
    races["norm_country"] = races["country"].fillna("").map(_normalize)
    races["norm_key"] = races["circuit_key"].fillna("").map(_normalize)
    
    q = _normalize(query)
    
    # Fuzzy match across all fields
    mask = (
        races["norm_circuit"].str.contains(q, na=False) |
        races["norm_location"].str.contains(q, na=False) |
        races["norm_country"].str.contains(q, na=False) |
        races["norm_key"].str.contains(q, na=False)
    )
    
    return races[mask].sort_values(["season", "round"])


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
    df,feature_cols, model = _load_base()
    race_df = df[(df["season"] == season)].copy()
    if race_df.empty:
        raise ValueError(f"No features found foder season {season}")
    
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
