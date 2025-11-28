import pandas as pd
from scipy.stats import kendalltau

from config import FEATURES_PATH, TEST_SEASONS
from .predict import predict_all_for_seasons


def evaluate_order_metric():
    # Load processed features dataset
    df = pd.read_parquet(FEATURES_PATH)
    
    # Get predictions on test seasons
    model_preds = predict_all_for_seasons(TEST_SEASONS)

    # True finishing positions for test seasons
    truth = df[df["season"].isin(TEST_SEASONS)][["season", "round", "driver_code", "target_pos"]]
    
    # Merge predicted and true positions
    merged = model_preds.merge(truth, on=["season", "round", "driver_code"], how="left")
    
    scores = []
    # Calculate Kendall tau for each race
    for (season, rnd), grp in merged.groupby(["season", "round"]):
        # Skip races with missing ground truth
        if grp["target_pos"].isna().any():
            continue
        tau, _ = kendalltau(grp["target_pos"], grp["pred_pos"])
        scores.append(tau)
    
    if scores:
        mean_tau = sum(scores) / len(scores)
        print(f"Mean Kendall Tau rank correlation on test seasons: {mean_tau:.3f}")
    else:
        print("No valid test races found for evaluation.")
