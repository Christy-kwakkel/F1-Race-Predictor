import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import kendalltau
from config import FEATURES_PATH, TEST_SEASONS, MODEL_PATH


def load_model():
    """Load trained Linear Regression model."""
    model_path_pkl = Path(MODEL_PATH) / "model_linear.pkl"
    with open(model_path_pkl, 'rb') as f:
        return pickle.load(f)


def evaluate_order_metric():
    """Evaluate Linear Regression model using Kendall Tau rank correlation."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating Linear Regression model on test set...")
    print(f"{'='*60}\n")
    
    # Load model
    model = load_model()
    
    # Load processed features dataset
    df = pd.read_parquet(FEATURES_PATH)
    
    # Get test data
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    feature_cols = [c for c in df.columns if c not in ["season", "round", "driver_code", "team_name", "target_pos"]]
    
    X_test = test[feature_cols]
    y_test = test["target_pos"]
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Create predictions dataframe
    test["pred_pos"] = y_pred
    
    # True finishing positions for test seasons
    truth = df[df["season"].isin(TEST_SEASONS)][["season", "round", "driver_code", "target_pos"]]
    
    # Merge predicted and true positions
    merged = test[["season", "round", "driver_code", "pred_pos"]].merge(
        truth, on=["season", "round", "driver_code"], how="left"
    )
    
    scores = []
    race_details = []
    
    # Calculate Kendall tau for each race
    for (season, rnd), grp in merged.groupby(["season", "round"]):
        # Skip races with missing ground truth
        if grp["target_pos"].isna().any():
            continue
        
        tau, p_val = kendalltau(grp["target_pos"], grp["pred_pos"])
        scores.append(tau)
        race_details.append({
            "season": season,
            "round": rnd,
            "tau": tau,
            "p_value": p_val
        })
    
    if scores:
        mean_tau = sum(scores) / len(scores)
        median_tau = sorted(scores)[len(scores)//2]
        min_tau = min(scores)
        max_tau = max(scores)
        
        print(f"{'='*60}")
        print(f"LINEAR REGRESSION - KENDALL TAU RESULTS")
        print(f"{'='*60}")
        print(f"Mean Kendall Tau:   {mean_tau:.3f}")
        print(f"Median Kendall Tau: {median_tau:.3f}")
        print(f"Races evaluated:    {len(scores)}")
        print(f"Min Tau:            {min_tau:.3f} (worst race)")
        print(f"Max Tau:            {max_tau:.3f} (best race)")
        print(f"{'='*60}\n")
        
        # Interpretation
        if mean_tau > 0.5:
            print("STRONG rank correlation (>0.5) - Model orders finishers well")
        elif mean_tau > 0.3:
            print("MODERATE rank correlation (0.3-0.5) - Some ordering errors")
        else:
            print("WEAK rank correlation (<0.3) - Poor ordering performance")
        
        # Show worst 3 races
        race_df = pd.DataFrame(race_details).sort_values("tau")
        print(f"\nWorst 3 races (lowest Kendall Tau):")
        print(race_df.head(3)[["season", "round", "tau"]].to_string(index=False))
        
        # Show best 3 races
        print(f"\nBest 3 races (highest Kendall Tau):")
        print(race_df.tail(3)[["season", "round", "tau"]].to_string(index=False))
        
    else:
        print("No valid test races found for evaluation.")
