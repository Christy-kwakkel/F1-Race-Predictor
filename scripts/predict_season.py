from src.model.predict import predict_single_season
from pathlib import Path
import pandas as pd
from config import PROCESSED_DIR
print("in file")

def main():
    season = int(input("Enter race season (e.g. 2024): ").strip())
    preds_df = predict_single_season(season)
    print(f"\nPredicted finishing order for season {season}:\n")

    # Sorting
    preds_df = preds_df.sort_values(["season", "round", "pred_pos", ]).reset_index(drop=True)
    preds_df["pred_rank"] = preds_df.groupby(["season", "round"]).cumcount() + 1  

    # Saving
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED_DIR / "2024_predictions.csv"
    preds_df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")

    # Printing results
    print("\nTop predictions by race:")
    print(preds_df.groupby("round").head(3)[["round", "driver_code", "pred_pos", "pred_rank"]].to_string(index=False))

if __name__ == "__main__":
    main()
