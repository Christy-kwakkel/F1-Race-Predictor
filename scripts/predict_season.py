from src.model.predict import predict_single_season
from pathlib import Path
import pandas as pd
print("in file")

def main():
    season = int(input("Enter race season (e.g. 2024): ").strip())
    preds_df = predict_single_season(season)
    print(f"\nPredicted finishing order for season {season}:\n")

    # Sorting
    preds_df = preds_df.sort_values(["season", "round", "pred_pos", ]).reset_index(drop=True)
    preds_df["pred_rank"] = preds_df.groupby(["season", "round"]).cumcount() + 1  

    # Saving
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "2024_predictions.csv"
    preds_df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")

    # Printing results
    print("\nTop predictions by race:")
    print(preds_df.groupby("round").head(3)[["round", "driver_code", "pred_pos", "pred_rank"]].to_string(index=False))

if __name__ == "__main__":
    main()
