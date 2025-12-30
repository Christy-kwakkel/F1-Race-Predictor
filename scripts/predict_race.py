from src.model.predict import predict_single_race, resolve_gp_name
from pathlib import Path
import pandas as pd
from config import GP_NAME_MAPPING

def predict_race():
    print("F1 Race Predictor")

    gp_input = input("Enter GP name (e.g. 'Monaco Grand Prix', 'Abu Dhabi Grand Prix'): ").strip()
    
    try:
        season, round_num = resolve_gp_name(gp_input)
        
        # Predict for all seasons at this GP (or specify season)
        season_input = input("Enter season (e.g. 2024) or press Enter for latest: ").strip()
        if season_input:
            season = int(season_input)
        else:
            season = 2024  # Default to latest
            
        preds_df = predict_single_race(season, round_num)
        
        print(f"\n {gp_input.upper()} ({season} R{round_num}) Predictions:\n")
        for _, row in preds_df.iterrows():
            print(f"{row['pred_rank']:2d}. {row['driver_code']:>4} ({row['team_name']:<20}) - score: {row['pred_pos']:.3f}")
        
    except ValueError as e:
        print(f"{e}")
        print("Available GPs:")
        for gp in GP_NAME_MAPPING.keys():
            print(f"  • {gp}")

if __name__ == "__main__":
    predict_race()
