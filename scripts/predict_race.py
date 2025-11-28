from src.model.predict import predict_single_race

def main():
    season = int(2024)        #(input("Enter race season (e.g. 2024): ").strip())
    rnd = int(input("Enter race round number (e.g. 1): ").strip())
    preds_df = predict_single_race(season, rnd)

    print(f"\nPredicted finishing order for season {season}, round {rnd}:\n")
    for _, row in preds_df.iterrows():
        print(f"{row['pred_rank']:2d}. {row['driver_code']} ({row['team_name']}) - score: {row['pred_pos']:.3f}")

if __name__ == "__main__":
    main()
