from src.model.predict import predict_single_race, resolve_gp_name
import pandas as pd
from config import ACTUAL_CSV


def build_compare_table():
    print("F1 Race Prediction vs Actual")

    gp_input = input("Enter GP name (e.g. 'Monaco Grand Prix', 'Abu Dhabi Grand Prix'): ").strip()

    try:
        _, round_num = resolve_gp_name(gp_input)

        season_input = input("Enter season (e.g. 2024) or press Enter for 2024: ").strip()
        season = int(season_input) if season_input else 2024

        # Predictions
        preds_df = predict_single_race(season, round_num)
        preds_df["pred_rank"] = pd.to_numeric(preds_df["pred_rank"], errors="coerce")
        preds_df = preds_df.sort_values("pred_rank").reset_index(drop=True)

        # Actuals
        actuals = pd.read_csv(ACTUAL_CSV)
        actuals = actuals[(actuals["season"] == season) & (actuals["round"] == round_num)].copy()
        actuals["finishing_pos"] = pd.to_numeric(actuals["finishing_pos"], errors="coerce")
        actuals = actuals.sort_values(['finishing_pos']).reset_index(drop=True)

        # sort by finishing_pos
        # actuals = actuals.sort_values("finishing_pos")

        print(f"\n{gp_input.upper()} ({season} R{round_num})\n")
        print(f"{'Predicted':30s}  {'Actual':30s} {'Difference':20s}")
        print("-" * 80)

        # Align by position index
        max_len = max(len(preds_df), len(actuals))
        for i in range(max_len):
            # predicted row i (0-based)
            if i < len(preds_df):
                prow = preds_df.iloc[i]
                pred_str = f"{prow['pred_rank']} {prow['driver_code']:>4} ({prow['team_name']})"
                prow['pred_rank'] = int(prow['pred_rank'])
            else:
                pred_str = ""

            # actual row i
            if i < len(actuals):
                arow = actuals.iloc[i]
                # if you have `code` and team_name in actuals, adjust as needed
                actual_str = f"{(arow['finishing_pos']):>1} {arow['code'].upper():>4} ({arow['team_name']})"
                arow['finishing_pos'] = int(arow['finishing_pos'])
            else:
                actual_str = ""

            # difference row
            dif_str = ((prow['pred_rank']) - (arow['finishing_pos']) )

            print(f"{pred_str:30s}  {actual_str:30s} {dif_str:20s}") 

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    build_compare_table()
