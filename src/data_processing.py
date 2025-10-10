#@id:chat.disableAIFeatures

import fastf1
import pandas as pd
from load_data import load_raw_data
from weather_data import get_weather_data
from merge_data import merge_data
from features import compute_features
import datetime
import os

# Optionally, install matplotlib for plotting and seaborn for styling:
# pip install matplotlib seaborn

PROCESSED_PATH = "../data/processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)
print("directory made/exists")


def get_latest_seasons():
    current_year = datetime.datetime.now().year
    try:
        latest_schedule = fastf1.get_event_schedule(current_year)
        print(f"latest available season from Fastf1: {current_year}")
    except Exception:
        current_year -= 1
        print(f"falling back to {current_year}")

    # valid seasons
    train_seasons = list(range(2018, current_year - 3))     # training data, consistent rules 
    test_seasons = list(range(current_year - 3, current_year - 1))      # testing data, still constistent rules
    predict_seasons = list(range(current_year, current_year + 1))   # prediction data, still still consistent rules (2025 data not available yet)

    return train_seasons, test_seasons, predict_seasons

TRAIN_SEASONS, TEST_SEASONS, PREDICT_SEASONS = get_latest_seasons()

def split_and_save(df):
    train_df = df[df["year"].isin(TRAIN_SEASONS)]
    test_df = df[df["year"].isin(TEST_SEASONS)]
    predict_df = df[df["year"].isin(PREDICT_SEASONS)]

    train_df = compute_features(train_df)
    test_df = compute_features(test_df, exclude_current_race=True)
    predict_df = compute_features(predict_df, only_past_data=True)

    train_df.to_csv(os.path.join(PROCESSED_PATH, "train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_PATH, "test.csv"), index=False)
    predict_df.to_csv(os.path.join(PROCESSED_PATH, "predict.csv"), index=False)

    print(f"proccessed data saved to {PROCESSED_PATH}")


if __name__ == "__main__":
    print("loading raw data")
    datasets = load_raw_data()
    print("fetching weather data")
    weather_df = get_weather_data(TRAIN_SEASONS + TEST_SEASONS + PREDICT_SEASONS)
    print("merging all the data")
    df = merge_data(datasets, weather_df)   #df_2025
    print("splitting and saving in designated files")
    split_and_save(df)

    #df_2025 = get_2025_data()




# def get_2025_data():
#     rows = []
#     for season in PREDICT_SEASONS:
#         try:
#             races_list = fastf1.ergast.get_season(season).get_races()
#         except Exception as e:
#             print(f"No data for season", e)

#         for race in races_list:
#             race_id = race["raceId"]
#             round_num = race["round"]
#             race_name = race["raceName"]
#             circuit = race["Circuit"]["circuitName"]

#             for driver in race["Results"]:
#                 driver_id = driver["Driver"]["driverId"]
#                 constructor_id = driver["Constructor"]["constructorId"]
#                 qualifying_pos = driver.get("grid", None)

#                 rows.append({
#                     "raceId" : race_id,
#                     "round" : round_num,
#                     "raceName" : race_name,
#                     "circuit" : circuit,
#                     "driverId" : driver_id,
#                     "constructorId" : constructor_id,
#                     "qualifying_pos" : qualifying_pos
#                 })
#     df_2025 = pd.DataFrame(rows)
#     return df_2025
    