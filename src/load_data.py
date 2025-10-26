import os 
import pandas as pd


RAW_PATH = "C:/Users/chris/OneDrive/Desktop/Personal Project year 2/Personal-Project/data/raw/kaggle_F1_data"
os.makedirs(RAW_PATH, exist_ok=True)
print("directory made/exists")

def load_raw_data():
    files = [
        "circuits", "constructor_results", "constructor_standings", "constructors",
        "driver_standings", "drivers", "lap_times", "pit_stops", "qualifying",
        "races", "results", "seasons", "sprint_results", "status"
    ]
    return {f: pd.read_csv(os.path.join(RAW_PATH, f"{f}.csv")) for f in files}

    # datasets = {}
    # for file in files:
    #     path = os.path.join(RAW_PATH, f"{file}.csv")
    #     datasets[file] = pd.read_csv(path)
    # return datasets


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