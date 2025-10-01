#@id:chat.disableAIFeatures

import pandas as pd
import fastf1
import os

# Optionally, install matplotlib for plotting and seaborn for styling:
# pip install matplotlib seaborn

# get the Jolpica-F1 API
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"

RAW_PATH = "data/raw/kaggle_F1_data"
PROCESSED_PATH = "data/processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)


# valid seasons
TRAIN_SEASONS = list(range(2010, 2022))     # training data, consistent rules 
TEST_SEASONS = list(range(2022, 2024))      # testing data, still constistent rules
PREDICT_SEASONS = list(range(2024, 2025))   # prediction data, still still consistent rules



def load_raw_data():
    # load datasets and return datasets
    circuits = pd.read_csv(RAW_PATH + "/circuits.csv")
    constructor_results = pd.read_csv(RAW_PATH + "/constructor_results.csv")
    contstructor_standings = pd.read_csv(RAW_PATH + "/constructor_standings.csv")
    constructors = pd.read_csv(RAW_PATH + "/constructors.csv")
    driver_standings = pd.read_csv(RAW_PATH + "/driver_standings.csv")
    drivers = pd.read_csv(RAW_PATH + "/drivers.csv")
    lap_times = pd.read_csv(RAW_PATH + "/lap_times.csv")
    pit_stops = pd.read_csv(RAW_PATH + "/pit_stops.csv")
    qualifying = pd.read_csv(RAW_PATH + "/qualifying.csv")
    races = pd.read_csv(RAW_PATH + "/races.csv")
    results = pd.read_csv(RAW_PATH + "/results.csv")
    seasons = pd.read_csv(RAW_PATH + "/seasons.csv")
    sprint_results = pd.read_csv(RAW_PATH + "/sprint_results.csv")
    status = pd.read_csv(RAW_PATH + "/status.csv")

    return {
        "circuits": circuits,
        "constructor_results": constructor_results,
        "constructor_standings": contstructor_standings,
        "constructors": constructors,
        "driver_standings": driver_standings,
        "drivers": drivers,
        "lap_times": lap_times,
        "pit_stops": pit_stops,
        "qualifying": qualifying,
        "races": races,
        "results": results,
        "seasons": seasons,
        "sprint_results": sprint_results,
        "status": status
    }

def merge_data(datasets):
    # merge datasets to create a final datasets for modeling
    results = datasets["results"]
    races = datasets["races"]
    circuits = datasets["circuits"]
    drivers = datasets["drivers"]
    constructors = datasets["constructors"]
    qualifying = datasets["qualifying"]

    df = results.merge(races[["raceId", "year", "round", "circuitId"]], on="raceId", how="left")
    df = df.merge(circuits[["circuitId", "name", "location", "country"]], on="circuitId", how="left")
    df = df.merge(drivers[["driverId", "surname", "dob", "nationality"]], on="driverId", how="left")
    df = df.merge(constructors[["constructorId", "name"]], on="constructorId", how="left")
    df = df.merge(qualifying[["raceId", "driverId", "q1", "q2", "q3"]], on=["raceId", "driverId"], how="left")  

    # clean positions
    df = df.dropna(subset=["positionOrder"])
    df["positionOrder"] = df["positionOrder"].astype(int)
    return df


def compute_features(df, exclude_current_race=False, only_past_data=False):
    df = df.sort_values(by=["driverId", "year", "round"])

    # Form average finishing position of the current season, if there have been at least 3 races in the season

    # if less than 3 races, look at the last 3 races of a driver

    driver_forms = []
    for driver, group in df.groupby("driverId"):
        group = group.sort_values(by=["year", "round"])
        form_values = []

        for i, row in group.iterrows():
            # all races of the same season before the current race
            past_season = group[(group["year"] == row["year"]) & (group["round"] < row["round"])]
                                
            if len(past_season) >= 3:
                form = past_season["positionOrder"].mean()
            else:
                past_all = group[group["round"] < row["round"]]
                form = past_all["positionOrder"].tail(3).mean() if len(past_all) > 0 else None
            
            form_values.append(form)

        group["driver_form"] = form_values
        driver_forms.append(group)
        
    df = pd.concat(driver_forms)

    df["driver_form"] = df["driver_form"].fillna(df["positionOrder"].mean())


    constructor_forms = []
    for constructor, group in df.groupby("constructorId"):
        group = group.sort_values(by=["year", "round"])
        form_values = []

        for i, row in group.iterrows():
            past_season = group[(group["year"] == row["year"]) & (group["round"] < row["round"])]
            if len(past_season) >= 3:
                form = past_season["positionOrder"].mean()
            else:
                past_all = group[group["round"] < row["round"]]
                form = past_all["positionOrder"].tail(3).mean() if len(past_all) > 0 else None

            form_values.append(form)

        group["constructor_form"] = form_values
        constructor_forms.append(group)

    df = pd.concat(constructor_forms)

    return df



def get_2025_data():
    for season in PREDICT_SEASONS:
        races_list = fastf1.ergast.get_season(season).get_races()
        rows = []

        for race in races_list:
            race_id = race["raceId"]
            round_num = race["round"]
            race_name = race["raceName"]
            circuit = race["Circuit"]["circuitName"]

            for driver in race["Results"]:
                driver_id = driver["Driver"]["driverId"]
                constructor_id = driver["Constructor"]["constructorId"]
                qualifying_pos = driver.get("grid", None)

                rows.append({
                    "raceId" : race_id,
                    "round" : round_num,
                    "raceName" : race_name,
                    "circuit" : circuit,
                    "driverId" : driver_id,
                    "constructorId" : constructor_id,
                    "qualifying_pos" : qualifying_pos
                })
    df_2025 = pd.DataFrame(rows)
    return df_2025

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
    datasets = load_raw_data()
    df = merge_data(datasets)
    split_and_save(df)



    