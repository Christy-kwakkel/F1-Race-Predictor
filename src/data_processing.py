#@id:chat.disableAIFeatures

import pandas as pd
import fastf1
import os

# Optionally, install matplotlib for plotting and seaborn for styling:
# pip install matplotlib seaborn

# get the Jolpica-F1 API
fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"

RAW_PATH = "data/raw/kaggle_F1_data"
PROCESSED_PATH = "data/processed/final_dataset.csv"

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

if __name__ == "__main__":
    print("Data processing complete. Processed data saved to", PROCESSED_PATH)



    