import pandas as pd
import os

RAW_PATH = "data/raw/kaggle_F1_data"
PROCESSED_PATH = "data/processed/final_dataset.csv"

def process_data():
    # load datasets
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



    