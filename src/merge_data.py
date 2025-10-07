import pandas as pd

def merge_data(datasets, weather_df=None):
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

    if weather_df is not None:
        df = df.merge(weather_df, on=["year", "round"], how="left")

    df = df.dropna(subset=["postitionOrder"])
    df["positionOrder"] = df["positionOrder"].astype(int)

    return df
    