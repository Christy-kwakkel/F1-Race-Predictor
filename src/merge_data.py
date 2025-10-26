import pandas as pd

def merge_data(datasets, weather_df=None):
    results = datasets["results"]
    races = datasets["races"]
    circuits = datasets["circuits"]
    drivers = datasets["drivers"]
    constructors = datasets["constructors"]
    qualifying = datasets["qualifying"]

    df = (results
        .merge(races[["raceId", "year", "round", "circuitId"]], on="raceId", how="left")
        .merge(circuits[["circuitId", "name", "location", "country"]], on="circuitId", how="left")
        .merge(drivers[["driverId", "surname", "dob", "nationality"]], on="driverId", how="left")
        .merge(constructors[["constructorId", "name"]], on="constructorId", how="left")
        .merge(qualifying[["raceId", "driverId", "q1", "q2", "q3"]], on=["raceId", "driverId"], how="left")
    )

    if weather_df is not None and not weather_df.empty:
        df = df.merge(
            weather_df, 
            on=["year", "round"], 
            how="left"
        )

    df = df.dropna(subset=["positionOrder"])
    df["positionOrder"] = df["positionOrder"].astype(int)

    # df["is_post22"] = (df["year"] >= 2022).astype(int)
    # df["year_norm"] = df["year"] - df["year"].min()

    return df
    