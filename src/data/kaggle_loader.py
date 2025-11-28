import pandas as pd
from config import (
    KAGGLE_RACES_CSV, KAGGLE_RESULTS_CSV, KAGGLE_DRIVERS_CSV,
    KAGGLE_CONSTRUCTORS_CSV, KAGGLE_QUALI_CSV, KAGGLE_CIRCUITS_CSV
)

def load_kaggle_core():
    races = pd.read_csv(KAGGLE_RACES_CSV)
    results = pd.read_csv(KAGGLE_RESULTS_CSV)
    drivers = pd.read_csv(KAGGLE_DRIVERS_CSV)
    constructors = pd.read_csv(KAGGLE_CONSTRUCTORS_CSV)
    quali = pd.read_csv(KAGGLE_QUALI_CSV)
    circuits = pd.read_csv(KAGGLE_CIRCUITS_CSV)
    return dict(
        races=races,
        results=results,
        drivers=drivers,
        constructors=constructors,
        quali=quali,
        circuits=circuits,
    )
