TRAIN_SEASONS = [2020, 2021]
TEST_SEASONS = [2022, 2023]
PREDICT_SEASONS = [2024]

FASTF1_CACHE_DIR = "data/external/fastf1_cache"

# Updated raw Kaggle csv path
KAGGLE_BASE = r"C:\Users\chris\OneDrive\Desktop\Personal Project year 2\trying\data\raw\kaggle_F1_data"
KAGGLE_RACES_CSV = f"{KAGGLE_BASE}/races.csv"
KAGGLE_RESULTS_CSV = f"{KAGGLE_BASE}/results.csv"
KAGGLE_DRIVERS_CSV = f"{KAGGLE_BASE}/drivers.csv"
KAGGLE_CONSTRUCTORS_CSV = f"{KAGGLE_BASE}/constructors.csv"
KAGGLE_QUALI_CSV = f"{KAGGLE_BASE}/qualifying.csv"
KAGGLE_CIRCUITS_CSV = f"{KAGGLE_BASE}/circuits.csv"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

FEATURES_PATH = "data/processed/features.parquet"
MODEL_PATH = "data/processed/model_lgbm.txt"
