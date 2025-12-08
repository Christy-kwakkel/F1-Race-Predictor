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


GP_NAME_MAPPING = {
    "saudi arabian grand prix": ("Saudi Arabia", 2),
    "australian grand prix": ("Australia", 3),
    "japanese grand prix": ("Japan", 4),
    "chinese grand prix": ("China", 5),
    "miami grand prix": ("Miami", 6),
    "emilia romagna grand prix": ("Imola", 7),
    "monaco grand prix": ("Monaco", 8),
    "canadian grand prix": ("Canada", 9),
    "spanish grand prix": ("Spain", 10),
    "austrian grand prix": ("Austria", 11),
    "british grand prix": ("Silverstone", 12),
    "hungarian grand prix": ("Hungary", 13),
    "belgian grand prix": ("Spa", 14),
    "dutch grand prix": ("Zandvoort", 15),
    "italian grand prix": ("Monza", 16),
    "azerbaijan grand prix": ("Baku", 17),
    "singapore grand prix": ("Singapore", 18),
    "united states grand prix": ("Austin", 19),
    "mexico city grand prix": ("Mexico City", 20),
    "sao paulo grand prix": ("Interlagos", 21),
    "las vegas grand prix": ("Las Vegas", 22),
    "qatar grand prix": ("Qatar", 23),
    "abu dhabi grand prix": ("Yas Marina", 24),
}


OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

FEATURES_PATH = "data/processed/features.parquet"
MODEL_PATH = "data/processed/model_lgbm.txt"
