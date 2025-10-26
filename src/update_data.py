from data_processing import get_latest_seasons, split_and_save
from load_data import load_raw_data
from weather_data import get_weather_data
from merge_data import merge_data

def update_dataset():
    print("updating F1 dataset")
    train, test, predict = get_latest_seasons()
    datasets = load_raw_data()
    df = merge_data(datasets, get_weather_data(train + test + predict))
    split_and_save(df, train, test, predict)
    print("update complete")
    
if __name__ == "__main__":
    update_dataset()