from data_processing import get_latest_seasons, load_raw_data, get_weather_data, merge_data, split_and_save

def update_dataset():
    print("updating F1 dataset")
    datasets = load_raw_data()
    train, test, predict = get_latest_seasons()

    all_years = train + test + predict
    weather_df = get_weather_data(all_years)
    df = merge_data(datasets, weather_df)
    split_and_save(df)
    print("update complete")
    
if __name__ == "__main__":
    update_dataset()