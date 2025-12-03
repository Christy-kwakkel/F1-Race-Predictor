import pandas as pd
from pathlib import Path
from src.data.kaggle_loader import load_kaggle_core
import warnings

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

def save_2024_results():
    kaggle = load_kaggle_core()

    results = kaggle["results"]
    races = kaggle["races"]
    drivers = kaggle["drivers"]
    teams = kaggle["constructors"]

    races_2024 = races[races["year"] == 2024]

    results_2024 = results.merge(races_2024[["raceId", "year", "round", "name"]], on="raceId")
    results_2024 = results_2024.merge(drivers[['driverId', 'driverRef', 'code']], on='driverId', how='left')
    results_2024 = results_2024.merge(teams[['constructorId', 'name']], left_on='constructorId', right_on='constructorId', how='left')

    final_results = pd.DataFrame({
        'season': results_2024['year'],
        'round': results_2024['round'],
        'raceId': results_2024['raceId'],
        'driver_code': results_2024['driverRef'],
        'code': results_2024['code'],
        'race_name': results_2024['name_x'],
        'grid_position': results_2024['grid'],
        'finishing_pos': results_2024['position'],
        'points': results_2024['points']
    })

    final_results = final_results.sort_values(["season", "round", "finishing_pos"]).reset_index(drop=True)

    out_dir = Path('data/processed')
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / '2024_actual_results.csv'
    final_results.to_csv(csv_path, index=False)
    
    print(f"Saved actual 2024 results: {csv_path}")
    print(f"Total rows: {len(final_results)}")
    print(f"Total races: {final_results['round'].nunique()}")
    print(final_results.head(10).to_string(index=False))

if __name__ == "__main__":
    save_2024_results()
