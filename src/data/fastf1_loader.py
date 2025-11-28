import fastf1
import pandas as pd
from config import FASTF1_CACHE_DIR

fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)


def load_race_session(season: int, round_or_name) -> fastf1.core.Session:
    session = fastf1.get_session(season, round_or_name, 'R')
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    return session


def session_to_driver_race_df(session) -> pd.DataFrame:
    results = session.results
    df = results[['DriverNumber', 'Abbreviation', 'TeamName', 'GridPosition', 'Position', 'Status']]
    df = df.rename(columns={
        'DriverNumber': 'driver_number',
        'Abbreviation': 'driver_code',
        'TeamName': 'team_name',
        'GridPosition': 'grid',
        'Position': 'position'
    })
    df['season'] = session.event.year
    df['round'] = session.event.RoundNumber
    df['race_name'] = session.event.EventName
    df['circuit_name'] = session.event.Location
    return df.reset_index(drop=True)


def get_weather_features(session) -> pd.DataFrame:
    try:
        w = session.weather_data
        if w is None or len(w) == 0:
            raise ValueError("No weather data available")
        
        features = {
            'season': session.event.year,
            'round': session.event.RoundNumber,
            'air_temp_mean': float(w['AirTemp'].mean()),
            'track_temp_mean': float(w['TrackTemp'].mean()),
            'wind_speed_mean': float(w['WindSpeed'].mean()),
            'humidity_mean': float(w['Humidity'].mean()),
            'pressure_mean': float(w['Pressure'].mean()),
            'rain_flag': int((w['Rainfall'] > 0).any())
        }
    except Exception:
        # Fallback for missing weather data (common for older races)
        features = {
            'season': session.event.year,
            'round': session.event.RoundNumber,
            'air_temp_mean': None,
            'track_temp_mean': None,
            'wind_speed_mean': None,
            'humidity_mean': None,
            'pressure_mean': None,
            'rain_flag': None
        }
    
    return pd.DataFrame([features])


def build_fastf1_base(seasons):
    rows = []
    for year in seasons:
        rnd = 1
        while True:
            try:
                session = load_race_session(year, rnd)
            except Exception:
                break
            base_df = session_to_driver_race_df(session)
            weather_df = get_weather_features(session)
            base_df = base_df.merge(weather_df, on=['season', 'round'], how='left')
            rows.append(base_df)
            rnd += 1
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
