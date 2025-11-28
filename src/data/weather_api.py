import datetime as dt
import pandas as pd
import requests
from config import OPEN_METEO_URL

def fetch_race_weather(lat, lon, date, tz="UTC"):
    start = date.strftime("%Y-%m-%d")
    end = start
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,precipitation",
        "timezone": tz,
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if "hourly" not in data:
        return None
    hourly = pd.DataFrame(data["hourly"])
    features = {
        "air_temp_mean_ext": hourly["temperature_2m"].mean(),
        "humidity_mean_ext": hourly["relative_humidity_2m"].mean(),
        "pressure_mean_ext": hourly["pressure_msl"].mean(),
        "wind_speed_mean_ext": hourly["wind_speed_10m"].mean(),
        "rain_flag_ext": (hourly["precipitation"] > 0).any(),
    }
    return features
