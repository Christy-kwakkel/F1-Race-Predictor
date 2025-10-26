import fastf1
import pandas as pd

def get_weather_data(years):
    rows = []
    for year in years:
        if year < 2018:
            print(f"skipping {year}, no weather data available in fastf1")
            continue
        
        try: 
            schedule = fastf1.get_event_schedule(year)
            for _, event in schedule.iterrows():
                session = fastf1.get_session(year, event["RoundNumber"], "R")
                session.load(laps=False, telemetry=False, weather=True)
                w = session.weather_data.mean(numeric_onlu=True)

                if session.weather_data is None or session.weather_data.empty:
                    print(f"no weather data for {year}")
                    continue

                weather = session.weather_data.mean(numeric_only=True)
                rows.append({
                    "year" : year,
                    "round": event["RoundNumber"],
                    "temp_air": weather['AirTemp'],
                    "temp_track": weather['TrackTemp'],
                    "humidity": weather['Humidity'],
                    "wind_speed": weather['WindSpeed'],
                    "pressure": weather['Pressure'],
                })

        except Exception as e:
            print(f" No weather for {year}: {e}")
    
    print(f"Collected weather data for races")
    return pd.DataFrame(rows)