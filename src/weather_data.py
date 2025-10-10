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
                try:
                    round_number = event["RoundNumber"]
                    session = fastf1.get_session(year, event["RoundNumber"], "R")
                    session.load(laps=False, telemetry=False, weather=True)

                    if session.weather_data is None or session.weather_data.empty:
                        print(f"no weather data for {year} round {round_number}")
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
        except Exception as e:
            print(f"could not load schedule for {year}: {e}")
    
    df_weather = pd.DataFrame(rows)
    print(f"Collected weather data for {len(df_weather)} races")
    return pd.DataFrame(rows)