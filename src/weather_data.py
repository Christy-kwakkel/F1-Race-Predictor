import fastf1
import pandas as pd

def get_weather_data(years):
    rows = []
    for year in years:
        try: 
            schedule = fastf1.get_event_schedule(year)
            for _, event in schedule.iterrows():
                try:
                    session = fastf1.getsession(year, event["Roundnumber"], "R")
                    session.load(laps=False, telemetry=False, weather=True)
                    weather = session.weather_data.mean()
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
    return pd.DataFrame(rows)