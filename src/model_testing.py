from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("../data/processed/RF_model.pkl")

# Load prediction dataset
predict_df = pd.read_csv("../data/processed/predict.csv")

# Ask for year
year = int(input("Enter the year (e.g., 2024, 2025): "))

while True:
    try:
        # show available race locations for that year
        available_locations = predict_df[predict_df["year"] == year]["location"].unique()
        
        if len(available_locations) == 0:
            raise ValueError("No races found for this year. Try again.")
        
        print("\nAvailable locations for", year, ":", list(available_locations))
        location = input("Enter the race location from the list above: ").title()
        
        # filter predict_df for the selected race
        race_df = predict_df[(predict_df["year"] == year) & (predict_df["location"] == location)]
        
        if race_df.empty:
            raise ValueError("No race found for this location in the selected year. Try again.")
        
        break  # valid race found, exit loop
    
    except ValueError as e:
        print(e)
        break

# features for prediction
X_race = race_df[["driver_form", "constructor_form", "grid"]]
predictions = model.predict(X_race)
race_df["predicted_position"] = predictions.round(0)

# sorting 1-20
race_df = race_df.sort_values(by="predicted_position", ascending=True).reset_index(drop=True)

# show results
print(race_df[["surname", "name_y", "predicted_position"]])
