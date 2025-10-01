from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

# load the training and testing csv files
train_df = pd.read_csv("../data/processed/train.csv")
test_df = pd.read_csv("../data/processed/test.csv")
predict_df = pd.read_csv("../data/processed/predict.csv")

# features (X) and target (y)
X_train = train_df[["driver_form", "constructor_form", "grid"]]
y_train = train_df["positionOrder"]

X_test = test_df[["driver_form", "constructor_form", "grid"]]
y_test = test_df["positionOrder"]

# model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: ", mae)



# headers = list(predict_df.columns)
# print(headers)

# for col in predict_df.columns:
#     print(col)


# PREDICT.CSV #
# Ask for year and location
year = int(input("\nEnter the year (e.g., 2024, 2025): "))

# Show available race locations for that year
available_locations = predict_df[predict_df["year"] == year]["location"].unique()
print("\nAvailable locations for", year, ":", available_locations)

location = input("\nEnter the race location from the list above: ").title()

# Filter dataframe for that race
race_df = predict_df[(predict_df["year"] == year) & (predict_df["location"] == location)]

year = int(input("Enter the year (e.g., 2024, 2025): "))

while True:
    try:
        available_locations = predict_df[predict_df["year"] == year]["location"].unique()
        
        if len(available_locations) == 0:
            raise ValueError("No races found for this year. Try again.")
        
        print("Available locations for", year, ":", list(available_locations))
        location = input("Enter the race location: ").title()
        
        # filter predict_df for the selected race
        race_df = predict_df[(predict_df["year"] == year) & (predict_df["location"] == location)]
        
        if race_df.empty:
            raise ValueError("No race found for this location in the selected year. Try again.")
        
        break  # valid race found, exit loop
    except ValueError as e:
        print(e)
        continue

# features for prediction
X_race = race_df[["driver_form", "constructor_form", "grid"]]
predictions = model.predict(X_race)
race_df["predicted_position"] = predictions

# show results
print(race_df[["surname", "name_y", "predicted_position"]])