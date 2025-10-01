
# random forest n_estimators=200, random_state=42

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

predict_df = pd.read_csv("../data/processed/predict.csv")
X_predict = predict_df[["driver_form", "constructor_form", "grid"]]

predictions = model.predict(X_predict)
predict_df["predicted_position"] = predictions

print(predict_df[["driverId", "constructorId", "raceName", "predicted_position"]])
