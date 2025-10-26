from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import joblib
import numpy as np

def training_model():
    # load the training and testing csv files
    train_df = pd.read_csv("../data/processed/train.csv")
    test_df = pd.read_csv("../data/processed/test.csv")
    predict_df = pd.read_csv("../data/processed/predict.csv")

    # weights
    weights = np.where(train_df["is_post2022"] == 1, 3.0, 1.0)

    # features (X) and target (y)
    X_train = train_df[["driver_form", "constructor_form", "grid"]]
    y_train = train_df["positionOrder"]

    X_test = test_df[["driver_form", "constructor_form", "grid"]]
    y_test = test_df["positionOrder"]

    # model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)

    # evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error: ", mae)

    # saving the model
    joblib.dump(model, "../data/processed/RF_model.pkl")
    print("saved model")




if __name__ == "__main__":
    training_model()

# headers = list(predict_df.columns)
# print(headers)

# for col in predict_df.columns:
#     print(col)


