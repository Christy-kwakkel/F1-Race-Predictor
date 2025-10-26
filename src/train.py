from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import pandas as pd
import joblib
import numpy as np

def training_model():
    # load the training and testing csv files
    train_df = pd.read_csv("../data/processed/train.csv")
    test_df = pd.read_csv("../data/processed/test.csv")

    # weights
    # weights = np.where(train_df["is_post2022"] == 1, 3.0, 1.0)

    # features (X) and target (y)
    features = ["driver_form", "constructor_form", "grid", "ispost2022"]
    X_train, y_train = train_df[features], train_df["positionOrder"]
    X_test, y_test = test_df[features], test_df["positionOrder"]

    # model
    scaler = StandardScaler().fit(X_train)
    model = GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, random_state=42)
    model.fit(scaler.transform(X_train), y_train)

    # evaluation
    
    mae = mean_absolute_error(y_test, model.predict(scaler.transform(X_test)))
    print("Mean Absolute Error: ", mae)
    joblib.dump((model.scaler), "../data/processed/GBR_model.pkl")
    print("model saved")

if __name__ == "__main__":
    training_model()

# headers = list(predict_df.columns)
# print(headers)

# for col in predict_df.columns:
#     print(col)


