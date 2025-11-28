import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from config import FEATURES_PATH, TRAIN_SEASONS, TEST_SEASONS, MODEL_PATH

def train_model():
    df = pd.read_parquet(FEATURES_PATH)
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test = df[df["season"].isin(TEST_SEASONS)].copy()

    feature_cols = [c for c in df.columns if c not in ["season", "round", "driver_code", "team_name", "target_pos"]]

    X_train, y_train = train[feature_cols], train["target_pos"]
    X_test, y_test = test[feature_cols], test["target_pos"]

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)

    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42
    }

    model = lgb.train(params, train_set, num_boost_round=2000, valid_sets=[train_set, val_set], valid_names=["train", "val"], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])


    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f"Test MAE (position): {mae:.3f}")

    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
