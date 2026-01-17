import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from config import FEATURES_PATH, TRAIN_SEASONS, TEST_SEASONS, MODEL_PATH


def load_data():
    """Load and split data into train/test sets."""
    df = pd.read_parquet(FEATURES_PATH)
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    
    feature_cols = [c for c in df.columns if c not in ["season", "round", "driver_code", "team_name", "target_pos"]]
    
    X_train, y_train = train[feature_cols], train["target_pos"]
    X_test, y_test = test[feature_cols], test["target_pos"]
    
    return X_train, X_test, y_train, y_test, feature_cols


def evaluate_model(model, X_test, y_test):
    """Evaluate model and print metrics."""
    y_pred_test = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)
    
    print(f"\n{'='*60}")
    print(f"LINEAR REGRESSION TEST RESULTS")
    print(f"{'='*60}")
    print(f"R² Score:  {r2:.3f}  (explains {r2*100:.1f}% of variance)")
    print(f"MAE:       {mae:.3f} positions")
    print(f"RMSE:      {rmse:.3f} positions")
    print(f"MSE:       {mse:.3f}")
    print(f"MAPE:      {mape:.1%} (relative error)")
    print(f"{'='*60}\n")
    
    return {"r2": r2, "mae": mae, "mse": mse, "rmse": rmse, "mape": mape}


def train_model():
    """Train Linear Regression model."""
    print("\n{'='*60}")
    print("Training Linear Regression model...")
    print(f"{'='*60}\n")
    
    X_train, X_test, y_train, y_test, feature_cols = load_data()
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"Training complete")
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Save using pickle
    model_dir = Path(MODEL_PATH)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path_pkl = model_dir / "model_linear.pkl"
    
    with open(model_path_pkl, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path_pkl}")
    
    return model


if __name__ == "__main__":
    train_model()
