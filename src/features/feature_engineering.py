import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import IMG_DIR, MODEL_PATH


def add_driver_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling average features for driver performance.
    Calculates 3, 5, and 10-race rolling averages for position and points.
    """
    df = df.sort_values(["driver_code", "season", "round"]).copy()
    
    # Rolling averages for position (lower is better)
    df["drv_avg_pos_3"] = (
        df.groupby("driver_code")["position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["drv_avg_pos_5"] = (
        df.groupby("driver_code")["position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["drv_avg_pos_10"] = (
        df.groupby("driver_code")["position"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    
    # Rolling averages for points (higher is better)
    df["drv_avg_pts_3"] = (
        df.groupby("driver_code")["points"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["drv_avg_pts_5"] = (
        df.groupby("driver_code")["points"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["drv_avg_pts_10"] = (
        df.groupby("driver_code")["points"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    
    # Fill NaN with median for new drivers
    for col in ["drv_avg_pos_3", "drv_avg_pos_5", "drv_avg_pos_10"]:
        df[col] = df[col].fillna(df["position"].median())
    
    for col in ["drv_avg_pts_3", "drv_avg_pts_5", "drv_avg_pts_10"]:
        df[col] = df[col].fillna(0)
    
    return df


def add_team_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling average features for team performance.
    Calculates 3, 5, and 10-race rolling averages for position and points.
    """
    df = df.sort_values(["team_name", "season", "round"]).copy()
    
    # Rolling averages for position
    df["team_avg_pos_3"] = (
        df.groupby("team_name")["position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["team_avg_pos_5"] = (
        df.groupby("team_name")["position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["team_avg_pos_10"] = (
        df.groupby("team_name")["position"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    
    # Rolling averages for points
    df["team_avg_pts_3"] = (
        df.groupby("team_name")["points"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["team_avg_pts_5"] = (
        df.groupby("team_name")["points"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["team_avg_pts_10"] = (
        df.groupby("team_name")["points"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    
    # Fill NaN with median for new teams
    for col in ["team_avg_pos_3", "team_avg_pos_5", "team_avg_pos_10"]:
        df[col] = df[col].fillna(df["position"].median())
    
    for col in ["team_avg_pts_3", "team_avg_pts_5", "team_avg_pts_10"]:
        df[col] = df[col].fillna(0)
    
    return df


def finalize_features(df: pd.DataFrame):
    """
    Split dataframe into features (X), target (y), and metadata.
    Encodes categorical variables and selects final feature set.
    """
    from sklearn.preprocessing import LabelEncoder
    
    # Encode driver and team
    df = df.copy()
    df["driver_id_enc"] = LabelEncoder().fit_transform(df["driver_code"])
    df["team_id_enc"] = LabelEncoder().fit_transform(df["team_name"])
    
    # Define feature columns
    feature_cols = [
        'grid',
        'team_id_enc', 'driver_id_enc',
        'air_temp_mean', 'track_temp_mean', 'wind_speed_mean',
        'humidity_mean', 'pressure_mean', 'rain_flag',
        'drv_avg_pos_3', 'drv_avg_pos_5', 'drv_avg_pos_10',
        'drv_avg_pts_3', 'drv_avg_pts_5', 'drv_avg_pts_10',
        'team_avg_pos_3', 'team_avg_pos_5', 'team_avg_pos_10',
        'team_avg_pts_3', 'team_avg_pts_5', 'team_avg_pts_10'
    ]
    
    # Metadata columns for tracking predictions
    meta_cols = ['season', 'round', 'driver_code', 'team_name']
    
    # Target
    y = df['position']
    
    # Features
    X = df[feature_cols].copy()
    
    # Fill any remaining NaN
    X = X.fillna(X.median())
    
    # Metadata
    meta = df[meta_cols].copy()
    
    return X, y, meta


# def plot_feature_importance():
#     """
#     Plot and save feature importance for Linear Regression model.
#     Shows top 20 features by absolute coefficient value.
#     """
#     import pickle
#     model_path_pkl = Path(MODEL_PATH) / "model_linear.pkl"
    
#     with open(model_path_pkl, 'rb') as f:
#         model = pickle.load(f)
    
#     feature_names = [
#         'grid', 'team_id_enc', 'driver_id_enc',
#         'air_temp_mean', 'track_temp_mean', 'wind_speed_mean',
#         'humidity_mean', 'pressure_mean', 'rain_flag',
#         'drv_avg_pos_3', 'drv_avg_pos_5', 'drv_avg_pos_10',
#         'drv_avg_pts_3', 'drv_avg_pts_5', 'drv_avg_pts_10',
#         'team_avg_pos_3', 'team_avg_pos_5', 'team_avg_pos_10',
#         'team_avg_pts_3', 'team_avg_pts_5', 'team_avg_pts_10'
#     ]
    
#     coefficients = model.coef_
    
#     importance_df = pd.DataFrame({
#         'feature': feature_names,
#         'coefficient': coefficients,
#         'importance': np.abs(coefficients)
#     })
    
#     # Sort by absolute importance and take top 20 (we have 21 features)
#     importance_df = importance_df.sort_values('importance', ascending=False).head(20)
#     importance_df = importance_df.sort_values('importance', ascending=True)
    
#     # Create figure with white background
#     fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
#     ax.set_facecolor('white')
    
#     # Create horizontal bar chart (uniform blue color like reference)
#     bars = ax.barh(importance_df['feature'], importance_df['importance'], 
#                    color='#1f77b4', height=0.7)
    
#     # Styling to match reference image
#     ax.set_xlabel('Importance', fontsize=13, fontweight='normal')
#     ax.set_ylabel('Feature', fontsize=13, fontweight='normal')
#     ax.set_title('Top 20 Feature Importances', fontsize=15, fontweight='normal', pad=15)
    
#     # Add grid
#     ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
#     ax.set_axisbelow(True)
    
#     # Clean up spines
#     ax.spines['top'].set_visible(True)
#     ax.spines['right'].set_visible(True)
#     ax.spines['left'].set_visible(True)
#     ax.spines['bottom'].set_visible(True)
    
#     # Set spine colors to black
#     for spine in ax.spines.values():
#         spine.set_edgecolor('black')
#         spine.set_linewidth(0.8)
    
#     # Adjust tick parameters
#     ax.tick_params(axis='both', which='major', labelsize=10, length=5, width=0.8)
    
#     plt.tight_layout()
    
#     # Save figure
#     img_dir = Path(IMG_DIR)
#     img_dir.mkdir(parents=True, exist_ok=True)
#     output_path = img_dir / "feature_importance.png"
    
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     print(f"Feature importance plot saved to {output_path}")
    
#     # Save CSV
#     csv_path = img_dir / "feature_importance.csv"
#     importance_df.sort_values('importance', ascending=False).to_csv(csv_path, index=False)
#     print(f"Feature importance data saved to {csv_path}")
    
#     # Print top 5
#     print("\nTop 5 Most Important Features:")
#     print("="*60)
#     top5 = importance_df.sort_values('importance', ascending=False).head(5)
#     for idx, row in top5.iterrows():
#         direction = "positive" if row['coefficient'] > 0 else "negative"
#         print(f"{row['feature']:<20} | Importance: {row['importance']:>7.3f} | Coef: {row['coefficient']:>7.3f} ({direction})")
    
#     plt.close()


# if __name__ == "__main__":
    # plot_feature_importance()
