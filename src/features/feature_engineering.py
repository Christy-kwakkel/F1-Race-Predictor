import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import IMG_DIR, MODEL_PATH


def plot_feature_importance():
    """
    Plot and save feature importance for Linear Regression model.
    Shows top 20 features by absolute coefficient value.
    """
    import pickle
    model_path_pkl = Path(MODEL_PATH) / "model_linear.pkl"
    
    with open(model_path_pkl, 'rb') as f:
        model = pickle.load(f)
    
    feature_names = [
        'grid', 'team_id_enc', 'driver_id_enc',
        'air_temp_mean', 'track_temp_mean', 'wind_speed_mean',
        'humidity_mean', 'pressure_mean', 'rain_flag',
        'drv_avg_pos_3', 'drv_avg_pos_5', 'drv_avg_pos_10',
        'drv_avg_pts_3', 'drv_avg_pts_5', 'drv_avg_pts_10',
        'team_avg_pos_3', 'team_avg_pos_5', 'team_avg_pos_10',
        'team_avg_pts_3', 'team_avg_pts_5', 'team_avg_pts_10'
    ]
    
    coefficients = model.coef_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'importance': np.abs(coefficients)
    })
    
    # Sort by absolute importance and take top 20 (we have 21 features)
    importance_df = importance_df.sort_values('importance', ascending=False).head(20)
    importance_df = importance_df.sort_values('importance', ascending=True)
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Create horizontal bar chart (uniform blue color like reference)
    bars = ax.barh(importance_df['feature'], importance_df['importance'], 
                   color='#1f77b4', height=0.7)
    
    # Styling to match reference image
    ax.set_xlabel('Importance', fontsize=13, fontweight='normal')
    ax.set_ylabel('Feature', fontsize=13, fontweight='normal')
    ax.set_title('Top 20 Feature Importances', fontsize=15, fontweight='normal', pad=15)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Clean up spines
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    # Set spine colors to black
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, length=5, width=0.8)
    
    plt.tight_layout()
    
    # Save figure
    img_dir = Path(IMG_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)
    output_path = img_dir / "feature_importance.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Feature importance plot saved to {output_path}")
    
    # Save CSV
    csv_path = img_dir / "feature_importance.csv"
    importance_df.sort_values('importance', ascending=False).to_csv(csv_path, index=False)
    print(f"Feature importance data saved to {csv_path}")
    
    # Print top 5
    print("\nTop 5 Most Important Features:")
    print("="*60)
    top5 = importance_df.sort_values('importance', ascending=False).head(5)
    for idx, row in top5.iterrows():
        direction = "positive" if row['coefficient'] > 0 else "negative"
        print(f"{row['feature']:<20} | Importance: {row['importance']:>7.3f} | Coef: {row['coefficient']:>7.3f} ({direction})")
    
    plt.close()


if __name__ == "__main__":
    plot_feature_importance()
