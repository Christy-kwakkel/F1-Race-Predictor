from src.data.build_dataset import build_and_save_features
from src.model.train import train_model
from src.model.evaluate import evaluate_order_metric
from config import TRAIN_SEASONS, TEST_SEASONS, PREDICT_SEASONS, FEATURES_PATH


def main():
    print("\n" + "="*70)
    print("F1 RACE OUTCOME PREDICTOR - FULL TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Build features (comment out if already built)
    print("\nStep 1/3: Building features from scratch...")
    print("(This may take 5-10 minutes for all seasons...)")
    # build_and_save_features()  # Uncomment to rebuild features
    print("Skipping feature build (already exists)")
    
    # Step 2: Train model
    print("\nStep 2/3: Training model...")
    train_model()
    
    # Step 3: Evaluate model
    print("\nStep 3/3: Evaluating model...")
    evaluate_order_metric()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
