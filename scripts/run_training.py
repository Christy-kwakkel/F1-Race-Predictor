from src.data.build_dataset import build_and_save_features
from src.model.train import train_model
from src.model.evaluate import evaluate_order_metric

from config import TRAIN_SEASONS, TEST_SEASONS, PREDICT_SEASONS, FEATURES_PATH


def main():
    build_and_save_features()
    train_model()
    evaluate_order_metric()

if __name__ == "__main__":
    main()
