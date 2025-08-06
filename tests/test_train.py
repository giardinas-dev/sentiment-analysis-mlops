from src.utils import prepare_data_and_model
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import SentimentAnalyzer

def test_training_pipeline_runs():
    assets = prepare_data_and_model(subset_ratio=0.05)  # fast subset

    sentiment = SentimentAnalyzer(
        assets["model"],
        assets["tokenizer"],
        assets["config"],
        assets["loss_fn"],
        assets["optimizer"],
        epochs=1
    )

    metrics = sentiment.train(
        assets["X_train"],
        assets["y_train"],
        assets["X_test"],
        assets["y_test"],
        epochs=1
    )

    assert "train_losses" in metrics
    assert len(metrics["train_losses"]) == 1
