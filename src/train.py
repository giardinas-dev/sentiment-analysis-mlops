import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
from huggingface_hub import login
from utils import SentimentAnalyzer, prepare_data_and_model

# --- Parse command line arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--model_name", type=str, default="giardinsdev/sentiment-analyzer-twitter")
parser.add_argument("--subset_ratio", type=str, default=0.1)
args = parser.parse_args()

# --- Hugging Face login ---
import os
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# --- Load data, model, tokenizer, etc. ---
assets = prepare_data_and_model(learning_rate=args.learning_rate, model_name = args.model_name , subset_ratio = args.subset_ratio)

model = assets["model"]
tokenizer = assets["tokenizer"]
X_train = assets["X_train"]
y_train = assets["y_train"]
config = assets["config"]
X_test = assets["X_test"]
y_test = assets["y_test"]
optimizer = assets["optimizer"]
loss_fn = assets["loss_fn"]
device = assets["device"]

# --- Initialize SentimentAnalyzer ---
sentiment = SentimentAnalyzer(
    model, tokenizer, config,
    loss_fn, optimizer,
    epochs=args.epochs,
    freeze_base=True
)

# --- Train & Evaluate ---
sentiment.train(X_train, y_train, X_test, y_test, epochs=args.epochs)
sentiment.evaluate(X_test, y_test)

# --- Push to Hugging Face Hub ---
sentiment.model.push_to_hub(args.model_name)
sentiment.tokenizer.push_to_hub(args.model_name)
