# Standard library
import os

# Scientific computing
import numpy as np
import pandas as pd
from scipy.special import softmax

# Progress bar
from tqdm import tqdm

# Machine learning and evaluation
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# PyTorch
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# Transformers (Hugging Face)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig
)

# Hugging Face Hub
from huggingface_hub import login

# KaggleHub (custom dataset loader)
import kagglehub
from kagglehub import KaggleDatasetAdapter

class SentimentAnalyzer:
    def __init__(self, model, tokenizer, config, criterion = None , opt = None, epochs=15, freeze_base=True):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.epochs = epochs
        self.freeze_base = freeze_base
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = opt 
        self.criterion = criterion
        self.model.to(self.device)

        if freeze_base:
            self.freeze_base_layers()
        else:
            self.unfreeze_all_layers()

    def freeze_base_layers(self):
      for name, param in self.model.named_parameters():
        if any(f"roberta.encoder.layer.{i}." in name for i in range(8, 12)) or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False



    def unfreeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def preprocess(self, text):
        return self.tokenizer(text, return_tensors='pt')

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            encoded_input = self.preprocess(text).to(self.device)
            output = self.model(**encoded_input)
            scores = output[0][0].detach().cpu().numpy()
            probs = softmax(scores)

        ranking = np.argsort(probs)[::-1]
        results = {self.config.id2label[i]: float(np.round(probs[i], 4)) for i in ranking}
        return results

    def print_prediction(self, text):
        results = self.predict(text)
        print(f"Sentiment analysis for: \"{text}\"")
        for i, (label, score) in enumerate(results.items(), 1):
            print(f"{i}) {label}: {score}")

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=15):
        if self.optimizer is None:
            raise ValueError("Optimizer non fornito.")
        if self.criterion is None:
            raise ValueError("Loss Function non fornita.")

        optimizer = self.optimizer

        train_losses = []
        val_losses = []
        val_accuracies = []
        val_f1 = []

        num_samples = len(X_train)

        print("Inizio training...")

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            self.model.train()
            total_loss = 0

            for i in tqdm(range(0, num_samples, self.batch_size),"batch count"):
                batch_texts = X_train[i:i + self.batch_size].tolist()
                batch_labels = y_train[i:i + self.batch_size].tolist()

                # Preprocessing
                texts = [str(t) for t in batch_texts]  # fix importante!
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                labels = torch.tensor(batch_labels).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / (num_samples / self.batch_size)
            train_losses.append(avg_train_loss)

            if X_val is not None and y_val is not None:
                val_loss, val_acc, val_f1_score = self.evaluate(X_val, y_val)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                val_f1.append(val_f1_score)

                tqdm.write(
                    f"[Epoch {epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1_score:.4f}"
                )
            else:
                tqdm.write(f"[Epoch {epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f}")
        
        # Salva il modello localmente
        self.model.save_pretrained("sentiment-twitter-colab")
        self.tokenizer.save_pretrained("sentiment-twitter-colab")
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "val_f1": val_f1
        }

    def evaluate(self, X_val, y_val):
        if self.optimizer is None:
            raise ValueError("Optimizer non fornito.")
        if self.criterion is None:
            raise ValueError("Loss Function non fornita.")
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        num_samples = len(X_val)

        with torch.no_grad():
            for i in tqdm(range(0, num_samples, self.batch_size),"Testing samples"):
                batch_texts = X_val[i:i + self.batch_size].tolist()
                batch_labels = y_val[i:i + self.batch_size].tolist()
                texts = [str(t) for t in batch_texts]
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                labels = torch.tensor(batch_labels).to(self.device)

                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / (num_samples / self.batch_size)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, accuracy, f1


def get_stratified_subset(df, label_col='sentiment', subset_ratio=0.2, random_state=42):
    _, subset = train_test_split(
        df,
        test_size=subset_ratio,
        stratify=df[label_col],
        random_state=random_state
    )
    return subset




def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def prepare_data_and_model(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", subset_ratio=0.1, learning_rate = 3e-5, hf_token = None):
    print("Prepare model and data...")
    # Load tokenizer, config, model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)
    config = AutoConfig.from_pretrained(model_name, use_auth_token=hf_token)

 


    # Load dataset from Hugging Fac via kagglehub
    hf_dataset = kagglehub.load_dataset(
        KaggleDatasetAdapter.HUGGING_FACE,
        "jp797498e/twitter-entity-sentiment-analysis",
        "twitter_training.csv"
    )

    # Convert to DataFrame and rename columns
    dataset = hf_dataset.to_pandas()
    dataset.columns = ['Tweet ID', 'entity', 'sentiment', 'Tweet content']

    # Get stratified subset
    dataset = get_stratified_subset(dataset, label_col='sentiment', subset_ratio=subset_ratio)

    # Clean dataset
    dataset.drop_duplicates(inplace=True)
    dataset = dataset[dataset['sentiment'] != 'Irrelevant']
    dataset = dataset[dataset['Tweet content'].notnull()]

    # Feature and label separation
    X = dataset['Tweet content']
    y = dataset['sentiment']

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Compute class weights
    counts = dataset['sentiment'].value_counts()
    negative_count = counts.get('Negative', 0)
    positive_count = counts.get('Positive', 0)
    neutral_count = counts.get('Neutral', 0)
    count_tensor = torch.tensor([negative_count, positive_count, neutral_count], dtype=torch.float)
    weights = 1.0 / count_tensor
    weights = weights / weights.sum()
    criterion = CrossEntropyLoss(weight=weights.to(device))

    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train_encoded,
        "y_test": y_test_encoded,
        "label_encoder": le,
        "optimizer": optimizer,
        "loss_fn": criterion,
        "device": device,
        "dataset": dataset
    }

