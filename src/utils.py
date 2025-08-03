import torch
import numpy as np
from scipy.special import softmax
from tqdm import tqdm

class SentimentAnalyzer:
    def __init__(self, model, tokenizer, config, epochs=10, freeze_base=True):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.epochs = epochs
        self.freeze_base = freeze_base
        self.batch_size = 32

        if freeze_base:
            self.freeze_base_layers()
        else:
            self.unfreeze_all_layers()

    def freeze_base_layers(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = "classifier" in name

    def unfreeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def preprocess(self, text):
        return self.tokenizer(text, return_tensors='pt')

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            encoded_input = self.preprocess(text)
            output = self.model(**encoded_input)
            scores = output[0][0].detach().numpy()
            probs = softmax(scores)

        ranking = np.argsort(probs)[::-1]
        results = {self.config.id2label[i]: float(np.round(probs[i], 4)) for i in ranking}
        return results

    def print_prediction(self, text):
        results = self.predict(text)
        print(f"Sentiment analysis for: \"{text}\"")
        for i, (label, score) in enumerate(results.items(), 1):
            print(f"{i}) {label}: {score}")

    def train(self, X_train, y_train, optimizer, loss_fn, device):
        self.model.to(device)
        self.model.train()

        num_samples = len(X_train)
        for epoch in tqdm(range(self.epochs),'epochs'):
            total_loss = 0
            for i in tqdm(range(0, num_samples, self.batch_size),'batch count'):
                batch_texts = X_train[i:i+self.batch_size]
                batch_labels = y_train[i:i+self.batch_size]

                optimizer.zero_grad()

                # Tokenizza batch
                texts = [str(t) for t in batch_texts]  # fix importante!
                encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
                encodings = self.tokenizer(list(batch_texts), padding=True, truncation=True, return_tensors="pt").to(device)
                labels = torch.tensor(batch_labels).to(device)

                outputs = self.model(**encodings)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (num_samples / self.batch_size)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

    def evaluate(self, X_test, y_test, loss_fn, device):
        self.model.to(device)
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0
        num_samples = len(X_test)

        with torch.no_grad():
            for i in range(0, num_samples, self.batch_size):
                batch_texts = X_test[i:i+self.batch_size]
                batch_labels = y_test[i:i+self.batch_size]

                texts = [str(t) for t in batch_texts]  # fix importante!
                encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
                labels = torch.tensor(batch_labels).to(device)

                outputs = self.model(**encodings)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                loss = loss_fn(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / (num_samples / self.batch_size)
        accuracy = correct / total
        print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy


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

