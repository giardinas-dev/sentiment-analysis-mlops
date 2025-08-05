from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import kagglehub
from kagglehub import KaggleDatasetAdapter
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
# Preprocess text (username and link placeholders

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import SentimentAnalyzer, preprocess, get_stratified_subset


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# Carica il dataset Hugging Face tramite kagglehub
hf_dataset = kagglehub.load_dataset(
    KaggleDatasetAdapter.HUGGING_FACE,
    "jp797498e/twitter-entity-sentiment-analysis",
    "twitter_training.csv"
)



# Converti in pandas DataFrame
dataset = hf_dataset.to_pandas()
# Imposta i nomi delle colonne corretti (se necessario)
columns = ['Tweet ID', 'entity', 'sentiment', 'Tweet content']
dataset.columns = columns
dataset =get_stratified_subset(dataset, label_col='sentiment', subset_ratio=0.1)



print(dataset, 'sentiment','Tweet content' )
# Opzionale: rimuovi duplicati
dataset.drop_duplicates(inplace=True)

dataset = dataset[dataset['sentiment'] != 'Irrelevant']
dataset = dataset[dataset['Tweet content'].notnull()]
# Controlla info e statistiche base
print(dataset.info())
print(dataset.describe())
print("Missing values per colonna:\n", dataset.isnull().sum())
print("Duplicati nel dataset:", dataset.duplicated().sum())



print("Distribuzione classi sentiment:\n", dataset['sentiment'].value_counts())
# Separazione features e target

X = dataset['Tweet content']
y = dataset['sentiment']

print("I numeri relativi agli input e i label corrispondono:",len(X) == len(y))
print("Numeri totali:", len(X))
# Split in train e test mantenendo proporzioni classi con stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(X.shape, y.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
# Verifica dimensioni split
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
loss_fn = CrossEntropyLoss()


sentiment = SentimentAnalyzer(model, tokenizer, config)
print(X_train)
sentiment.train(X_train, y_train_encoded, optimizer, loss_fn, device)
sentiment.evaluate(X_test, y_test_encoded, loss_fn, device)