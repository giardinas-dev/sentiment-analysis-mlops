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
from huggingface_hub import login, create_repo
# Preprocess text (username and link placeholders

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import SentimentAnalyzer, preprocess, get_stratified_subset,prepare_data_and_model


assets = prepare_data_and_model()

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


sentiment = SentimentAnalyzer(model, tokenizer, config, loss_fn, optimizer, epochs=10, freeze_base=True)


#login(token="🟡 IL_TUO_TOKEN")

sentiment.train(X_train, y_train, X_test, y_test, epochs=1)
sentiment.evaluate(X_test, y_test)


sentiment.model.push_to_hub("giardinsdev/sentiment-analyzer-twitter")
sentiment.tokenizer.push_to_hub("giardinsdev/sentiment-analyzer-twitter")