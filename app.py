import gradio as gr
from src.utils import SentimentAnalyzer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configurazioni modello (modifica in base al modello che usi)
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

# Inizializzo tokenizer e modello
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Config dummy per id2label (dipende dal modello, qui es. 5 classi)
config = type('Config', (), {})()  # creo un oggetto vuoto
config.id2label = {
    0: "1 star",
    1: "2 stars",
    2: "3 stars",
    3: "4 stars",
    4: "5 stars",
}

# Istanza del sentiment analyzer
analyzer = SentimentAnalyzer(model, tokenizer, config)

def analyze_sentiment(text):
    results = analyzer.predict(text)
    return {k: f"{v*100:.2f}%" for k, v in results.items()}

# UI Gradio
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Inserisci testo da analizzare..."),
    outputs=gr.Label(num_top_classes=5),
    title="Sentiment Analyzer Twitter",
    description="Analizza il sentiment del testo inserito usando un modello BERT multilingue."
)

if __name__ == "__main__":
    iface.launch()
