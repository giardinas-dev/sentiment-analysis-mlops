import gradio as gr
from src.utils import SentimentAnalyzer
import torch

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig


MODEL = "giardinsdev/sentiment-analyzer-twitter"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

analyzer = SentimentAnalyzer(model, tokenizer, config)

# Mappa label -> colore
LABEL_COLORS = {
    "negative": "red",
    "neutral": "gray",
    "positive": "green"
}

def analyze_sentiment(text):
    results = analyzer.predict(text)
    # Costruisco HTML colorato
    lines = []
    for label, score in results.items():
        color = LABEL_COLORS.get(label.lower(), "black")
        lines.append(f'<div style="color:{color}; font-weight:bold;">{label}: {score*100:.2f}%</div>')
    return "".join(lines)

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Inserisci testo da analizzare..."),
    outputs=gr.HTML(),  # Qui cambio l’output in HTML per il rendering dei colori
    title="Sentiment Analyzer Twitter",
    description="Analizza il sentiment del testo inserito usando un modello BERT multilingue."
)

if __name__ == "__main__":
    iface.launch()
