import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import gradio as gr
import requests
import os
from src.utils import SentimentAnalyzer
MODEL = "giardinsdev/sentiment-analyzer-twitter"
HF_TOKEN = os.environ.get("token-space2")  # Hugging Face Spaces legge automaticamente i secrets come variabili d'ambiente

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=HF_TOKEN)
config = AutoConfig.from_pretrained(MODEL, use_auth_token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, use_auth_token=HF_TOKEN)
# Assumiamo che SentimentAnalyzer sia una classe già definita da te che usa model, tokenizer e config
analyzer = SentimentAnalyzer(model, tokenizer, config)

LABEL_COLORS = {
    "negative": "red",
    "neutral": "gray",
    "positive": "green"
}

# URL del tuo endpoint FastAPI metrics (modifica con il tuo URL)
METRICS_URL = "https://metrics-fastapi-sentiment-analysis.onrender.com/metrics"

def analyze_sentiment(text):
    results = analyzer.predict(text)
    
    # Trovo il sentiment con il punteggio più alto
    top_label = max(results, key=results.get)
    top_score = results[top_label]
    
    # Invio i dati della metrica con POST
    payload = {
        "sentiment": top_label,
        "value": float(top_score),
        "text": text
    }
    try:
        response = requests.post(METRICS_URL, json=payload, timeout=2)
        response.raise_for_status()
    except Exception as e:
        print(f"Errore invio metrica: {e}")
    
    # Costruisco HTML colorato per output
    lines = []
    for label, score in results.items():
        color = LABEL_COLORS.get(label.lower(), "black")
        lines.append(f'<div style="color:{color}; font-weight:bold;">{label}: {score*100:.2f}%</div>')
    return "".join(lines)

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Inserisci testo da analizzare..."),
    outputs=gr.HTML(),
    title="Sentiment Analyzer Twitter",
    description="Analizza il sentiment del testo inserito usando un modello BERT multilingue."
)
iface.launch(enable_queue=False)