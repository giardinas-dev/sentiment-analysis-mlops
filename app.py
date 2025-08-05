import gradio as gr
from src.utils import SentimentAnalyzer
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig

import matplotlib.pyplot as plt
import io
import time
from prometheus_client import Counter, Gauge, Histogram

# Caricamento modello
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest" #MODEL = "giardinsdev/sentiment-analyzer-twitter"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

analyzer = SentimentAnalyzer(model, tokenizer, config)

# Metriche Prometheus
REQUEST_COUNT = Counter('request_count', 'Numero totale richieste')
ERROR_COUNT = Counter('error_count', 'Numero totale errori')
CURRENT_USERS = Gauge('current_users', 'Numero utenti attivi')
PROCESSING_TIME = Histogram('processing_time_seconds', 'Tempo di elaborazione della richiesta')

# Variabile di stato utenti attivi
active_users = 0

LABEL_COLORS = {
    "negative": "red",
    "neutral": "gray",
    "positive": "green"
}

def analyze_sentiment(text):
    global active_users
    REQUEST_COUNT.inc()
    active_users += 1
    CURRENT_USERS.set(active_users)

    start_time = time.time()
    try:
        results = analyzer.predict(text)
    except Exception:
        ERROR_COUNT.inc()
        raise
    finally:
        duration = time.time() - start_time
        PROCESSING_TIME.observe(duration)
        active_users -= 1
        CURRENT_USERS.set(active_users)

    lines = []
    for label, score in results.items():
        color = LABEL_COLORS.get(label.lower(), "black")
        lines.append(f'<div style="color:{color}; font-weight:bold;">{label}: {score*100:.2f}%</div>')
    return "".join(lines)

def plot_metrics():
    fig, axs = plt.subplots(3, 1, figsize=(5, 8))

    # 1) Counter: richieste ed errori
    axs[0].bar(["Requests", "Errors"], [REQUEST_COUNT._value.get(), ERROR_COUNT._value.get()])
    axs[0].set_title("Counter: richieste ed errori")

    # 2) Gauge: utenti attivi
    axs[1].bar(["Utenti attivi"], [CURRENT_USERS._value.get()], color="orange")
    axs[1].set_ylim(0, 10)
    axs[1].set_title("Gauge: utenti attivi")

    # 3) Histogram: durata media richiesta
    count = PROCESSING_TIME._count.get()
    total_time = PROCESSING_TIME._sum.get()
    avg_time = total_time / count if count > 0 else 0
    axs[2].bar(["Durata media (s)"], [avg_time], color="purple")
    axs[2].set_ylim(0, 1)
    axs[2].set_title("Histogram: durata media richiesta")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

# Interfaccia sentiment
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Inserisci testo da analizzare..."),
    outputs=gr.HTML(),
    title="Sentiment Analyzer Twitter",
    description="Analizza il sentiment del testo inserito usando un modello BERT multilingue."
)

# Interfaccia metriche
metrics_interface = gr.Interface(
    fn=plot_metrics,
    inputs=[],
    outputs=gr.Image(type="pil"),
    title="Metriche Prometheus visualizzate nella UI"
)

# Tab multipli UI
demo = gr.TabbedInterface([iface, metrics_interface], ["Analisi Sentiment", "Metriche"])

if __name__ == "__main__":
    demo.launch()
