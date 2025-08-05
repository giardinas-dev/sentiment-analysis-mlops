import gradio as gr
from src.utils import SentimentAnalyzer
import torch

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig


from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from starlette.routing import Route
from starlette.applications import Starlette


MODEL = "giardinsdev/sentiment-analyzer-twitter"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

analyzer = SentimentAnalyzer(model, tokenizer, config)
REQUEST_COUNT = Counter('request_count', 'Numero di richieste sentiment')

# Mappa label -> colore
LABEL_COLORS = {
    "negative": "red",
    "neutral": "gray",
    "positive": "green"
}

def analyze_sentiment(text):
    REQUEST_COUNT.inc()  # incremento contatore metriche
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
# Endpoint Prometheus per /metrics
async def metrics(request):
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# App Starlette che espone /metrics e serve Gradio su /
app = Starlette(routes=[
    Route("/metrics", metrics),
    Route("/{path:path}", lambda request: Response(iface.launch(prevent_thread_lock=True, inline=False), media_type="text/html"))
])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

