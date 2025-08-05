import gradio as gr
from src.utils import SentimentAnalyzer
import torch

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# Istanza del sentiment analyzer
analyzer = SentimentAnalyzer(model, tokenizer, config)
       

def analyze_sentiment(text):
    results = analyzer.predict(text)
    lines = [f"{label}: {score*100:.2f}%" for label, score in results.items()]
    return "\n".join(lines)

# UI Gradio
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Inserisci testo da analizzare..."),
    outputs=gr.Textbox(), 
    title="Sentiment Analyzer Twitter",
    description="Analizza il sentiment del testo inserito usando un modello BERT multilingue."
)

if __name__ == "__main__":
    iface.launch()
