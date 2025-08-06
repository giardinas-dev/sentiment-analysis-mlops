import responses
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import SentimentAnalyzer 
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

import responses
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import SentimentAnalyzer


@responses.activate
def test_metrics_sent_posted():
    print("test_metrics_sent_posted")
    responses.add(
        responses.POST,
        "https://metrics-fastapi-sentiment-analysis.onrender.com/metrics",
        json={"status": "ok"},
        status=200
    )

    # Supponiamo che qui venga eseguita la chiamata POST (da inserire nel test)
    import requests
    response = requests.post("https://metrics-fastapi-sentiment-analysis.onrender.com/metrics", json={"text": "test"})

    # Assert: verifichiamo che la richiesta sia stata effettuata e che la risposta sia corretta
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == "https://metrics-fastapi-sentiment-analysis.onrender.com/metrics"
    assert responses.calls[0].request.method == responses.POST
