import responses
from src.utils import SentimentAnalyzer 

@responses.activate
def test_metrics_sent_posted():
    responses.add(
        responses.POST,
        "https://metrics-fastapi-sentiment-analysis.onrender.com/metrics",
        json={"status": "ok"},
        status=200
    )

    result = analyze_sentiment("This is amazing!")
    assert "positive" in result.lower()
