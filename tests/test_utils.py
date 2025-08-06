import pytest
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from src.utils import SentimentAnalyzer

@pytest.fixture
def setup_analyzer():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    analyzer = SentimentAnalyzer(model, tokenizer, config)
    return analyzer

def test_predict_output_keys(setup_analyzer):
    text = "I love AI!"
    result = setup_analyzer.predict(text)
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())
    assert all(isinstance(k, str) for k in result.keys())

def test_predict_sorted_confidence(setup_analyzer):
    result = setup_analyzer.predict("Great job!")
    values = list(result.values())
    assert values == sorted(values, reverse=True)
