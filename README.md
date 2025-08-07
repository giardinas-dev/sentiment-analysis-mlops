


# Sentiment Analyzer Twitter


## Overview

This project focuses on sentiment analysis and online reputation monitoring using machine learning models and MLOps methodologies. It leverages fine-tuning of pre-trained models to analyze social media data and automatically classify user sentiments as positive, neutral, or negative. The system includes CI/CD pipelines, deployment strategies, and continuous monitoring tools to help businesses proactively manage and improve their reputation online.
## Features

- **Fine-tuning of Pre-trained Models:** Adapt state-of-the-art models (e.g., RoBERTa) for domain-specific sentiment analysis.
- **Automated Sentiment Analysis:** Process social media text data to detect sentiment trends in real-time.
- **CI/CD Pipelines:** Automated workflows for training, testing, and deploying models.
- **Continuous Monitoring:** Integration with monitoring tools to track model performance and sentiment shifts over time.
- **Scalable Deployment:** Options for deploying models on platforms like HuggingFace or custom infrastructure.

## Project Structure
.github/workflows
    ci-cd.yml
    train.yml
src
    __pycache__
    __init__.py
    train.py
    utils.py
tests
    __init__.py
    test_integration.py
    test_train.py
    test_utils.py
.gitignore
README.md
app.py
requirements-ci.txt
requirements.txt


- **app.py**: Hugging Face Space app for inference and metrics posting.  
- **src/**: Python modules for model training, dataset handling, and utilities.  
- **train.py**: Script to train the model locally or on Hugging Face.  
- **tests/**: Unit and integration tests.  
- **.github/workflows/**: CI/CD pipelines for training and deployment.

# Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

export HF_TOKEN="your_hf_token_here"  # Linux/macOS
set HF_TOKEN=your_hf_token_here       # Windows
```

---

# Training

To train the model locally:

```bash
python src/train.py --learning_rate 5e-5 --model_name "distilbert-base-uncased" --subset_ratio 0.1
```


---

# CI/CD Workflows

- **ci-cd.yml**: Continuous integration for testing and deployment.
- **train.yml**: Automatic training pipeline for Hugging Face Spaces.

# App Usage (`app.py`)

The web app allows users to:
1. Submit text for sentiment analysis.  
2. Receive prediction results.  
3. Post metrics (average sentiment values, text lengths, counts) to an external monitoring endpoint.

# Metrics & Monitoring

The app automatically:
- Sends sentiment statistics via POST requests to a monitoring API.
- Supports visualization for:
  - Sentiment distribution over the last 24 hours.
  - Average sentiment value trends.
  - Text length vs. sentiment correlations.

Example API call:
```python
import requests
metrics = {"sentiment": "positive", "value": 0.8, "text": "I'm happy"}
requests.post("https://your-monitoring-endpoint.com/metrics", json=metrics)
```


---


# Development & Visualization

- VS Code or Google Colab*can be used to run training, visualize metrics, or debug the app.  
- GPU support is available in Colab or Hugging Face Spaces for faster training.  
- Use `matplotlib` and `seaborn` for plotting metrics locally.

# Testing

Run unit and integration tests:

```bash
pytest tests/
```

---

# Notes

- Environment variable `HF_TOKEN` must be active for training and downloading datasets.  
- Ensure proper dataset availability in Hugging Face or Kaggle when running `train.py`.  
- For stratified sampling, make sure `subset_ratio` is passed as a float.
