import requests
"""
def test_space_html_response():
    text = "I love this! It's amazing."  # Test positivo

    url = "https://giardinsdev-sentiment-analyzer-twitter.hf.space/api/predict/"  # ⚠️ slash finale importante

    response = requests.post(url, json={"data": [text]})
    assert response.status_code == 200, f"Status code: {response.status_code}, Response: {response.text}"

    data = response.json()
    assert "data" in data, f"'data' key missing in response: {data}"

    html_output = data["data"][0]
    assert any(label in html_output.lower() for label in ["positive", "neutral", "negative"]), \
        f"No sentiment label found in HTML output: {html_output}"
"""