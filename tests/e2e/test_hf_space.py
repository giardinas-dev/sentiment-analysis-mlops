import requests

def test_space_html_response():
    text = "This is fantastic!"
    url = "https://giardinsdev-sentiment-analyzer-twitter.hf.space/"  # oppure localhost se lo testi localmente

    response = requests.post(url + "api/predict", json={"data": [text]})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    html_output = data["data"][0]
    assert "positive" in html_output.lower() or "neutral" in html_output.lower() or "negative" in html_output.lower()
