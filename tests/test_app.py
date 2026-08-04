import pytest
from app import app

@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    return app.test_client()

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Wine Quality Predictor" in response.data

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"

def test_prediction(client):
    sample = {
        "fixed acidity": "7.4",
        "volatile acidity": "0.70",
        "citric acid": "0.00",
        "residual sugar": "1.9",
        "chlorides": "0.076",
        "free sulfur dioxide": "11",
        "total sulfur dioxide": "34",
        "density": "0.9978",
        "pH": "3.51",
        "sulphates": "0.56",
        "alcohol": "9.4",
    }
    response = client.post("/predict", data=sample)
    assert response.status_code == 200
    assert b"Predicted quality" in response.data
