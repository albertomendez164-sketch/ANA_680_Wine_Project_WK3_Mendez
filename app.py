"""Flask application for red-wine quality prediction."""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).with_name("wine_quality_model.pkl")
model = joblib.load(MODEL_PATH)

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


@app.get("/")
def home():
    return render_template("index.html", features=FEATURES)


@app.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.post("/predict")
def predict():
    try:
        # Read the values entered by the user
        values = [float(request.form[name]) for name in FEATURES]

        # Create a DataFrame with the correct feature names
        input_data = pd.DataFrame([values], columns=FEATURES)

        # Predict wine quality
        prediction = float(model.predict(input_data)[0])

        # Keep prediction between 0 and 10
        prediction = max(0.0, min(10.0, prediction))

        rounded = int(round(prediction))

        return render_template(
            "index.html",
            features=FEATURES,
            prediction=f"Predicted quality: {prediction:.2f} / 10",
            rounded=f"Nearest whole-number score: {rounded}",
            submitted=request.form,
        )

    except (KeyError, TypeError, ValueError) as exc:
        return render_template(
            "index.html",
            features=FEATURES,
            error=f"Please enter a valid number for every field. Details: {exc}",
            submitted=request.form,
        ), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)