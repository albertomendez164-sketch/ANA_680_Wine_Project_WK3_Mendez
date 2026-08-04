# Wine Quality Prediction – ANA 680 Assignment 5

## Project Overview

This project predicts the quality of red wine using a Linear Regression machine learning model trained on the UCI Wine Quality dataset. The application was developed using Flask, containerized with Docker, deployed to Heroku, and includes AWS SageMaker implementations with and without container technology.

## Dataset

- UCI Wine Quality Dataset
- Red Wine Dataset
- 1,599 observations
- 11 physicochemical features
- Target: Wine Quality (0–10)

## Machine Learning Model

Algorithm:
- Linear Regression

Evaluation Metrics:

- MAE: 0.5035
- RMSE: 0.6245
- R²: 0.4032

## Technologies Used

- Python
- Scikit-learn
- Flask
- Docker
- Heroku
- AWS SageMaker
- GitHub Actions

## Project Structure

```text
app.py
train_model.py
templates/
tests/
Dockerfile
Procfile
requirements.txt
wine_quality_model.pkl
Problem_1_Wine_Quality_Heroku.ipynb
Problem_2_SageMaker_Linear_Regression.ipynb
```

## Running Locally

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Application:

```
http://127.0.0.1:5000
```

## Deployment

### GitHub Repository

(Add your GitHub repository URL here)

### Heroku Application

(Add your Heroku URL here)

## Author

Alberto Mendez

National University

ANA 680 – Machine Learning Engineering
