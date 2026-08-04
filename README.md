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

https://github.com/albertomendez164-sketch/ANA_680_Wine_Project_WK3_Mendez

### Heroku Application

https://ana-680-wine-projectwk3-mendez-1d3521ca56c4.herokuapp.com/

## Author

Alberto Mendez

National University

ANA 680 – Machine Learning Engineering
