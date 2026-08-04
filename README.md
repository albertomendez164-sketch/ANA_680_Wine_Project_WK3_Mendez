# Wine Quality Prediction — ANA 680 Assignment 5

This project predicts red-wine quality from 11 physicochemical features using
Scikit-learn linear regression.

## Model results

- MAE: 0.5035
- RMSE: 0.6245
- R²: 0.4032

The target is an ordinal score, but the assignment specifies linear regression.
The web application displays both the continuous prediction and its nearest
whole-number score.

## Main files

- `Problem_1_Wine_Quality_Heroku.ipynb`
- `Problem_2_SageMaker_Linear_Regression.ipynb`
- `app.py`
- `train_model.py`
- `wine_quality_model.pkl`
- `Dockerfile`
- `heroku.yml`
- `sagemaker_container/`
- `tests/test_app.py`
- `.github/workflows/test.yml`

## Run locally

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python train_model.py
    pytest -q
    python app.py

## Run with Docker

    docker build --platform linux/amd64 -t wine-quality-app .
    docker run -p 5000:5000 -e PORT=5000 wine-quality-app

## Submission placeholders

- GitHub Problem 1 notebook URL: ADD URL
- Heroku application URL: ADD URL
- GitHub Problem 2 notebook URL: ADD URL
- Troubleshooting summary: ADD IF NEEDED
