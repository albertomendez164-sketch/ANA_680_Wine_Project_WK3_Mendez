"""SageMaker-compatible custom-container training entry point."""
import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

train_dir = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
model_dir.mkdir(parents=True, exist_ok=True)

csv_files = list(train_dir.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV file found in {train_dir}")

data = pd.read_csv(csv_files[0], sep=";")
X = data.drop(columns=["quality"])
y = data["quality"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression()),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)

metrics = {
    "MAE": float(mean_absolute_error(y_test, pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
    "R2": float(r2_score(y_test, pred)),
}
joblib.dump(model, model_dir / "model.pkl")
(model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics))
