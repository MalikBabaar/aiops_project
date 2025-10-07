import os
import joblib
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from celery_app import celery_app

MODEL_PATH_FILE = "models/current_model_path.txt"
VECTORIZER_PATH_FILE = "models/current_vectorizer_path.txt"

@celery_app.task(bind=True)
def retrain_model_task(self, logs):
    df_features = pd.DataFrame(logs)
    if "log" not in df_features.columns:
        return {"error": "Logs must contain a 'log' field."}

    new_vectorizer = TfidfVectorizer(max_features=5000)
    X = new_vectorizer.fit_transform(df_features["log"])

    contamination = 0.05
    new_model = IsolationForest(contamination=contamination, random_state=42)
    new_model.fit(X)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models", exist_ok=True)
    model_filename = f"models/isolation_forest_{version}.joblib"
    vectorizer_filename = f"models/vectorizer_{version}.joblib"

    joblib.dump(new_model, model_filename)
    joblib.dump(new_vectorizer, vectorizer_filename)

    with open(MODEL_PATH_FILE, "w") as f:
        f.write(model_filename)
    with open(VECTORIZER_PATH_FILE, "w") as f:
        f.write(vectorizer_filename)

    y_pred = new_model.predict(X)
    y_pred = [0 if p == 1 else 1 for p in y_pred]
    anomalies = sum(y_pred)
    anomaly_ratio = anomalies / len(y_pred)

    if "label" in df_features.columns:
        y_true = df_features["label"].tolist()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        acc = prec = rec = f1 = None

    return {
        "new_model_version": version,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "detected_anomalies": anomalies,
        "anomaly_ratio": anomaly_ratio,
    }
