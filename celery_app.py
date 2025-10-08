# celery_app.py
from celery import Celery

# Connect to RabbitMQ broker (replace 'rabbitmq' with 'localhost' if running locally)
celery_app = Celery(
    "aiops_tasks",
    broker="amqp://guest:guest@localhost:5672//",
    backend="rpc://"
)

@celery_app.task
def retrain_model_task(data):
    import os
    import joblib
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from datetime import datetime

    # --- Simulated retraining process ---
    df = pd.DataFrame(data)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["log"])

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)

    os.makedirs("models", exist_ok=True)
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = f"models/isolation_forest_{version}.joblib"
    vectorizer_file = f"models/vectorizer_{version}.joblib"

    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)

    return {
        "version": version,
        "status": "completed",
        "model_path": model_file,
        "vectorizer_path": vectorizer_file
    }
