import os
import re
import joblib
import pandas as pd
import psutil
import yaml
from dotenv import load_dotenv
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi.responses import JSONResponse, Response
from prometheus_client import Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from typing import List

# ---------------- CONFIG ---------------- #
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

app = FastAPI()

# Paths
ANOMALY_HISTORY_FILE = "anomaly_history.csv"
MODEL_PATH_FILE = "latest_model.txt"
VECTORIZER_PATH_FILE = "latest_vectorizer.txt"

def get_latest_model_and_vectorizer():
    try:
        with open(MODEL_PATH_FILE, "r") as f:
            model_path = f.read().strip()
        with open(VECTORIZER_PATH_FILE, "r") as f:
            vectorizer_path = f.read().strip()

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer, model_path, vectorizer_path
    except Exception:
        return None, None, None, None

model, vectorizer, MODEL_PATH, VECTORIZER_PATH = get_latest_model_and_vectorizer()
API_PORT = int(os.getenv("API_PORT", config.get("api_port", 5000)))
ANOMALY_THRESHOLD = float(config.get("anomaly_threshold", 0))

# ---------------- INPUT MODEL ---------------- #
class LogInput(BaseModel):
    log: str
    label: int | None = None   # optional field

# ---------------- API ROUTES ---------------- #
@app.post("/analyze")
async def analyze(payload: LogInput):
    if model is None:
        raise HTTPException(status_code=500, detail="No trained model found. Retrain first.")

    try:
        # Normalize payload.log
        log_text = payload.log
        if isinstance(log_text, dict):
            log_text = log_text.get("log", str(log_text))

        # ML detection
        X = vectorizer.transform([log_text])
        score = model.decision_function(X)[0]
        is_anomaly = bool(score < ANOMALY_THRESHOLD)

        # Rule-based detection (force anomaly if keywords found)
        critical_keywords = ["ERROR", "FATAL", "CRITICAL", "OUT OF MEMORY", "KERNEL PANIC", "SHUTDOWN"]
        if any(keyword.lower() in log_text.lower() for keyword in critical_keywords):
            is_anomaly = True

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    current_timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "timestamp": current_timestamp,
        "log": log_text,
        "anomaly_score": round(float(score), 5),
        "is_anomaly": is_anomaly
    }

    # Save anomaly history
    record = pd.DataFrame([result])
    if os.path.exists(ANOMALY_HISTORY_FILE):
        record.to_csv(ANOMALY_HISTORY_FILE, mode="a", header=False, index=False)
    else:
        record.to_csv(ANOMALY_HISTORY_FILE, index=False)

    return JSONResponse(content=result)

# ---------------- Anomaly History Route ---------------- #
@app.get("/anomaly-history")
async def anomaly_history():
    if not os.path.exists(ANOMALY_HISTORY_FILE):
        return []  # no anomalies yet

    try:
        df = pd.read_csv(ANOMALY_HISTORY_FILE)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load anomaly history: {e}")


# ---------------- Celery-based Retraining (Async) ---------------- #
@app.post("/retrain")
async def retrain_async(new_logs: List[LogInput]):
    """Trigger async retraining via Celery"""
    try:
        logs = [item.dict() for item in new_logs]
        task = retrain_model_task.delay(logs)
        return {"message": "Retraining started", "task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start retraining: {e}")


@app.get("/retrain/status/{task_id}")
def retrain_status(task_id: str):
    """Check retraining task status"""
    task = retrain_model_task.AsyncResult(task_id)
    if task.state == "PENDING":
        return {"status": "pending"}
    elif task.state == "SUCCESS":
        return {"status": "completed", "result": task.result}
    elif task.state == "FAILURE":
        return {"status": "failed", "error": str(task.info)}
    else:
        return {"status": task.state}

@app.get("/overview")
def overview():
    total_logs = anomalies = 0
    if os.path.exists(ANOMALY_HISTORY_FILE):
        df = pd.read_csv(ANOMALY_HISTORY_FILE)
        total_logs = len(df)
        if "is_anomaly" in df.columns:
            anomalies = df["is_anomaly"].sum()

    return {
        "total_logs": int(total_logs),
        "anomalies": int(anomalies),
        "models_deployed": 1,
        "active_users": 1
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "vectorizer_path": VECTORIZER_PATH
    }

@app.get("/model-info")
def model_info():
    if not os.path.exists(MODEL_PATH_FILE):
        raise HTTPException(status_code=404, detail="No model found")

    with open(MODEL_PATH_FILE, "r") as f:
        model_path = f.read().strip()

    return {
        "status": "ok",
        "latest_model": model_path,
        "file_size": os.path.getsize(model_path) if os.path.exists(model_path) else 0,
        "last_updated": datetime.fromtimestamp(os.path.getmtime(model_path), tz=timezone.utc).isoformat() if os.path.exists(model_path) else None
    }

# ---------------- System Stats ---------------- #
registry = CollectorRegistry()
cpu_gauge = Gauge("system_cpu_usage_percent", "CPU usage percentage", registry=registry)
mem_gauge = Gauge("system_memory_usage_percent", "Memory usage percentage", registry=registry)
disk_gauge = Gauge("system_disk_usage_percent", "Disk usage percentage", registry=registry)
net_sent_gauge = Gauge("system_network_sent_bytes", "Network sent in bytes", registry=registry)
net_recv_gauge = Gauge("system_network_received_bytes", "Network received in bytes", registry=registry)

@app.get("/system-stats")
def get_system_stats():
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    net_io = psutil.net_io_counters()

    cpu_gauge.set(cpu)
    mem_gauge.set(mem)
    disk_gauge.set(disk)
    net_sent_gauge.set(net_io.bytes_sent)
    net_recv_gauge.set(net_io.bytes_recv)

    return {
        "cpu": cpu,
        "memory": mem,
        "disk": disk,
        "network_sent": net_io.bytes_sent,
        "network_recv": net_io.bytes_recv,
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
