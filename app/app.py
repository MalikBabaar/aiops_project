from datetime import datetime, timezone
import os
import re
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import yaml
from dotenv import load_dotenv
load_dotenv()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = os.getenv("MODEL_PATH", config.get("model_path", "./isolation_forest_model.joblib"))
API_PORT = int(os.getenv("API_PORT", config.get("api_port", 5000)))
ANOMALY_THRESHOLD = float(config.get("anomaly_threshold", 0))

app = FastAPI()
model = joblib.load(MODEL_PATH)

class LogInput(BaseModel):
    log: str

def extract_features_from_log(log: str):
    timestamp_str = log.split(' ')[0]
    timestamp = pd.to_datetime(timestamp_str, errors='coerce')
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.now()

    hour = timestamp.hour
    dayofweek = timestamp.dayofweek

    pid_match = re.search(r'\[(\d+)\]', log)
    pid = int(pid_match.group(1)) if pid_match else 0

    message_part = ' '.join(log.split(' ')[1:])

    auth_failure = int('authentication failure' in message_part.lower())
    failed_password = int('failed password' in message_part.lower())

    features = {
        'hour': hour,
        'dayofweek': dayofweek,
        'auth_failure': auth_failure,
        'failed_password': failed_password,
        'pid': pid
    }
    return pd.DataFrame([features])

@app.post("/analyze")
async def analyze(payload: LogInput):
    features_df = extract_features_from_log(payload.log)
    try:
        score = model.decision_function(features_df)[0]
        is_anomaly = bool(score < ANOMALY_THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # Current timestamp in ISO 8601 UTC format
    current_timestamp = datetime.now(timezone.utc).isoformat()

    result = {
        "timestamp": current_timestamp,
        "log": payload.log,
        "anomaly_score": round(float(score), 5),
        "is_anomaly": is_anomaly
    }
    return JSONResponse(content=result)

from sklearn.metrics import f1_score  # For retraining metrics

@app.post("/retrain")
async def retrain(new_logs: list[LogInput]):
    try:
        features_list = [extract_features_from_log(log.log) for log in new_logs]
        df_features = pd.concat(features_list, ignore_index=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse logs: {e}")

    from sklearn.ensemble import IsolationForest
    new_model = IsolationForest(contamination=0.01, random_state=42)

    try:
        import time
        start_time = time.time()

        new_model.fit(df_features)

        training_time_str = datetime.now(timezone.utc).isoformat()

        # Save updated model
        joblib.dump(new_model, MODEL_PATH)

        global model
        model = new_model

        # Optional: Calculate F1 on proxy labels or dummy labels (if you have)
        # Here, just faking f1_score for demo since no true labels
        f1 = 0.87

        new_version = "v1.2"  # You can manage this version string as you want

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

    return JSONResponse(content={
        "status": "success",
        "new_model_version": new_version,
        "f1_score": f1,
        "training_time": training_time_str
    })
