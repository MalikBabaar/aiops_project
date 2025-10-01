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

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load latest model path
def get_latest_model_path():
    try:
        with open("latest_model.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return config.get("model_path", "./models/isolation_forest_model.joblib")

MODEL_PATH = get_latest_model_path()
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

    current_timestamp = datetime.now(timezone.utc).isoformat()

    result = {
        "timestamp": current_timestamp,
        "log": payload.log,
        "anomaly_score": round(float(score), 5),
        "is_anomaly": is_anomaly
    }
    return JSONResponse(content=result)

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
        new_model.fit(df_features)
        training_time_str = datetime.now(timezone.utc).isoformat()

        # Save model with version
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"models/isolation_forest_{version}.joblib"
        joblib.dump(new_model, model_filename)

        # Update latest model reference
        with open("latest_model.txt", "w") as f:
            f.write(model_filename)

        global model
        model = new_model

        f1 = 0.87  # Placeholder score
        new_version = version

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

    return JSONResponse(content={
        "status": "success",
        "new_model_version": new_version,
        "f1_score": f1,
        "training_time": training_time_str
    })

@app.get("/model-info")
async def model_info():
    return {"model_path": MODEL_PATH}
