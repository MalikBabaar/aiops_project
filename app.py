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

# Load config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = os.getenv("MODEL_PATH", config.get("model_path", "./isolation_forest_model.joblib"))
API_PORT = int(os.getenv("API_PORT", config.get("api_port", 5000)))
LOG_PATH = os.getenv("LOG_PATH", config.get("log_path", "./data/logs"))
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", config.get("anomaly_threshold", 0)))

app = FastAPI()

# Load model once at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

class LogInput(BaseModel):
    log: str

def extract_features_from_log(log: str) -> pd.DataFrame:
    """
    Extract features from a single log line.
    Features:
      - hour of log
      - day of week
      - auth_failure (binary)
      - failed_password (binary)
      - pid (integer if found)
    """
    try:
        # Try to parse ISO timestamp at start
        timestamp_str = log.split(' ')[0]
        timestamp = pd.to_datetime(timestamp_str, errors='coerce')
        if pd.isna(timestamp):
            timestamp = pd.Timestamp.now()

        hour = timestamp.hour
        dayofweek = timestamp.dayofweek

        # PID extraction - match number inside brackets e.g., component[1234]
        pid_match = re.search(r'\[(\d+)\]', log)
        pid = int(pid_match.group(1)) if pid_match else 0

        # Extract message after timestamp and hostname (roughly)
        # Here assumes second token is hostname, adjust if needed
        tokens = log.split(' ')
        message_part = ' '.join(tokens[2:]) if len(tokens) > 2 else ''

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
    except Exception as e:
        # Log or raise error if needed
        raise ValueError(f"Failed to extract features: {e}")

@app.post("/analyze")
async def analyze(payload: LogInput):
    """
    Analyze a single log entry, return anomaly score and label.
    """
    try:
        features_df = extract_features_from_log(payload.log)
        score = model.decision_function(features_df)[0]
        is_anomaly = score < ANOMALY_THRESHOLD
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing log: {e}")

    return JSONResponse(content={
        "log": payload.log,
        "anomaly_score": score,
        "is_anomaly": bool(is_anomaly)
    })

@app.post("/retrain")
async def retrain(new_logs: list[LogInput]):
    """
    Retrain the Isolation Forest model with new logs.
    Note: This will overwrite the existing model.
    """
    from sklearn.ensemble import IsolationForest

    try:
        features_list = [extract_features_from_log(log.log) for log in new_logs]
        df_features = pd.concat(features_list, ignore_index=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse logs for retraining: {e}")

    try:
        new_model = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
        new_model.fit(df_features)
        joblib.dump(new_model, MODEL_PATH)

        global model
        model = new_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

    return {"detail": f"Model retrained on {len(new_logs)} logs."}
