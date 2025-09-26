import os
import re
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import yaml
# Load environment variables if using dotenv
from dotenv import load_dotenv
load_dotenv()

# Load default config from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Now assign config variables, env vars take precedence if set
MODEL_PATH = os.getenv("MODEL_PATH", config.get("model_path", "./isolation_forest_model.joblib"))
API_PORT = int(os.getenv("API_PORT", config.get("api_port", 5000)))
LOG_PATH = os.getenv("LOG_PATH", config.get("log_path", "./data/logs"))
ANOMALY_THRESHOLD = float(config.get("anomaly_threshold", 0))

app = FastAPI()

# Load model at startup
model = joblib.load(MODEL_PATH)

# Threshold for anomaly (you can adjust this based on your model)
ANOMALY_THRESHOLD = 0  # Adjust as needed

# Pydantic model for input validation
class LogInput(BaseModel):
    log: str

def extract_features_from_log(log: str):
    # Extract timestamp (assumed ISO 8601 at start of log)
    timestamp_str = log.split(' ')[0]
    timestamp = pd.to_datetime(timestamp_str, errors='coerce')
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.now()

    hour = timestamp.hour
    dayofweek = timestamp.dayofweek

    # Extract pid from brackets, e.g., component[1234]
    pid_match = re.search(r'\[(\d+)\]', log)
    pid = int(pid_match.group(1)) if pid_match else 0

    # Extract message part (naively skipping timestamp and hostname)
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


from fastapi.responses import JSONResponse

@app.post("/analyze")
async def analyze(payload: LogInput):
    features_df = extract_features_from_log(payload.log)
    try:
        score = model.decision_function(features_df)[0]
        is_anomaly = bool(score < ANOMALY_THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    result = {
        "log": payload.log,
        "anomaly_score": score,
        "is_anomaly": is_anomaly
    }

    # Return JSONResponse
    return JSONResponse(content=result)


@app.post("/retrain")
async def retrain(new_logs: list[LogInput]):
    """
    Retrain model on new logs.

    Expecting list of logs: [{"log": "..."}, {"log": "..."}]
    """
    # Parse logs into features
    try:
        features_list = [extract_features_from_log(log.log) for log in new_logs]
        df_features = pd.concat(features_list, ignore_index=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse logs: {e}")

    # Here you’d load your training labels or generate labels
    # For anomaly detection usually unsupervised, so you might just fit again or update the model

    from sklearn.ensemble import IsolationForest
    new_model = IsolationForest(contamination=0.01, random_state=42)

    # Fit new model
    try:
        new_model.fit(df_features)
        # Save updated model
        joblib.dump(new_model, MODEL_PATH)
        global model
        model = new_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

    return {"detail": f"Model retrained on {len(new_logs)} logs."}
