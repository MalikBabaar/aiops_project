from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import re

app = FastAPI()

# Load your trained model
model = joblib.load("isolation_forest_model.joblib")

# Regex pattern for parsing logs
pattern = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[\+\-]\d{2}:\d{2})\s+'
    r'(?P<host>[^\s]+)\s+'
    r'(?P<component>[^\s]+)\[(?P<pid>\d+)\]:\s+'
    r'(?P<message>.*)$'
)

class LogLine(BaseModel):
    log: str

def extract_features(log_line: str) -> pd.DataFrame:
    match = pattern.match(log_line)
    if not match:
        raise ValueError("Log line format not recognized")

    data = match.groupdict()
    timestamp = pd.to_datetime(data['timestamp'])
    message = data['message']

    # Feature extraction logic similar to your previous script
    hour = timestamp.hour
    dayofweek = timestamp.dayofweek
    auth_failure = int('authentication failure' in message.lower())
    failed_password = int('failed password' in message.lower())
    pid = int(data['pid'])

    features = pd.DataFrame([{
        'hour': hour,
        'dayofweek': dayofweek,
        'auth_failure': auth_failure,
        'failed_password': failed_password,
        'pid': pid
    }])
    return features

@app.post("/analyze")
def analyze_log(log_line: LogLine):
    try:
        features = extract_features(log_line.log)
        pred = model.predict(features)
        # IsolationForest returns -1 for anomaly, 1 for normal
        score = int(pred[0])
        return {"anomaly": score}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
