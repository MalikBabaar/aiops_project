## REAMDE 

# JBoss Logs Anomaly Detection with Isolation Forest

This project implements an anomaly detection pipeline for JBoss server logs using an Isolation Forest model. It includes log parsing, feature extraction, model training, and serving predictions via a REST API.

---

## Overview

The goal is to detect unusual or suspicious events in JBoss server logs to help with proactive monitoring and incident response.

---

## Workflow

### 1. Fetch Logs from JBoss Server

- Obtain raw logs from your JBoss server (e.g., `server.log`).
- Logs can be collected via SSH, file transfer, or using log shippers like Filebeat or Fluentd.

### 2. Parse Logs with Regex

- Use Regular Expressions (regex) to extract structured data from unstructured log lines.
- Extract key fields such as:
  - **Timestamp** (e.g., `2025-09-20T17:42:41.378092+00:00`)
  - **Host / Server name**
  - **Component and Process ID (PID)** (e.g., `sshd[1234]`)
  - **Message content**

- Example regex pattern (Python):

  ```python
  pattern = r'(?P<timestamp>\S+) (?P<host>\S+) (?P<component>\S+)\[(?P<pid>\d+)\]: (?P<message>.*)'


3. Feature Extraction

From parsed logs, extract these features for modeling:

Feature	Description
hour	Hour of the day (0-23) from timestamp
dayofweek	Day of the week (0=Monday to 6=Sunday) from timestamp
pid	Process ID extracted from the component
auth_failure	Binary flag — 1 if message contains "authentication failure"
failed_password	Binary flag — 1 if message contains "failed password"

4. Proxy Labels for Evaluation (Optional)

Generate labels based on presence of keywords:

Label = 1 if auth_failure or failed_password is detected.

Label = 0 otherwise.

These help evaluate the anomaly detection performance without manual labels.

5. Model Training

Train an Isolation Forest on the extracted features:

from sklearn.ensemble import IsolationForest

features = df[['hour', 'dayofweek', 'pid', 'auth_failure', 'failed_password']]
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(features)


Save the trained model for inference.

6. Anomaly Detection (Inference)

Parse incoming logs, extract features, and compute anomaly scores.

Classify logs as anomalies based on a threshold.

Example output JSON:

{
  "timestamp": "2025-09-25T12:34:56Z",
  "log": "JBoss server started successfully",
  "anomaly_score": 0.02,
  "is_anomaly": false
}

7. API Endpoint

A FastAPI server exposes /analyze endpoint:

Accepts log lines as POST requests.

Returns anomaly prediction JSON.

/retrain endpoint allows retraining the model with new data.


## Setup and Run Instructions

Install dependencies (if needed):

pip install -r requirements.txt


Build the Docker image:

docker build -t aiops_backend .


Run the Docker container with environment variables from .env:

docker run -d -p 5000:5000 --env-file .env aiops_backend
