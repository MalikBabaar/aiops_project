import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_model(input_path, output_path):
    df = pd.read_csv(input_path)

    # Feature extraction
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['auth_failure'] = df['message'].str.contains('authentication failure', case=False).astype(int)
    df['failed_password'] = df['message'].str.contains('failed password', case=False).astype(int)
    df['pid'] = pd.to_numeric(df['pid'], errors='coerce').fillna(0)

    features = df[['hour', 'dayofweek', 'auth_failure', 'failed_password', 'pid']]
    labels = df['proxy_anomaly_label']

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(features)

    preds = model.predict(features)
    preds_binary = (preds == -1).astype(int)

    print("Accuracy:", accuracy_score(labels, preds_binary))
    print(classification_report(labels, preds_binary))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_v1.pkl")
    print("Model saved to models/model_v1.pkl")

if __name__ == "__main__":
    train_model("jboss_logs_processed.csv", "../models/model_v1.pkl")
