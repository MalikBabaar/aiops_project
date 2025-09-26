import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

# Load labeled dataset for evaluation
df = pd.read_csv('jboss_logs_with_true_labels.csv')

# Feature extraction (make sure it's the same as in training)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['auth_failure'] = df['message'].str.contains('authentication failure', case=False).astype(int)
df['failed_password'] = df['message'].str.contains('failed password', case=False).astype(int)
df['pid'] = pd.to_numeric(df['pid'], errors='coerce').fillna(0)

features = df[['hour', 'dayofweek', 'auth_failure', 'failed_password', 'pid']]

# Load trained model
clf = joblib.load('isolation_forest_model.joblib')

# Predict anomalies
df['anomaly'] = clf.predict(features)
df['pred_label'] = df['anomaly'].map({1: 0, -1: 1})  # Convert to 0=normal, 1=anomaly

# True labels column name must be 'true_label' with 0/1
y_true = df['true_label']
y_pred = df['pred_label']

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_true, y_pred))
