import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data with features and proxy label
df = pd.read_csv('jboss_parsed_logs.csv')

# Convert timestamp to datetime if needed
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Extract features again or load if already present
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

df['auth_failure'] = df['message'].str.contains('authentication failure', case=False).astype(int)
df['failed_password'] = df['message'].str.contains('failed password', case=False).astype(int)
df['pid'] = pd.to_numeric(df['pid'], errors='coerce').fillna(0)

features = df[['hour', 'dayofweek', 'auth_failure', 'failed_password', 'pid']]

# Use your proxy label column created earlier as true labels
# For example, from create_proxy_labels.py output:
# proxy_anomaly_label = 1 means anomaly, 0 means normal
if 'proxy_anomaly_label' not in df.columns:
    raise ValueError("Missing 'proxy_anomaly_label' column in data. Please create it first.")

true_labels = df['proxy_anomaly_label']

# Train Isolation Forest
clf = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

clf.fit(features)

# Predict anomalies: -1 means anomaly, 1 means normal
preds = clf.predict(features)

# Convert preds to 0 = normal, 1 = anomaly for comparison
preds_binary = (preds == -1).astype(int)

# Evaluate model performance
print("Model Performance on Training Data (using proxy labels):")
print("Accuracy:", accuracy_score(true_labels, preds_binary))
print(classification_report(true_labels, preds_binary, digits=4))

# Save predictions in dataframe
df['anomaly_pred'] = preds_binary

# Save model and results
joblib.dump(clf, 'isolation_forest_model.joblib')
df.to_csv('jboss_logs_with_predictions.csv', index=False)

print("Training complete. Model and results saved.")
