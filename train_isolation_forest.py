import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib  # for saving the model

# Load extracted features CSV or run feature extraction here (reuse your feature_extraction.py code)

# For simplicity, read from CSV if you saved features, or run feature extraction code inline
df = pd.read_csv('jboss_parsed_logs.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['auth_failure'] = df['message'].str.contains('authentication failure', case=False).astype(int)
df['failed_password'] = df['message'].str.contains('failed password', case=False).astype(int)
df['pid'] = pd.to_numeric(df['pid'], errors='coerce').fillna(0)

features = df[['hour', 'dayofweek', 'auth_failure', 'failed_password', 'pid']]

# Train Isolation Forest
clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
clf.fit(features)

# Predict anomalies (1 = normal, -1 = anomaly)
df['anomaly'] = clf.predict(features)

# Save model for later use
joblib.dump(clf, 'isolation_forest_model.joblib')

# Save results with anomaly labels
df.to_csv('jboss_logs_with_anomalies.csv', index=False)

print("Training complete. Anomaly detection results saved to jboss_logs_with_anomalies.csv")
