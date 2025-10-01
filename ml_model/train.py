import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib  # to save model

def train_isolation_forest(processed_csv_path, model_output_path):
    # Load the preprocessed data
    df = pd.read_csv(processed_csv_path)

    # Select features for anomaly detection
    # You can customize features here as needed
    features = ['hour', 'dayofweek', 'auth_failure', 'failed_password', 'pid']

    X = df[features]

    # Train Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(X)

    # Add anomaly scores and predictions to dataframe
    df['anomaly_score'] = model.decision_function(X)  # higher = normal, lower = anomaly
    df['anomaly'] = model.predict(X)  # -1 = anomaly, 1 = normal

    # Save model for later use
    joblib.dump(model, model_output_path)
    print(f"Isolation Forest model saved to {model_output_path}")

    # Save results with anomaly info
    df.to_csv("jboss_logs_with_anomalies.csv", index=False)
    print("Saved logs with anomaly predictions to jboss_logs_with_anomalies.csv")

if __name__ == "__main__":
    train_isolation_forest("jboss_logs_processed.csv", "isolation_forest_model.joblib")
