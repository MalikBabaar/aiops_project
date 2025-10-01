import pandas as pd
import os

def preprocess_jboss_logs(input_path, output_path):
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['auth_failure'] = df['message'].str.contains('authentication failure', case=False).astype(int)
    df['failed_password'] = df['message'].str.contains('failed password', case=False).astype(int)
    df['pid'] = pd.to_numeric(df['pid'], errors='coerce').fillna(0)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_jboss_logs("../data/jboss_logs_sample.csv", "jboss_logs_processed.csv")
