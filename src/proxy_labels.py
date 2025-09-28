import pandas as pd

# Load your parsed logs with features
df = pd.read_csv('jboss_parsed_logs.csv')

# Convert timestamp to datetime if not done yet
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract or recalculate features
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['auth_failure'] = df['message'].str.contains('authentication failure', case=False).astype(int)
df['failed_password'] = df['message'].str.contains('failed password', case=False).astype(int)
df['pid'] = pd.to_numeric(df['pid'], errors='coerce').fillna(0)

# Create a proxy anomaly label: 1 if either auth_failure or failed_password detected, else 0
df['proxy_anomaly_label'] = ((df['auth_failure'] == 1) | (df['failed_password'] == 1)).astype(int)

print(df[['message', 'auth_failure', 'failed_password', 'proxy_anomaly_label']].head())

# Now you can use 'proxy_anomaly_label' as your true label for evaluation
df.to_csv('jboss_parsed_logs.csv', index=False)
print("Proxy labels added and saved to 'jboss_parsed_logs.csv'")
