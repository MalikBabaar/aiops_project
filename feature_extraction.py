import pandas as pd

# Load the parsed logs
df = pd.read_csv('jboss_parsed_logs.csv')

# Convert timestamp to datetime object
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract hour of day and day of week
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

# Example: flag if message contains "authentication failure"
df['auth_failure'] = df['message'].str.contains('authentication failure', case=False).astype(int)

# Example: flag if message contains "Failed password"
df['failed_password'] = df['message'].str.contains('failed password', case=False).astype(int)

# Example: pid as numeric
df['pid'] = pd.to_numeric(df['pid'], errors='coerce').fillna(0)

# Drop columns you don't want for modeling
features = df[['hour', 'dayofweek', 'auth_failure', 'failed_password', 'pid']]

print(features.head())
