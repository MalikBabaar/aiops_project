import pandas as pd

# Load logs
df = pd.read_csv("jboss_parsed_logs.csv")

# Initialize all as normal
df["proxy_anomaly_label"] = 0

# Network/Connection Issues
df.loc[df["message"].str.contains("connection refused", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("connection reset", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("connection timeout", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("network unreachable", case=False, na=False), "proxy_anomaly_label"] = 1

# Security/Authentication Issues
df.loc[df["message"].str.contains("invalid user", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("unauthorized", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("permission denied", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("access denied", case=False, na=False), "proxy_anomaly_label"] = 1

# System Failures
df.loc[df["message"].str.contains("out of memory", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("segmentation fault", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("kernel panic", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("disk full", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("no space left", case=False, na=False), "proxy_anomaly_label"] = 1

# Application Failures
df.loc[df["message"].str.contains("stack trace", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("null pointer", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("cannot allocate", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("deadlock detected", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("database error", case=False, na=False), "proxy_anomaly_label"] = 1

# Service/Deployment Issues
df.loc[df["message"].str.contains("service unavailable", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("failed to start", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("failed to initialize", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("deployment failed", case=False, na=False), "proxy_anomaly_label"] = 1
df.loc[df["message"].str.contains("startup failure", case=False, na=False), "proxy_anomaly_label"] = 1


# Save new dataset
df.to_csv("jboss_parsed_logs_with_labels.csv", index=False)
print("✅ Proxy labels added. Saved as jboss_parsed_logs_with_labels.csv")
