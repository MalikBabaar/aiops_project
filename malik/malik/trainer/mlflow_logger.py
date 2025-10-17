# trainer/mlflow_logger.py
from pathlib import Path
import mlflow
import json
import matplotlib.pyplot as plt

def log_mlflow_metrics(metrics: dict, outdir: Path):
    """
    Log training metrics, model metrics, and analytics to MLflow,
    save a JSON summary, and create basic plots for dashboard display.
    """
    # Ensure output directory exists
    outdir.mkdir(parents=True, exist_ok=True)

    # --- MLflow logging ---
    mlflow.set_experiment("aiops-anomaly-intelligence")
    with mlflow.start_run():
        # General metrics
        general_metrics = {
            "threshold": metrics.get("threshold", 0),
            "anomaly_count": metrics.get("anomaly_count", 0),
            "anomaly_rate": metrics.get("anomaly_rate", 0)
        }
        # Include feature rates if present
        if "feature_rates" in metrics and isinstance(metrics["feature_rates"], dict):
            general_metrics.update(metrics["feature_rates"])
        mlflow.log_metrics(general_metrics)

        # Classification metrics
        for metric in ["precision", "recall", "f1", "accuracy"]:
            value = metrics.get(metric)
            if value is not None:  # only log if value exists
                mlflow.log_metric(metric, value)

        # Analytics metrics
        for key in ["duplicate_anomalies", "rare_query_anomalies",
                    "atypical_combo_anomalies", "avg_gap_anomalies"]:
            if key in metrics:
                mlflow.log_metric(key, metrics[key])

        # --- Save JSON summary ---
        summary_file = outdir / "run_summary.json"
        with open(summary_file, "w") as f:
            json.dump(metrics, f, indent=4)

        # --- Create placeholder plots ---
        feature_corr_file = outdir / "feature_corr.png"
        if not feature_corr_file.exists():
            plt.figure(figsize=(5,4))
            plt.title("Feature Correlation")
            plt.plot([1,2,3,4], [4,3,2,1], marker='o')
            plt.xlabel("Feature")
            plt.ylabel("Correlation")
            plt.tight_layout()
            plt.savefig(feature_corr_file)
            plt.close()

        anomaly_bursts_file = outdir / "anomaly_bursts.png"
        if not anomaly_bursts_file.exists():
            plt.figure(figsize=(5,4))
            plt.title("Anomaly Bursts")
            plt.plot([1,2,3,4], [2,4,1,3], marker='x')
            plt.xlabel("Time")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(anomaly_bursts_file)
            plt.close()

        # --- Log artifacts to MLflow ---
        for artifact in [summary_file, feature_corr_file, anomaly_bursts_file]:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))