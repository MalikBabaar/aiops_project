from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, Optional
import os
import json
import numbers
import mlflow
from mlflow.tracking import MlflowClient


def _is_number(x: Any) -> bool:
    """Return True for int/float-like values (incl. numpy scalar types)."""
    if isinstance(x, numbers.Number):
        return True
    try:
            float(x)  # attempt float cast for numpy scalars/strings like "1.0"
            return True
    except Exception:
        return False


def _safe_log_metrics(d: Dict[str, Any]) -> None:
    """Log only numeric metrics to MLflow; ignore keys with None / non-numeric values."""
    numeric = {}
    for k, v in d.items():
        if v is None:
            continue
        if _is_number(v):
            try:
                numeric[k] = float(v)
            except Exception:
                pass  # skip non-castable silently
    if numeric:
        mlflow.log_metrics(numeric)


def _log_artifact_if_exists(path: Path, artifact_path: Optional[str] = None) -> None:
    """Best-effort logging of a single artifact; skip any that are missing."""
    try:
        if path and Path(path).is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception:
        # never fail the run because of artifact upload
        pass


def build_mlflow_run_url(run_id: str) -> Optional[str]:
    """
    Construct a clickable MLflow UI URL for a run if MLFLOW_TRACKING_URI is http(s).
    Example: http://localhost:5001/#/experiments/<exp_id>/runs/<run_id>
    """
    try:
        tracking = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        if tracking.startswith(("http://", "https://")):
            client = MlflowClient()
            run = client.get_run(run_id)
            exp_id = run.info.experiment_id
            return f"{tracking}/#/experiments/{exp_id}/runs/{run_id}"
    except Exception:
        pass
    return None


def log_mlflow_metrics(
    metrics: Dict[str, Any],
    outdir: Path,
    experiment_name: str = "aiops-anomaly-intelligence",
    extra_artifacts: Optional[Iterable[Path]] = None,
) -> Optional[str]:
    """
    Log training and analytics outputs to MLflow and persist run_summary.json.

    Returns the MLflow run_id (or None on best-effort failure).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Persist/update run_summary.json locally (handy for debugging)
    summary_path = outdir / "run_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass  # do not interrupt MLflow logging if local write fails

    # Start the MLflow run and log
    mlflow.set_experiment(experiment_name)
    run_id: Optional[str] = None

    with mlflow.start_run() as run:
        # ---- Params: keep a few as params for easier scanning in UI
        params = {}
        if "threshold" in metrics and _is_number(metrics["threshold"]):
            params["threshold"] = float(metrics["threshold"])
        if "metrics_source" in metrics and metrics["metrics_source"] is not None:
            params["metrics_source"] = str(metrics["metrics_source"])
        if params:
            mlflow.log_params(params)

        # ---- Metrics: numeric-only
        _safe_log_metrics(metrics)

        # ---- Artifacts
        # Group plots + sample CSV under "static-artifacts" for a clean UI
        static_names = [
            "feature_corr.png",
            "anomaly_bursts.png",
            "duplicate_ids.png",
            "rare_queries.png",
            "gap_anomalies.png",
            "combo_anomalies.png",
            "sample_anomalies.csv",
        ]
        for name in static_names:
            _log_artifact_if_exists(outdir / name, artifact_path="static-artifacts")

        # Root-level artifacts (or choose a different folder if you prefer)
        root_names = ["scored.csv", "model.joblib", "scaler.joblib", "freq_table.parquet"]
        for name in root_names:
            _log_artifact_if_exists(outdir / name)

        # Always log the summary JSON
        _log_artifact_if_exists(summary_path)

        # Any extras supplied by the caller
        if extra_artifacts:
            for p in extra_artifacts:
                _log_artifact_if_exists(Path(p))

        # Return run id to caller
        try:
            run_id = run.info.run_id
        except Exception:
            run_id = None

    return run_id