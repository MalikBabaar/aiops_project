from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, Optional
import json
import numbers
import mlflow


def _is_number(x: Any) -> bool:
    """Return True for int/float-like values (incl. numpy scalar types)."""
    if isinstance(x, numbers.Number):
        return True
    try:
        # attempt float cast for numpy scalars
        float(x)
        return True
    except Exception:
        return False


def _safe_log_metrics(d: Dict[str, Any]) -> None:
    """
    Log only numeric metrics to MLflow; ignore keys with None / non-numeric values.
    """
    numeric = {}
    for k, v in d.items():
        if v is None:
            continue
        if _is_number(v):
            try:
                numeric[k] = float(v)
            except Exception:
                # skip non-castable values silently
                pass
    if numeric:
        mlflow.log_metrics(numeric)


def _log_artifacts_if_exist(paths: Iterable[Path]) -> None:
    """Best-effort logging of artifacts; skip any that are missing."""
    for p in paths:
        try:
            if p and Path(p).exists():
                mlflow.log_artifact(str(p))
        except Exception:
            # never fail the run because of artifact upload
            pass


def log_mlflow_metrics(
    metrics: Dict[str, Any],
    outdir: Path,
    experiment_name: str = "aiops-anomaly-intelligence",
    extra_artifacts: Optional[Iterable[Path]] = None,
) -> Optional[str]:
    """
    Log training and analytics outputs to MLflow and persist run_summary.json.

    Parameters
    ----------
    metrics : dict
        Should include keys like: threshold, anomaly_count, anomaly_rate,
        precision, recall, f1, duplicate_anomalies, rare_query_anomalies,
        atypical_combo_anomalies, total_records, etc.
    outdir : Path
        Directory where the pipeline saved plots and files.
    experiment_name : str
        MLflow experiment name (defaults to 'aiops-anomaly-intelligence').
    extra_artifacts : iterable of Path, optional
        Any additional files to attach (if they exist).

    Returns
    -------
    run_id : Optional[str]
        The MLflow run id if available (None if something prevented retrieval).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Persist/refresh run_summary.json on disk (helpful for offline debugging)
    summary_path = outdir / "run_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        # Do not interrupt MLflow logging if local write fails
        pass

    # Start the MLflow run and log
    mlflow.set_experiment(experiment_name)
    run_id: Optional[str] = None
    with mlflow.start_run() as run:
        # 1) General / numeric metrics
        _safe_log_metrics(metrics)

        # 2) Attach all known artifacts from the pipeline (if present)
        default_artifacts = [
            outdir / "feature_corr.png",
            outdir / "anomaly_bursts.png",
            outdir / "duplicate_ids.png",
            outdir / "rare_queries.png",
            outdir / "gap_anomalies.png",
            outdir / "combo_anomalies.png",
            outdir / "sample_anomalies.csv",
            outdir / "scored.csv",
            outdir / "model.joblib",
            outdir / "scaler.joblib",
            outdir / "freq_table.parquet",
            summary_path,
        ]

        # Include any extra artifacts if the caller passes them
        if extra_artifacts:
            default_artifacts.extend(list(extra_artifacts))

        _log_artifacts_if_exist(default_artifacts)

        # 3) Return the run id to the caller
        try:
            run_id = run.info.run_id
        except Exception:
            run_id = None

    return run_id
