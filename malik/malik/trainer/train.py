from __future__ import annotations
import argparse, json, os, sys, math, joblib, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path

# --- Headless-safe plotting backend BEFORE importing pyplot ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,  # needed by choose_threshold
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow import sklearn as ml_sklearn
from mlflow.tracking import MlflowClient
from typing import Dict, List, Tuple, Optional

from malik.malik.trainer.mlflow_logger import log_mlflow_metrics

# ----------------------- MLflow Setup -----------------------
# Tracking URI from env or default
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
try:
    mlflow.set_tracking_uri(MLFLOW_URI)
except Exception:
    # Don't crash if URI not reachable at import time
    pass

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "aiops-anomaly-intelligence")

RANDOM_STATE = 42
NUMERIC_FEATS = ["status_encoded", "query_encoded", "timestamp_burst"]
BINARY_FEATS = [
    "is_request", "is_response", "has_error", "error_spike",
    "duplicate_id", "rare_query", "atypical_combo"
]
ALL_FEATS = NUMERIC_FEATS + BINARY_FEATS
def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


def map_status(x):
    try:
        x = int(x)
    except Exception:
        return 3  # unknown
    if 200 <= x < 300: return 0
    if 400 <= x < 500: return 1
    if 500 <= x < 600: return 2
    return 3


def build_features(df: pd.DataFrame, freq_table=None):
    """
    Feature builder used by both CLI (main) and retrain wrapper.
    Be defensive: ensure 'service' and 'query' exist even if caller didn't normalize.
    """
    df = df.copy()

    # --- Defensive defaults (needed when CLI main() calls build_features directly) ---
    if "service" not in df.columns:
        df["service"] = "unknown"
    if "query" not in df.columns:
        df["query"] = "unknown"

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Basic hygiene
    for col in ["is_request", "is_response", "has_error", "error_spike"]:
        if col not in df: df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    # Duplicate detection (15 min window)
    df["duplicate_id"] = 0
    if "request_id" in df:
        df["dup_key"] = df["service"].astype(str) + "\n" + df["request_id"].astype(str)
        df["prev_ts"] = df.groupby("dup_key")["timestamp"].shift(1)
        df["delta"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds()
        df.loc[df["delta"].notna() & (df["delta"] <= 900), "duplicate_id"] = 1
        df.drop(columns=["dup_key", "prev_ts", "delta"], inplace=True, errors="ignore")

    # Burst per service
    df["prev_ts_svc"] = df.groupby("service")["timestamp"].shift(1)
    df["timestamp_burst"] = (df["timestamp"] - df["prev_ts_svc"]).dt.total_seconds().fillna(0.0)
    df.drop(columns=["prev_ts_svc"], inplace=True, errors="ignore")

    # Encodings
    df["status_encoded"] = df.get("status_code", pd.Series(["unknown"] * len(df))).apply(map_status)

    if freq_table is None:
        fq = (df.groupby(["service", "query"]).size()
              .rename("count").reset_index())
        fq["logfreq"] = np.log1p(fq["count"])
        freq_table = fq[["service", "query", "logfreq"]]

    df = df.merge(freq_table, how="left", on=["service", "query"])
    df["query_encoded"] = df["logfreq"].fillna(0.0)
    df.drop(columns=["logfreq"], inplace=True, errors="ignore")

    # Rare query
    df["rank_pct"] = df.groupby("service")["query_encoded"].rank(pct=True, method="first")
    df["rare_query"] = (df["rank_pct"] <= 0.05).astype(int)
    df.drop(columns=["rank_pct"], inplace=True, errors="ignore")

    # Atypical combo
    df["atypical_combo"] = ((df["has_error"] == 1) & (df["rare_query"] == 1)).astype(int)

    for c in BINARY_FEATS:
        if c not in df: df[c] = 0
        df[c] = df[c].astype(int).fillna(0)
    for c in NUMERIC_FEATS:
        if c not in df: df[c] = 0.0
        df[c] = df[c].astype(float).fillna(0.0)

    return df, freq_table


def load_frames(paths):
    frames = []
    for path in paths:
        try:
            p = Path(path)
            if not p.exists():
                print(f"⚠️ path not found: {p}")
                continue
            print(f"✅ loading: {p}")
            try:
                df = pd.read_csv(p, engine="python", on_bad_lines="skip")
            except Exception:
                df = pd.read_csv(p, names=["log"], engine="python", on_bad_lines="skip")
            frames.append(df)
            print(f" rows: {len(df)}")
        except Exception as e:
            print(f"❌ failed to read {path}: {e}")
    return frames


def train_isoforest(X_train):
    model = IsolationForest(
        n_estimators=300, contamination=0.01,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train)
    raw = -model.score_samples(X_train)
    return model, raw


def choose_threshold(raw_scores, labels=None):
    if labels is not None:
        qs = np.quantile(raw_scores, np.linspace(0.80, 0.999, 50))
        best = (None, -1, None)
        for t in qs:
            pred = (raw_scores >= t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
            if f1 > best[1]:
                best = (t, f1, (p, r))
        thr, f1, pr = best
        if thr is not None:
            return float(thr), {"f1": float(f1), "precision": float(pr[0]), "recall": float(pr[1])}
    thr = float(np.quantile(raw_scores, 0.98))
    return thr, None


# ---------------------- Plotting Functions ----------------------
def save_feature_correlation(df, outdir):
    corr = df[NUMERIC_FEATS + BINARY_FEATS].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Extended Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(outdir / "feature_corr.png")
    plt.close()


def save_anomaly_bursts(df, outdir):
    anomalies = df[df["anomaly_flag"] == 1]
    if anomalies.empty:
        # placeholder
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No anomalies", ha="center", va="center")
        plt.axis("off")
        plt.savefig(outdir / "anomaly_bursts.png")
        plt.close()
        return
    counts = anomalies.groupby(pd.Grouper(key="timestamp", freq="h")).size()
    plt.figure(figsize=(12, 4))
    counts.plot(kind="bar")
    plt.title("Anomaly Bursts Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "anomaly_bursts.png")
    plt.close()


def plot_duplicate_ids(df, outdir):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="duplicate_id", data=df)
    plt.title("Duplicate IDs and Anomalies")
    plt.tight_layout()
    plt.savefig(outdir / "duplicate_ids.png")
    plt.close()


def plot_rare_queries(df, outdir):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="rare_query", data=df)
    plt.title("Rare Queries and Anomalies")
    plt.tight_layout()
    plt.savefig(outdir / "rare_queries.png")
    plt.close()


def plot_gap_anomalies(df, outdir):
    anomalies = df[df["anomaly_flag"] == 1].sort_values("timestamp")
    if len(anomalies) > 1:
        gaps = anomalies["timestamp"].diff().dt.total_seconds().dropna()
        plt.figure(figsize=(8, 4))
        sns.histplot(gaps, bins=30, kde=True)
        plt.title("Time Gaps Between Logs (Anomalies Only)")
        plt.xlabel("Gap (seconds)")
        plt.tight_layout()
        plt.savefig(outdir / "gap_anomalies.png")
        plt.close()
    else:
        plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "Not enough anomalies for gap histogram", ha="center", va="center")
        plt.axis("off")
        plt.savefig(outdir / "gap_anomalies.png")
        plt.close()


def plot_combo_anomalies(df, outdir):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="atypical_combo", data=df)
    plt.title("Atypical Error + Rare Query Combinations")
    plt.tight_layout()
    plt.savefig(outdir / "combo_anomalies.png")
    plt.close()


def log_sample_anomalies(df, outdir):
    sample_cols = ["timestamp", "service", "query", "anomaly_score",
                   "duplicate_id", "rare_query", "atypical_combo", "status_code", "request_id"]
    df_sample = df.sort_values("anomaly_score", ascending=False).head(10)
    df_sample = df_sample[[c for c in sample_cols if c in df_sample.columns]]
    df_sample.to_csv(outdir / "sample_anomalies.csv", index=False)


# ---------------------------- CLI main ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", help="Training data paths")
    ap.add_argument("--outdir", default="./run")
    ap.add_argument("--label-col", default="anomaly_tag")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Auto-detect both logcurr files (parent folder of trainer)
    input_paths = args.inputs or ["../logcurr.csv", "../logcurr.txt"]
    print("DEBUG: candidate input paths:", input_paths)

    frames = load_frames(input_paths)
    if not frames:
        raise FileNotFoundError(f"No input files found. Checked: {input_paths}")

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded dataframe with {len(df)} rows from {len(frames)} file(s)")

    # Build features (defensive guards inside)
    df, freq_table = build_features(df)

    y = None
    if args.label_col in df.columns:
        y = (df[args.label_col].astype(str).str.lower() == "anomaly").astype(int).values
        print("INFO: Labels found in data; using them for metric calculation.")

    feats = df[ALL_FEATS].values
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    if y is None:
        Xtr, Xte = train_test_split(feats_scaled, test_size=0.2, random_state=RANDOM_STATE)
        ytr = yte = None
    else:
        Xtr, Xte, ytr, yte = train_test_split(feats_scaled, y, test_size=0.2,
                                             random_state=RANDOM_STATE, stratify=y)

    model, raw_tr = train_isoforest(Xtr)
    thr, train_metrics = choose_threshold(raw_tr, ytr)

    df["anomaly_score"] = -model.score_samples(feats_scaled)
    if thr is None:
        thr = float(np.quantile(df["anomaly_score"].values, 0.98))
    df["anomaly_flag"] = (df["anomaly_score"] >= thr).astype(int)

    # metrics
    metrics = {
        "threshold": float(thr),
        "total_records": int(len(df)),
        "anomaly_count": int(df["anomaly_flag"].sum()),
        "anomaly_rate": float(df["anomaly_flag"].mean()),
        "duplicate_anomalies": int(df["duplicate_id"].sum()),
        "rare_query_anomalies": int(df["rare_query"].sum()),
        "atypical_combo_anomalies": int(df["atypical_combo"].sum()),
        "metrics_source": None
    }

    precision = recall = f1 = None
    if y is not None:
        try:
            raw_te = -model.score_samples(Xte)
            y_pred_te = (raw_te >= thr).astype(int)
            if 'yte' in locals() and yte is not None:
                precision = precision_score(yte, y_pred_te, zero_division=0)
                recall = recall_score(yte, y_pred_te, zero_division=0)
                f1 = f1_score(yte, y_pred_te, zero_division=0)
                metrics["metrics_source"] = "true_labels_test"
        except Exception:
            pass
        try:
            y_all = y
            y_pred_all = df["anomaly_flag"].values
            precision = precision_score(y_all, y_pred_all, zero_division=0)
            recall = recall_score(y_all, y_pred_all, zero_division=0)
            f1 = f1_score(y_all, y_pred_all, zero_division=0)
            metrics["metrics_source"] = metrics.get("metrics_source") or "true_labels_all"
        except Exception:
            pass
    else:
        contamination = 0.01
        n_pseudo = max(1, int(len(df) * contamination))
        top_idx = np.argsort(-df["anomaly_score"].values)[:n_pseudo]
        y_pseudo = np.zeros(len(df), dtype=int)
        y_pseudo[top_idx] = 1
        y_pred = df["anomaly_flag"].values
        precision = precision_score(y_pseudo, y_pred, zero_division=0)
        recall = recall_score(y_pseudo, y_pred, zero_division=0)
        f1 = f1_score(y_pseudo, y_pred, zero_division=0)
        metrics["metrics_source"] = "pseudo_top_percent"

    metrics["precision"] = float(precision) if precision is not None else None
    metrics["recall"] = float(recall) if recall is not None else None
    metrics["f1"] = float(f1) if f1 is not None else None

    # Save analytics + plots
    save_feature_correlation(df, outdir)
    save_anomaly_bursts(df, outdir)
    plot_duplicate_ids(df, outdir)
    plot_rare_queries(df, outdir)
    plot_gap_anomalies(df, outdir)
    plot_combo_anomalies(df, outdir)
    log_sample_anomalies(df, outdir)

    # Log to MLflow and attach artifacts
    try:
        run_id = log_mlflow_metrics(metrics, outdir, experiment_name=EXPERIMENT_NAME)
        if run_id:
            metrics["run_id"] = run_id
    except Exception:
        # Keep CLI resilient
        pass

    # Save model & scored data
    try:
        joblib.dump(model, outdir / "model.joblib")
    except Exception:
        pass
    try:
        df.to_csv(outdir / "scored.csv", index=False)
    except Exception:
        pass

    print("✅ Analytics complete. Metrics and artifacts logged to", outdir.resolve())


# ---------------------- Robust retrain wrapper ----------------------
# 1) UNIVERSAL SCHEMA NORMALIZER
# Map many possible dataset column names to the columns your pipeline expects.
_COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "timestamp": ["timestamp", "time", "ts", "datetime", "date_time", "event_time", "log_time"],
    "service": ["service", "svc", "service_name", "app", "application", "component"],
    "query": ["query", "endpoint", "uri", "path", "route", "operation"],
    "status_code": ["status_code", "status", "http_status", "code", "resp_code"],
    "request_id": ["request_id", "req_id", "trace_id", "correlation_id", "rid"],
    "is_request": ["is_request", "request", "req_flag"],
    "is_response": ["is_response", "response", "resp_flag"],
    "has_error": ["has_error", "error", "is_error", "err"],
    "error_spike": ["error_spike", "spike", "error_spike_flag"],
    # optional label col – default '--label-col' is "anomaly_tag"
    "anomaly_tag": ["anomaly_tag", "label", "ground_truth", "target", "y", "anomaly_label"],
}


def normalize_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df with columns normalized to the names expected by your pipeline.
    Also coerces basic types for flags and parses timestamps to UTC.
    Ensures 'service', 'query', and 'timestamp' exist (safe defaults if missing).
    """
    df = df.copy()

    # Build lowercase map for case-insensitive matching
    lowermap = {c.lower(): c for c in df.columns}

    # Decide renaming
    rename_map = {}
    for expected, candidates in _COLUMN_SYNONYMS.items():
        for cand in candidates:
            if cand.lower() in lowermap:
                rename_map[lowermap[cand.lower()]] = expected
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    # --- Safe defaults for essentials ---
    if "service" not in df.columns:
        df["service"] = "unknown"
    if "query" not in df.columns:
        df["query"] = "unknown"
    if "timestamp" not in df.columns:
        # Create monotonic UTC timestamps to enable burst features
        base = pd.Timestamp.utcnow()
        df["timestamp"] = base + pd.to_timedelta(np.arange(len(df)), unit="s")

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Coerce binary flags to int {0,1}
    for b in ("is_request", "is_response", "has_error", "error_spike"):
        if b in df.columns:
            df[b] = (
                df[b]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["1", "true", "t", "yes", "y"])
            ).astype(int)

    return df


# 2) VALIDATOR
def validate_dataset(df: pd.DataFrame, required: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Validate minimal requirements for the training pipeline.
    Returns (ok, message). If not ok, message explains what's missing.
    """
    required = required or ["timestamp", "service"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}"
    # Check timestamp parsability (after normalization we coerced already)
    if df["timestamp"].isna().all():
        return False, "All timestamps are invalid or empty after parsing."
    return True, "ok"


# 3) RETRAIN WRAPPER
def retrain_model(
    df: pd.DataFrame,
    outdir: str | Path = "./run_streamlit",
    label_col: str = "anomaly_tag",
    mlflow_experiment: str = "aiops-anomaly-intelligence",
) -> Tuple[int, dict, dict]:

    """
    Retrain the IsolationForest using the given dataframe.
    Returns: (exit_code, metrics, artifacts)
    exit_code:
      0 = success
      1 = input/validation error
      2 = training/scoring error
      3 = MLflow logging failure (training succeeded)
    """
    outdir = Path(__file__).resolve().parent / "run_streamlit"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Normalize & validate dataset so we can handle many datasets robustly ---
    df_norm = normalize_dataset_columns(df)
    ok, msg = validate_dataset(df_norm)

    artifacts = {
        "feature_corr": outdir / "feature_corr.png",
        "anomaly_bursts": outdir / "anomaly_bursts.png",
        "duplicate_ids": outdir / "duplicate_ids.png",
        "rare_queries": outdir / "rare_queries.png",
        "gap_anomalies": outdir / "gap_anomalies.png",
        "combo_anomalies": outdir / "combo_anomalies.png",
        "sample_anomalies": outdir / "sample_anomalies.csv",
        "run_summary": outdir / "run_summary.json",
        "scored": outdir / "scored.csv",
        "model": outdir / "model.joblib",
        "scaler": outdir / "scaler.joblib",
        "freq_table": outdir / "freq_table.parquet",
    }

    if not ok:
        return 1, {"error": msg}, artifacts

    try:
        # --- Build features ---
        df_feats, freq_table = build_features(df_norm)
        # Persist freq table for consistent serving later
        try:
            freq_table.to_parquet(artifacts["freq_table"])
        except Exception:
            pass

        # Optional labels
        y = None
        if label_col in df_feats.columns:
            y = (df_feats[label_col].astype(str).str.lower() == "anomaly").astype(int).values

        # Matrix + scaling
        X = df_feats[ALL_FEATS].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        try:
            joblib.dump(scaler, artifacts["scaler"])
        except Exception:
            pass

        # Split
        if y is None:
            Xtr, Xte = train_test_split(X_scaled, test_size=0.2, random_state=RANDOM_STATE)
            ytr = yte = None
        else:
            Xtr, Xte, ytr, yte = train_test_split(
                X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )

        # Train & threshold
        model, raw_tr = train_isoforest(Xtr)
        thr, _ = choose_threshold(raw_tr, ytr)

        # Score entire dataset
        df_feats["anomaly_score"] = -model.score_samples(X_scaled)
        if thr is None:
            thr = float(np.quantile(df_feats["anomaly_score"].values, 0.98))
        df_feats["anomaly_flag"] = (df_feats["anomaly_score"] >= float(thr)).astype(int)

        # Metrics
        metrics = {
            "threshold": float(thr),
            "total_records": int(len(df_feats)),
            "anomaly_count": int(df_feats["anomaly_flag"].sum()),
            "anomaly_rate": float(df_feats["anomaly_flag"].mean()),
            # Sums of flags (feature-derived indicators)
            "duplicate_anomalies": int(df_feats["duplicate_id"].sum() if "duplicate_id" in df_feats else 0),
            "rare_query_anomalies": int(df_feats["rare_query"].sum() if "rare_query" in df_feats else 0),
            "atypical_combo_anomalies": int(df_feats["atypical_combo"].sum() if "atypical_combo" in df_feats else 0),
            "metrics_source": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

        # Evaluate (with labels or pseudo)
        try:
            if y is not None:
                try:
                    raw_te = -model.score_samples(Xte)
                    y_pred_te = (raw_te >= float(thr)).astype(int)
                    if yte is not None:
                        metrics["precision"] = float(precision_score(yte, y_pred_te, zero_division=0))
                        metrics["recall"] = float(recall_score(yte, y_pred_te, zero_division=0))
                        metrics["f1"] = float(f1_score(yte, y_pred_te, zero_division=0))
                        metrics["metrics_source"] = "true_labels_test"
                except Exception:
                    pass
                if metrics["metrics_source"] is None:
                    y_all = y
                    y_pred_all = df_feats["anomaly_flag"].values
                    metrics["precision"] = float(precision_score(y_all, y_pred_all, zero_division=0))
                    metrics["recall"] = float(recall_score(y_all, y_pred_all, zero_division=0))
                    metrics["f1"] = float(f1_score(y_all, y_pred_all, zero_division=0))
                    metrics["metrics_source"] = "true_labels_all"
            else:
                contamination = 0.01
                n_pseudo = max(1, int(len(df_feats) * contamination))
                top_idx = np.argsort(-df_feats["anomaly_score"].values)[:n_pseudo]
                y_pseudo = np.zeros(len(df_feats), dtype=int)
                y_pseudo[top_idx] = 1
                y_pred = df_feats["anomaly_flag"].values
                metrics["precision"] = float(precision_score(y_pseudo, y_pred, zero_division=0))
                metrics["recall"] = float(recall_score(y_pseudo, y_pred, zero_division=0))
                metrics["f1"] = float(f1_score(y_pseudo, y_pred, zero_division=0))
                metrics["metrics_source"] = "pseudo_top_percent"
        except Exception:
            metrics["metrics_source"] = metrics.get("metrics_source") or "metrics_failed"

        # Plots & artifacts
        try:
            save_feature_correlation(df_feats, outdir)
            save_anomaly_bursts(df_feats, outdir)
            plot_duplicate_ids(df_feats, outdir)
            plot_rare_queries(df_feats, outdir)
            plot_gap_anomalies(df_feats, outdir)
            plot_combo_anomalies(df_feats, outdir)
            log_sample_anomalies(df_feats, outdir)
        except Exception:
            pass
        # Save model & scored data
        try:
            joblib.dump(model, artifacts["model"])
        except Exception:
            pass
        try:
            df_feats.to_csv(artifacts["scored"], index=False)
        except Exception:
            pass
        try:
            with open(artifacts["run_summary"], "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            pass

        # MLflow logging (also writes run_summary.json in log_mlflow_metrics)
        try:
            run_id = log_mlflow_metrics(metrics, outdir, experiment_name=mlflow_experiment)
            if run_id:
                metrics["run_id"] = run_id
        except Exception:
            # Training succeeded but MLflow logging failed → return exit_code = 3
            return 3, metrics, {k: str(v) for k, v in artifacts.items()}

        # Success
        return 0, metrics, {k: str(v) for k, v in artifacts.items()}

    except Exception as e:
        # Unexpected failure path
        return 2, {"error": f"training_failed: {e}"}, {k: str(v) for k, v in artifacts.items()}


if __name__ == "__main__":
    main()