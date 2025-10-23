import argparse, json, os, joblib, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from mlflow import sklearn
import mlflow
import sys


# ---------------- MLflow Setup ----------------
#mlruns_path = Path("C:/aiops_project/mlruns")
#mlruns_path.mkdir(parents=True, exist_ok=True)

#mlflow.set_tracking_uri(f"file:///{mlruns_path.as_posix()}")
#mlflow.set_experiment("aiops-anomaly-intelligence")

#mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, log_models=True, silent=True)

MLFLOW_URI = "http://mlflow:5001"
EXPERIMENT_NAME = "aiops-anomaly-intelligence"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()
try:
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        client.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as e:
    print(f"[⚠️ MLflow Warning] Could not create/set experiment: {e}")
    mlflow.set_experiment(EXPERIMENT_NAME)

def retrain_model(df_logs: pd.DataFrame = None, input_paths: list = None, contamination: float = 0.05, outdir: str = "./run"):
    """
    Full retraining pipeline for IsolationForest, including:
    - Data cleaning
    - Feature engineering
    - Scaling
    - Model training
    - Metrics computation
    - Plot generation
    - MLflow logging
    
    Can take a DataFrame directly or input file paths.
    Returns a dictionary with metrics, run_id, model & plots paths.
    """

    RANDOM_STATE = 42
    NUMERIC_FEATS = ["status_encoded", "query_encoded", "timestamp_burst"]
    BINARY_FEATS  = ["is_request", "is_response", "has_error", "error_spike",
                     "duplicate_id", "rare_query", "atypical_combo"]
    ALL_FEATS = NUMERIC_FEATS + BINARY_FEATS

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load from paths if df not provided ---
    if df_logs is None:
        frames = load_frames(input_paths or ["../logcurr.csv", "../logcurr.txt"])
        if not frames:
            raise FileNotFoundError(f"No input files found. Checked: {input_paths}")
        df_logs = pd.concat(frames, ignore_index=True)

    # --- 1️⃣ Data Cleaning ---
    # Ensure essential columns exist
    for col in ["service", "query", "request_id", "timestamp", "log"]:
        if col not in df_logs.columns:
            df_logs[col] = "unknown" if col not in ["timestamp", "log"] else pd.Timestamp.now() if col=="timestamp" else ""

    # Remove exact duplicates
    df_logs.drop_duplicates(subset=["request_id"], inplace=True, ignore_index=True)

    # Convert timestamp to datetime
    df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors="coerce")
    df_logs["timestamp"] = df_logs["timestamp"].fillna(pd.Timestamp.now())

    # Normalize text
    df_logs["log"] = df_logs["log"].astype(str).str.lower().str.strip()

    # Ensure binary columns exist and are 0/1
    for col in BINARY_FEATS:
        if col not in df_logs.columns:
            df_logs[col] = 0
        df_logs[col] = df_logs[col].astype(int)

    # --- 2️⃣ Feature Engineering ---
    df_logs, freq_table = build_features(df_logs)

    # --- 3️⃣ Scale features ---
    feats = df_logs[ALL_FEATS].values
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    # --- 4️⃣ Train IsolationForest ---
    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    model.fit(feats_scaled)
    df_logs["anomaly_score"] = -model.score_samples(feats_scaled)
    thr = float(np.quantile(df_logs["anomaly_score"], 1 - contamination))
    df_logs["anomaly_flag"] = (df_logs["anomaly_score"] >= thr).astype(int)

    # --- 5️⃣ Compute metrics ---
    if "label" in df_logs.columns and df_logs["label"].notna().any():
        y_true = df_logs["label"].astype(int).values
        y_pred = df_logs["anomaly_flag"].values
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        n_top = max(1, int(len(df_logs) * contamination))
        top_idx = np.argsort(-df_logs["anomaly_score"].values)[:n_top]
        y_pseudo = np.zeros(len(df_logs), dtype=int)
        y_pseudo[top_idx] = 1
        y_pred = df_logs["anomaly_flag"].values
        precision = precision_score(y_pseudo, y_pred, zero_division=0)
        recall = recall_score(y_pseudo, y_pred, zero_division=0)
        f1 = f1_score(y_pseudo, y_pred, zero_division=0)

    # --- 6️⃣ Prepare plots ---
    plots_paths = {}
    save_feature_correlation(df_logs, outdir); plots_paths["Extended Feature Correlation Heatmap"] = str(outdir / "feature_corr.png")
    save_anomaly_bursts(df_logs, outdir); plots_paths["Anomaly Bursts Over Time"] = str(outdir / "anomaly_bursts.png")
    plot_duplicate_ids(df_logs, outdir); plots_paths["Duplicate IDs and Anomalies"] = str(outdir / "duplicate_ids.png")
    plot_rare_queries(df_logs, outdir); plots_paths["Rare Queries and Anomalies"] = str(outdir / "rare_queries.png")
    plot_gap_anomalies(df_logs, outdir); plots_paths["Time Gaps Between Logs (Anomalies Only)"] = str(outdir / "gap_anomalies.png")
    plot_combo_anomalies(df_logs, outdir); plots_paths["Atypical Error + Rare Query Combinations"] = str(outdir / "combo_anomalies.png")

    # --- 7️⃣ Save model & vectorizer ---
    joblib.dump(model, outdir / "model.joblib")
    joblib.dump(TfidfVectorizer(max_features=5000).fit(df_logs["log"]), outdir / "vectorizer.joblib")

    # --- 8️⃣ MLflow logging ---
    mlflow.set_experiment("aiops-anomaly-intelligence")
    with mlflow.start_run() as run:
        mlflow.log_param("contamination", contamination)
        mlflow.log_metric("total_records", len(df_logs))
        mlflow.log_metric("anomaly_count", int(df_logs["anomaly_flag"].sum()))
        mlflow.log_metric("anomaly_rate", float(df_logs["anomaly_flag"].mean()))
        mlflow.log_metric("precision", float(precision))
        mlflow.log_metric("recall", float(recall))
        mlflow.log_metric("f1_score", float(f1))
        mlflow.log_artifact(outdir / "model.joblib")
        mlflow.log_artifact(outdir / "vectorizer.joblib")
        for path in plots_paths.values():
            mlflow.log_artifact(path)

    # --- 9️⃣ Return results ---
    return {
        "run_id": run.info.run_id,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_records": len(df_logs),
        "anomaly_count": int(df_logs["anomaly_flag"].sum()),
        "anomaly_rate": float(df_logs["anomaly_flag"].mean()),
        "duplicate_anomalies": int(df_logs["duplicate_id"].sum()),
        "rare_query_anomalies": int(df_logs["rare_query"].sum()),
        "atypical_combo_anomalies": int(df_logs["atypical_combo"].sum()),
        "plots_paths": plots_paths,
        "model_file": str(outdir / "model.joblib"),
        "vectorizer_file": str(outdir / "vectorizer.joblib")
    }


RANDOM_STATE = 42
NUMERIC_FEATS = ["status_encoded", "query_encoded", "timestamp_burst"]
BINARY_FEATS  = [
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
    except:
        return 3  # unknown
    if 200 <= x < 300: return 0
    if 400 <= x < 500: return 1
    if 500 <= x < 600: return 2
    return 3


def build_features(df: pd.DataFrame, freq_table=None):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Basic hygiene
    for col in ["is_request", "is_response", "has_error", "error_spike"]:
        if col not in df: df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    # Duplicate detection (15 min window)
    df["duplicate_id"] = 0
    if "request_id" in df:
        df["dup_key"] = df["service"].astype(str) + "|" + df["request_id"].astype(str)
        df["prev_ts"] = df.groupby("dup_key")["timestamp"].shift(1)
        df["delta"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds()
        df.loc[df["delta"].notna() & (df["delta"] <= 900), "duplicate_id"] = 1
        df.drop(columns=["dup_key", "prev_ts", "delta"], inplace=True)

    # Burst per service
    df["prev_ts_svc"] = df.groupby("service")["timestamp"].shift(1)
    df["timestamp_burst"] = (df["timestamp"] - df["prev_ts_svc"]).dt.total_seconds().fillna(0.0)
    df.drop(columns=["prev_ts_svc"], inplace=True)

    # Encodings
    df["status_encoded"] = df.get("status_code", pd.Series(["unknown"] * len(df))).apply(map_status)

    if freq_table is None:
        fq = (df.groupby(["service", "query"]).size()
              .rename("count").reset_index())
        fq["logfreq"] = np.log1p(fq["count"])
        freq_table = fq[["service", "query", "logfreq"]]
    df = df.merge(freq_table, how="left", on=["service", "query"])
    df["query_encoded"] = df["logfreq"].fillna(0.0)
    df.drop(columns=["logfreq"], inplace=True)

    # Rare query
    df["rank_pct"] = df.groupby("service")["query_encoded"].rank(pct=True, method="first")
    df["rare_query"] = (df["rank_pct"] <= 0.05).astype(int)
    df.drop(columns=["rank_pct"], inplace=True)

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
            print(f"   rows: {len(df)}")
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


# --- Plotting Functions ---
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
        # Save an empty placeholder plot
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
        # placeholder
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
    # if some columns missing, keep available ones
    df_sample = df_sample[[c for c in sample_cols if c in df_sample.columns]]
    df_sample.to_csv(outdir / "sample_anomalies.csv", index=False)


def log_mlflow_metrics(metrics: dict, outdir: Path):
    mlflow.set_experiment("aiops-anomaly-intelligence")
    with mlflow.start_run():
        # Log numeric metrics only (mlflow forbids None)
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass
        # attach artifacts
        for file in ["feature_corr.png", "anomaly_bursts.png",
                     "duplicate_ids.png", "rare_queries.png",
                     "gap_anomalies.png", "combo_anomalies.png",
                     "run_summary.json", "sample_anomalies.csv"]:
            f = outdir / file
            if f.exists():
                try:
                    mlflow.log_artifact(str(f))
                except Exception:
                    print("⚠️ failed mlflow.log_artifact for", f)


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

    df, freq_table = build_features(df)
    y = None
    if args.label_col in df.columns:
        y = (df[args.label_col].astype(str).str.lower() == "anomaly").astype(int).values
        print("INFO: Labels found in data; using them for metric calculation.")

    feats = df[ALL_FEATS].values
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    if y is None:
        # No labels: still create train/test splits so code paths work
        Xtr, Xte = train_test_split(feats_scaled, test_size=0.2, random_state=RANDOM_STATE)
        ytr = yte = None
    else:
        Xtr, Xte, ytr, yte = train_test_split(feats_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    model, raw_tr = train_isoforest(Xtr)

    thr, train_metrics = choose_threshold(raw_tr, ytr)

    df["anomaly_score"] = -model.score_samples(feats_scaled)
    # Use chosen threshold (if None, fallback to 98th percentile)
    if thr is None:
        thr = float(np.quantile(df["anomaly_score"].values, 0.98))
    df["anomaly_flag"] = (df["anomaly_score"] >= thr).astype(int)

    # compute metrics: prefer true labels if available, otherwise create pseudo labels
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
        # Evaluate on test set if available
        try:
            # score Xte
            raw_te = -model.score_samples(Xte)
            y_pred_te = (raw_te >= thr).astype(int)
            if 'yte' in locals() and yte is not None:
                precision = precision_score(yte, y_pred_te, zero_division=0)
                recall = recall_score(yte, y_pred_te, zero_division=0)
                f1 = f1_score(yte, y_pred_te, zero_division=0)
                metrics["metrics_source"] = "true_labels_test"
        except Exception:
            pass
        # fallback to compute on entire dataset vs provided labels
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
        # No true labels — create pseudo labels by selecting top-N anomalies
        contamination = 0.01  # same as model assumption
        n_pseudo = max(1, int(len(df) * contamination))
        top_idx = np.argsort(-df["anomaly_score"].values)[:n_pseudo]
        y_pseudo = np.zeros(len(df), dtype=int)
        y_pseudo[top_idx] = 1
        y_pred = df["anomaly_flag"].values
        precision = precision_score(y_pseudo, y_pred, zero_division=0)
        recall = recall_score(y_pseudo, y_pred, zero_division=0)
        f1 = f1_score(y_pseudo, y_pred, zero_division=0)
        metrics["metrics_source"] = "pseudo_top_percent"

    # attach metrics if present
    metrics["precision"] = float(precision) if precision is not None else None
    metrics["recall"] = float(recall) if recall is not None else None
    metrics["f1"] = float(f1) if f1 is not None else None

    # --- Save analytics + plots ---
    save_feature_correlation(df, outdir)
    save_anomaly_bursts(df, outdir)
    plot_duplicate_ids(df, outdir)
    plot_rare_queries(df, outdir)
    plot_gap_anomalies(df, outdir)
    plot_combo_anomalies(df, outdir)
    log_sample_anomalies(df, outdir)

    # write run_summary.json
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    # Log to MLflow and attach artifacts
    log_mlflow_metrics(metrics, outdir)

    # Optionally save model and scored data (kept minimal)
    try:
        joblib.dump(model, outdir / "model.joblib")
    except Exception:
        pass
    try:
        df.to_csv(outdir / "scored.csv", index=False)
    except Exception:
        pass

    print("✅ Analytics complete. Metrics and artifacts logged to", outdir.resolve())


if __name__ == "__main__":
    main()