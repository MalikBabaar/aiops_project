from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import pandas as pd
import io

from .train import retrain_model  # import the wrapper

app = FastAPI(title="Trainer API")

class TrainRequest(BaseModel):
    input_paths: List[str] = Field(default_factory=lambda: ["/data/ingest/ingest_buffer.csv"])
    outdir: str = "/data/models/run"
    label_col: Optional[str] = "anomaly_tag"
    mlflow_experiment: str = "aiops-anomaly-intelligence"

@app.post("/train")
def run_train(req: TrainRequest):
    frames = []
    for p in req.input_paths:
        path = Path(p)
        if not path.exists():
            raise HTTPException(400, f"Input path not found: {path}")
        try:
            try:
                df = pd.read_csv(path, engine="python", on_bad_lines="skip")
            except Exception:
                df = pd.read_csv(path, names=["log"], engine="python", on_bad_lines="skip")
            frames.append(df)
        except Exception as e:
            raise HTTPException(400, f"Failed to read {path}: {e}")

    if not frames:
        raise HTTPException(400, "No valid input files were loaded.")

    df = pd.concat(frames, ignore_index=True)

    exit_code, metrics, artifacts = retrain_model(
        df=df,
        outdir=req.outdir,
        label_col=req.label_col or "anomaly_tag",
        mlflow_experiment=req.mlflow_experiment
    )

    resp = {
        "ok": exit_code in (0, 3),
        "exit_code": exit_code,
        "metrics": metrics,
        "artifacts": artifacts,
    }
    if exit_code in (0, 3):
        return resp
    else:
        raise HTTPException(500, resp)

# Optional: upload-based retraining (handy later)
@app.post("/retrain")
async def retrain_upload(
    file: UploadFile = File(...),
    outdir: str = Form("/data/models/run_upload"),
    label_col: Optional[str] = Form("anomaly_tag"),
    mlflow_experiment: str = Form("aiops-anomaly-intelligence"),
):
    try:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content), engine="python", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(io.BytesIO(content), names=["log"], engine="python", on_bad_lines="skip")
    except Exception as e:
        raise HTTPException(400, f"Bad upload: {e}")

    exit_code, metrics, artifacts = retrain_model(
        df=df,
        outdir=outdir,
        label_col=label_col or "anomaly_tag",
        mlflow_experiment=mlflow_experiment
    )
    resp = {"ok": exit_code in (0, 3), "exit_code": exit_code, "metrics": metrics, "artifacts": artifacts}
    if exit_code in (0, 3):
        return resp
    else:
        raise HTTPException(500, resp)

@app.get("/healthz")
def healthz():
    return {"ok": True}
