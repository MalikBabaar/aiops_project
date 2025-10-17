from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import subprocess, json
from pathlib import Path

app = FastAPI(title="Trainer API")

class TrainRequest(BaseModel):
    input_paths: List[str] = Field(default_factory=lambda: ["/data/ingest/ingest_buffer.csv"])
    outdir: str = "/data/models/run"
    model_out: str = "/data/models/run/isoforest.joblib"
    scored_out: str = "/data/models/run/scored.csv"
    label_col: Optional[str] = "anomaly_tag"
    service: Optional[str] = None

@app.post("/train")
def run_train(req: TrainRequest):
    cmd = [
        "python",
        "train.py",
        "--inputs",
        *req.input_paths,
        "--outdir",
        req.outdir,
        "--model_out",
        req.model_out,
        "--scored_out",
        req.scored_out,
    ]
    if req.label_col:
        cmd += ["--label-col", req.label_col]
    if req.service:
        cmd += ["--service", req.service]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=True)
        summary_path = Path(req.outdir) / "run_summary.json"
        summary = {}
        if summary_path.exists():
            with open(summary_path) as fh:
                summary = json.load(fh)
        return {"ok": True, "stdout": p.stdout, "summary": summary}
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, e.stdout + "\n" + e.stderr)


@app.get("/healthz")
def healthz():
    return {"ok": True}
