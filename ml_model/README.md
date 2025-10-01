---

## 🧪 Sample Data

**Path:** `data/jboss_logs_sample.csv`

This file contains a small set of dummy JBoss logs to test the pipeline without needing real-time logs.

---

## 🧹 Preprocessing Script

**Path:** `ml_model/preprocess.py`

This script transforms raw log lines into structured features suitable for model training and inference.

---

## 📊 Feature Schema

The model expects the following features:

- `hour` (int): Hour of the log timestamp
- `dayofweek` (int): Day of the week (0=Monday)
- `auth_failure` (int): 1 if "authentication failure" is present
- `failed_password` (int): 1 if "failed password" is present
- `pid` (int): Process ID extracted from the log

---

## 🧠 Trained Model

**Format:** `.joblib`  
**Location:** `models/isolation_forest_<version>.joblib`

Each retrained model is saved with a timestamped version for traceability.

---

## 🏋️ Training Script

**File:** `train.py`

**Usage:**

```bash
python train.py --input data/jboss_logs_sample.csv --output models/isolation_forest_v1.joblib
