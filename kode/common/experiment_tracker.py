import os
import json
import time
import pandas as pd
from datetime import datetime

from kode.common.config import EXPERIMENTS_DIR

CSV_PATH = os.path.join(EXPERIMENTS_DIR, "experiments.csv")
JSONL_PATH = os.path.join(EXPERIMENTS_DIR, "experiments.jsonl")

def log_experiment(
    experiment_id: str,
    task: str,
    model_name: str,
    features_used: list,
    split_type: str,
    hyperparameters: dict,
    metrics: dict,
    runtime_seconds: float,
    seed: int = 42,
    notes: str = ""
):
    """Persist structured experiment results to CSV and JSONL without overwriting previous runs."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    
    record = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "task": task,
        "model_name": model_name,
        "n_features": len(features_used),
        "split_type": split_type,
        "seed": seed,
        "runtime_seconds": round(runtime_seconds, 2),
        "notes": notes,
        "hyperparameters": json.dumps(hyperparameters),
        **{f"metric_{k}": v for k, v in metrics.items()}
    }
    
    # Append to JSONL
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps({**record, "features_used": features_used}) + "\n")
        
    # Append to CSV
    df_new = pd.DataFrame([record])
    if not os.path.exists(CSV_PATH):
        df_new.to_csv(CSV_PATH, index=False)
    else:
        df_new.to_csv(CSV_PATH, mode='a', header=False, index=False)
        
    return record
