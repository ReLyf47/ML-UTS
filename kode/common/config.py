import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_INTERIM_EXTRACTED = DATA_INTERIM_DIR / "extracted"
DATA_INTERIM_NORMALIZED = DATA_INTERIM_DIR / "normalized"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_PROCESSED_COMMON = DATA_PROCESSED_DIR / "common"
DATA_PROCESSED_REGRESSION = DATA_PROCESSED_DIR / "regression"
DATA_PROCESSED_CLASSIFICATION = DATA_PROCESSED_DIR / "classification"
DATA_PROCESSED_CLUSTERING = DATA_PROCESSED_DIR / "clustering"
DATA_MANIFESTS_DIR = DATA_DIR / "manifests"

# Code and outputs directories
REPORTS_DIR = BASE_DIR / "reports"
OUTPUTS_DIR = BASE_DIR / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = BASE_DIR / "logs"

# Ensure essential directories exist
for p in [
    DATA_RAW_DIR, DATA_INTERIM_EXTRACTED, DATA_INTERIM_NORMALIZED,
    DATA_PROCESSED_COMMON, DATA_PROCESSED_REGRESSION,
    DATA_PROCESSED_CLASSIFICATION, DATA_PROCESSED_CLUSTERING,
    DATA_MANIFESTS_DIR, REPORTS_DIR, PREDICTIONS_DIR,
    MODELS_DIR, METRICS_DIR, LOGS_DIR
]:
    p.mkdir(parents=True, exist_ok=True)

# Pipeline configuration constants
RANDOM_SEED = 42
REVIEW_COST_IDR = 25000.0  # Rp 25.000 per manual fraud review
QUANTILES_REGRESSION = [0.50, 0.90]

# Expected dataset sizes
EXPECTED_ROWS = {
    "regresi_train": 250000,
    "regresi_test": 60000,
    "klasifikasi_train": 403077,
    "klasifikasi_test": 106674,
    "clustering": 120000
}
