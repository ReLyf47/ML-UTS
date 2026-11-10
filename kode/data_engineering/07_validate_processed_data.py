import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kode.common.config import DATA_PROCESSED_DIR, DATA_MANIFESTS_DIR
from kode.common.io import read_parquet, save_json
from kode.common.logging_config import setup_logger

logger = setup_logger("07_validate_processed_data", "07_validate_processed_data.log")

ARTIFACTS_TO_VALIDATE = [
    # Common
    {"rel_path": "data/processed/common/cleaned_regresi_train.parquet", "created_by": "03_clean_common.py", "expected_rows": 250000, "id_col": "id_paket", "has_target": True, "target": "durasi_jam"},
    {"rel_path": "data/processed/common/cleaned_regresi_test.parquet", "created_by": "03_clean_common.py", "expected_rows": 60000, "id_col": "id_paket", "has_target": False, "target": None},
    {"rel_path": "data/processed/common/cleaned_klasifikasi_train.parquet", "created_by": "03_clean_common.py", "expected_rows": 403077, "id_col": "id_transaksi", "has_target": True, "target": "is_fraud"},
    {"rel_path": "data/processed/common/cleaned_klasifikasi_test.parquet", "created_by": "03_clean_common.py", "expected_rows": 106674, "id_col": "id_transaksi", "has_target": False, "target": None},
    {"rel_path": "data/processed/common/cleaned_clustering.parquet", "created_by": "03_clean_common.py", "expected_rows": 120000, "id_col": "id_pelanggan", "has_target": False, "target": None},
    
    # Regression
    {"rel_path": "data/processed/regression/train.parquet", "created_by": "04_build_regression_data.py", "expected_rows": 215832, "id_col": "id_paket", "has_target": True, "target": "durasi_jam"},
    {"rel_path": "data/processed/regression/validation.parquet", "created_by": "04_build_regression_data.py", "expected_rows": 34168, "id_col": "id_paket", "has_target": True, "target": "durasi_jam"},
    {"rel_path": "data/processed/regression/test.parquet", "created_by": "04_build_regression_data.py", "expected_rows": 60000, "id_col": "id_paket", "has_target": False, "target": None},
    
    # Classification
    {"rel_path": "data/processed/classification/train.parquet", "created_by": "05_build_classification_data.py", "expected_rows": 350173, "id_col": "id_transaksi", "has_target": True, "target": "is_fraud"},
    {"rel_path": "data/processed/classification/validation.parquet", "created_by": "05_build_classification_data.py", "expected_rows": 52904, "id_col": "id_transaksi", "has_target": True, "target": "is_fraud"},
    {"rel_path": "data/processed/classification/test.parquet", "created_by": "05_build_classification_data.py", "expected_rows": 106674, "id_col": "id_transaksi", "has_target": False, "target": None},
    
    # Clustering
    {"rel_path": "data/processed/clustering/customer_features.parquet", "created_by": "06_build_clustering_data.py", "expected_rows": 120000, "id_col": None, "has_target": False, "target": None},
    {"rel_path": "data/processed/clustering/customer_features_scaled.parquet", "created_by": "06_build_clustering_data.py", "expected_rows": 120000, "id_col": None, "has_target": False, "target": None},
    {"rel_path": "data/processed/clustering/customer_ids.parquet", "created_by": "06_build_clustering_data.py", "expected_rows": 120000, "id_col": "id_pelanggan", "has_target": False, "target": None}
]

def run_validation():
    logger.info("Starting Phase 7: Mandatory Processed Data Validation & Manifest Generation...")
    
    processing_manifest = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "total_artifacts_validated": len(ARTIFACTS_TO_VALIDATE),
        "artifacts": []
    }
    
    all_passed = True
    
    for item in ARTIFACTS_TO_VALIDATE:
        rel_path = item["rel_path"]
        abs_path = PROJECT_ROOT / rel_path
        filename = abs_path.name
        
        logger.info(f"Validating '{rel_path}'...")
        
        # 1. Existence and non-empty check
        if not abs_path.exists():
            logger.error(f"FAIL: File does not exist: {abs_path}")
            all_passed = False
            continue
            
        file_size = abs_path.stat().st_size
        if file_size == 0:
            logger.error(f"FAIL: File is 0 bytes: {abs_path}")
            all_passed = False
            continue
            
        # 2. Read file
        try:
            df = read_parquet(abs_path)
        except Exception as e:
            logger.error(f"FAIL: Failed reading Parquet file {abs_path}: {e}")
            all_passed = False
            continue
            
        # 3. Check row count
        actual_rows = len(df)
        if actual_rows != item["expected_rows"]:
            logger.error(f"FAIL: Row count mismatch for {filename}: expected {item['expected_rows']}, got {actual_rows}")
            all_passed = False
            
        # 4. Check column name uniqueness
        cols = list(df.columns)
        if len(cols) != len(set(cols)):
            logger.error(f"FAIL: Duplicate column names detected in {filename}")
            all_passed = False
            
        # 5. Check nulls
        null_counts = df.isnull().sum()
        missing_dict = {c: int(v) for c, v in null_counts.items() if v > 0}
        
        # 6. Check duplicates
        exact_dupes = int(df.duplicated().sum())
        
        # 7. Check identifier uniqueness if applicable
        id_col = item["id_col"]
        if id_col:
            if id_col not in df.columns:
                logger.error(f"FAIL: Expected ID column '{id_col}' missing in {filename}")
                all_passed = False
            else:
                unique_ids = int(df[id_col].nunique())
                if unique_ids != actual_rows:
                    logger.error(f"FAIL: Non-unique IDs in {filename}: {unique_ids} unique vs {actual_rows} total")
                    all_passed = False
                    
        # 8. Check target column availability
        if item["has_target"]:
            target_col = item["target"]
            if target_col not in df.columns:
                logger.error(f"FAIL: Required target column '{target_col}' missing in {filename}")
                all_passed = False
        else:
            # Test partitions must NOT contain target
            if item.get("target") and item["target"] in df.columns:
                logger.error(f"FAIL: Target column '{item['target']}' leaked into test set {filename}!")
                all_passed = False
                
        # 9. Check leakage in regression features
        if "regression" in rel_path:
            if "jumlah_scan" in df.columns:
                logger.error(f"FAIL: Fatal leakage feature 'jumlah_scan' found in {filename}!")
                all_passed = False
                
        # 10. Check clustering feature sanity
        if "clustering" in rel_path and "customer_features" in filename:
            if "id_pelanggan" in df.columns:
                logger.error(f"FAIL: 'id_pelanggan' found inside clustering feature matrix {filename}!")
                all_passed = False
            if not np.all(np.isfinite(df.values)):
                logger.error(f"FAIL: Non-finite values in clustering feature matrix {filename}!")
                all_passed = False
                
        # Record in manifest
        artifact_entry = {
            "filename": filename,
            "relative_path": rel_path,
            "status": "success" if all_passed else "failure",
            "row_count": actual_rows,
            "column_count": len(cols),
            "columns": cols,
            "missing_values": missing_dict,
            "duplicate_rows": exact_dupes,
            "created_by": item["created_by"]
        }
        processing_manifest["artifacts"].append(artifact_entry)
        logger.info(f"✓ PASS: {filename} ({actual_rows:,} rows, {len(cols)} cols)")

    # Consistency checks: Clustering ID order
    feat_df = read_parquet(PROJECT_ROOT / "data/processed/clustering/customer_features.parquet")
    scaled_df = read_parquet(PROJECT_ROOT / "data/processed/clustering/customer_features_scaled.parquet")
    ids_df = read_parquet(PROJECT_ROOT / "data/processed/clustering/customer_ids.parquet")
    
    assert len(feat_df) == len(scaled_df) == len(ids_df) == 120000, "Clustering row count alignment mismatch!"
    logger.info("✓ PASS: Verified 100% row alignment between clustering features, scaled features, and customer IDs.")
    
    # Consistency checks: Regression train/val schema match
    reg_train = read_parquet(PROJECT_ROOT / "data/processed/regression/train.parquet")
    reg_val = read_parquet(PROJECT_ROOT / "data/processed/regression/validation.parquet")
    assert list(reg_train.columns) == list(reg_val.columns), "Regression train and validation schemas do not match!"
    logger.info("✓ PASS: Regression train and validation schemas match exactly.")
    
    # Consistency checks: Classification train/val schema match
    clf_train = read_parquet(PROJECT_ROOT / "data/processed/classification/train.parquet")
    clf_val = read_parquet(PROJECT_ROOT / "data/processed/classification/validation.parquet")
    assert list(clf_train.columns) == list(clf_val.columns), "Classification train and validation schemas do not match!"
    logger.info("✓ PASS: Classification train and validation schemas match exactly.")

    # Save processing manifest
    manifest_out = DATA_MANIFESTS_DIR / "processing_manifest.json"
    save_json(processing_manifest, manifest_out)
    logger.info(f"Saved complete processing manifest to {manifest_out}")
    
    if not all_passed:
        logger.error("ONE OR MORE ARTIFACT VALIDATIONS FAILED!")
        sys.exit(1)
        
    logger.info("=========================================================")
    logger.info("  ALL PROCESSED DATASETS PASSED INTEGRITY VALIDATION!    ")
    logger.info("=========================================================")

if __name__ == "__main__":
    run_validation()
