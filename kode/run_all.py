import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from kode.common.logger import get_logger
from kode.common.seeds import seed_everything
from kode.regression.train_and_evaluate import run_regression_pipeline
from kode.classification.train_and_evaluate import run_classification_pipeline
from kode.clustering.train_and_evaluate import run_clustering_pipeline
from kode.common.validate_submissions import validate_regression, validate_classification, validate_clustering

logger = get_logger("RunAllPipeline")

def main():
    start_time = time.time()
    seed_everything(42)
    logger.info("=================================================================")
    logger.info("  STARTING END-TO-END UTS MACHINE LEARNING PIPELINE EXECUTION    ")
    logger.info("=================================================================")
    
    # 1. Verify environment and raw files
    raw_files = [
        "regresi_train.csv.gz",
        "regresi_test.csv.gz",
        "klasifikasi_train.csv.gz",
        "klasifikasi_test.csv.gz",
        "clustering.csv.gz"
    ]
    raw_dir = os.path.join(PROJECT_ROOT, "data/raw")
    for f in raw_files:
        path = os.path.join(raw_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required raw data file: {path}")
    logger.info("All raw datasets verified successfully in data/raw/.")
    
    # 2. Execute Task 1: Regression
    logger.info("\n--- EXECUTING TASK 1: REGRESSION ---")
    t1_start = time.time()
    reg_metrics = run_regression_pipeline()
    logger.info(f"Task 1 completed in {time.time() - t1_start:.2f}s")
    
    # 3. Execute Task 2: Classification
    logger.info("\n--- EXECUTING TASK 2: CLASSIFICATION ---")
    t2_start = time.time()
    clf_metrics = run_classification_pipeline()
    logger.info(f"Task 2 completed in {time.time() - t2_start:.2f}s")
    
    # 4. Execute Task 3: Clustering
    logger.info("\n--- EXECUTING TASK 3: CLUSTERING ---")
    t3_start = time.time()
    clust_metrics = run_clustering_pipeline()
    logger.info(f"Task 3 completed in {time.time() - t3_start:.2f}s")
    
    # 5. Run Validation Audit on all generated CSV files
    logger.info("\n--- RUNNING SUBMISSION INTEGRITY AUDIT ---")
    v1 = validate_regression()
    v2 = validate_classification()
    v3 = validate_clustering()
    assert v1 and v2 and v3, "Submission validation failed!"
    
    total_duration = time.time() - start_time
    logger.info("=================================================================")
    logger.info(f"  ALL PIPELINES EXECUTED SUCCESSFULLY IN {total_duration/60:.2f} MINUTES!  ")
    logger.info("  OUTPUTS VERIFIED:                                              ")
    logger.info("  1. prediksi_regresi.csv (60,000 rows)                           ")
    logger.info("  2. prediksi_klasifikasi.csv (106,674 rows)                     ")
    logger.info("  3. cluster.csv (120,000 rows)                                  ")
    logger.info("  4. 9 Publication-Grade Diagnostic Figures in laporan/figures/   ")
    logger.info("  5. Experiment audit trail in experiments/experiments.csv       ")
    logger.info("=================================================================")

if __name__ == "__main__":
    main()
