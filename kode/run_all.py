import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kode.common.logger import get_logger
from kode.common.seeds import seed_everything
import importlib

inspect_all = importlib.import_module("kode.data_engineering.01_inspect_data").inspect_all
extract_and_normalize = importlib.import_module("kode.data_engineering.02_extract_data").extract_and_normalize
clean_common_pipeline = importlib.import_module("kode.data_engineering.03_clean_common").clean_common_pipeline
build_regression_dataset = importlib.import_module("kode.data_engineering.04_build_regression_data").build_regression_dataset
build_classification_dataset = importlib.import_module("kode.data_engineering.05_build_classification_data").build_classification_dataset
build_clustering_dataset = importlib.import_module("kode.data_engineering.06_build_clustering_data").build_clustering_dataset
run_validation = importlib.import_module("kode.data_engineering.07_validate_processed_data").run_validation
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

    # Step 1: Inspect raw data
    logger.info("\n--- STEP 1: INSPECT RAW DATA ---")
    inspect_all()

    # Step 2: Extract & interim normalization
    logger.info("\n--- STEP 2: EXTRACT AND INTERIM NORMALIZATION ---")
    extract_and_normalize()

    # Step 3: Common cleaned data
    logger.info("\n--- STEP 3: GENERATE COMMON CLEANED DATA ---")
    clean_common_pipeline()

    # Step 4: Build regression processed data
    logger.info("\n--- STEP 4: BUILD REGRESSION PROCESSED DATA ---")
    build_regression_dataset()

    # Step 5: Build classification processed data
    logger.info("\n--- STEP 5: BUILD CLASSIFICATION PROCESSED DATA ---")
    build_classification_dataset()

    # Step 6: Build clustering processed data
    logger.info("\n--- STEP 6: BUILD CLUSTERING PROCESSED DATA ---")
    build_clustering_dataset()

    # Step 7: Validate all processed data artifacts
    logger.info("\n--- STEP 7: VALIDATE PROCESSED DATA ARTIFACTS ---")
    run_validation()

    # Step 8: Train Task 1 (Regression) & predict
    logger.info("\n--- STEP 8: EXECUTE TASK 1 (REGRESSION) ---")
    t1_start = time.time()
    run_regression_pipeline()
    logger.info(f"Task 1 completed in {time.time() - t1_start:.2f}s")

    # Step 9: Train Task 2 (Classification) & predict
    logger.info("\n--- STEP 9: EXECUTE TASK 2 (CLASSIFICATION) ---")
    t2_start = time.time()
    run_classification_pipeline()
    logger.info(f"Task 2 completed in {time.time() - t2_start:.2f}s")

    # Step 10: Train Task 3 (Clustering) & predict
    logger.info("\n--- STEP 10: EXECUTE TASK 3 (CLUSTERING) ---")
    t3_start = time.time()
    run_clustering_pipeline()
    logger.info(f"Task 3 completed in {time.time() - t3_start:.2f}s")

    # Step 11: Validate final submission CSVs
    logger.info("\n--- STEP 11: VALIDATE FINAL SUBMISSION FILES ---")
    v1 = validate_regression()
    v2 = validate_classification()
    v3 = validate_clustering()
    assert v1 and v2 and v3, "Final submission validation failed!"

    total_duration = time.time() - start_time
    logger.info("=================================================================")
    logger.info(f"  ALL PIPELINES EXECUTED SUCCESSFULLY IN {total_duration/60:.2f} MINUTES!  ")
    logger.info("  OUTPUTS VERIFIED:                                              ")
    logger.info("  1. outputs/predictions/prediksi_regresi.csv (60,000 rows)      ")
    logger.info("  2. outputs/predictions/prediksi_klasifikasi.csv (106,674 rows) ")
    logger.info("  3. outputs/predictions/cluster.csv (120,000 rows)              ")
    logger.info("  4. Diagnostic figures in reports/figures/                      ")
    logger.info("  5. Manifests in data/manifests/                                ")
    logger.info("=================================================================")

if __name__ == "__main__":
    main()
