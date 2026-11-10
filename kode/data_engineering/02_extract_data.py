import os
import sys
import shutil
import gzip
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kode.common.config import DATA_RAW_DIR, DATA_INTERIM_EXTRACTED, DATA_INTERIM_NORMALIZED, DATA_MANIFESTS_DIR
from kode.common.io import find_raw_file, read_json, save_json
from kode.common.logging_config import setup_logger

logger = setup_logger("02_extract_data", "02_extract_data.log")

DATASET_NAMES = [
    "regresi_train",
    "regresi_test",
    "klasifikasi_train",
    "klasifikasi_test",
    "clustering"
]

def extract_and_normalize():
    logger.info("Starting Phase 1.2: Data Extraction & Interim Normalization...")
    
    manifest_path = DATA_MANIFESTS_DIR / "source_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"files": {}}

    for name in DATASET_NAMES:
        raw_file = find_raw_file(name)
        extracted_csv = DATA_INTERIM_EXTRACTED / f"{name}.csv"
        normalized_parquet = DATA_INTERIM_NORMALIZED / f"{name}.parquet"
        
        logger.info(f"Processing '{name}' from {raw_file.name}...")
        
        # 1. Extraction: If gzip, decompress; if CSV, copy to interim/extracted
        if raw_file.suffix == ".gz":
            logger.info(f"Decompressing gzip archive {raw_file.name} to {extracted_csv.name}...")
            with gzip.open(raw_file, "rb") as f_in, open(extracted_csv, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            if not extracted_csv.exists() or extracted_csv.stat().st_size != raw_file.stat().st_size:
                logger.info(f"Copying {raw_file.name} to interim/extracted/...")
                shutil.copy2(raw_file, extracted_csv)
            else:
                logger.info(f"Extracted CSV already up to date: {extracted_csv.name}")

        # 2. Interim normalization: read CSV, standardize string types & column whitespaces, save as Parquet
        logger.info(f"Normalizing '{name}' to {normalized_parquet.name}...")
        df = pd.read_csv(extracted_csv, low_memory=False)
        
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        
        # Clean text columns
        str_cols = df.select_dtypes(include=["object"]).columns
        for c in str_cols:
            df[c] = df[c].astype(str).str.strip()

        # Save interim Parquet
        df.to_parquet(normalized_parquet, index=False, compression="snappy")
        logger.info(f"✓ Saved normalized Parquet: {normalized_parquet.name} ({len(df):,} rows, {len(df.columns)} cols)")

        # Update manifest
        if name in manifest.get("files", {}):
            manifest["files"][name]["extracted_path"] = str(extracted_csv)
            manifest["files"][name]["normalized_path"] = str(normalized_parquet)
            manifest["files"][name]["processing_status"] = "extracted_and_normalized"

    save_json(manifest, manifest_path)
    logger.info(f"Updated source manifest at {manifest_path}")
    logger.info("Phase 1.2: Data Extraction & Interim Normalization completed.")

if __name__ == "__main__":
    extract_and_normalize()
