import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kode.common.config import DATA_INTERIM_NORMALIZED, DATA_PROCESSED_COMMON
from kode.common.io import read_parquet, save_parquet, save_json
from kode.common.logging_config import setup_logger

logger = setup_logger("03_clean_common", "03_clean_common.log")

def clean_common_pipeline():
    logger.info("Starting Phase 3: Shared Cleaned Data Generation...")
    
    summary = {
        "created_at": pd.Timestamp.now().isoformat(),
        "datasets": {}
    }

    # ==========================================
    # 1. Clean Regression Datasets
    # ==========================================
    logger.info("Cleaning regression datasets...")
    df_reg_train = read_parquet(DATA_INTERIM_NORMALIZED / "regresi_train.parquet")
    df_reg_test = read_parquet(DATA_INTERIM_NORMALIZED / "regresi_test.parquet")
    
    # Calculate median weight grouped by layanan strictly from train
    weight_medians = df_reg_train.groupby("layanan")["berat_kg"].median().to_dict()
    global_weight_median = float(df_reg_train["berat_kg"].median())
    
    for df, name in [(df_reg_train, "regresi_train"), (df_reg_test, "regresi_test")]:
        df["tgl_pickup"] = pd.to_datetime(df["tgl_pickup"])
        df["id_paket"] = df["id_paket"].astype(str).str.strip()
        
        # Missing indicator & deterministic imputation for berat_kg
        df["berat_kg_missing"] = df["berat_kg"].isnull().astype(int)
        imputed_weight = df["berat_kg"].copy()
        for svc, m_val in weight_medians.items():
            mask = (df["layanan"] == svc) & (imputed_weight.isnull())
            imputed_weight.loc[mask] = m_val
        df["berat_kg"] = imputed_weight.fillna(global_weight_median).astype(float)
        
        # Validate non-negative numbers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != "durasi_jam" and (df[col] < 0).any():
                logger.warning(f"Found negative values in {name}.{col}, checking...")
                
        out_path = DATA_PROCESSED_COMMON / f"cleaned_{name}.parquet"
        save_parquet(df, out_path)
        logger.info(f"✓ Saved {out_path.name}: {len(df):,} rows, {len(df.columns)} cols")
        
        summary["datasets"][name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_cells": int(df.isnull().sum().sum()),
            "path": str(out_path)
        }

    # ==========================================
    # 2. Clean Classification Datasets
    # ==========================================
    logger.info("Cleaning classification datasets...")
    df_clf_train = read_parquet(DATA_INTERIM_NORMALIZED / "klasifikasi_train.parquet")
    df_clf_test = read_parquet(DATA_INTERIM_NORMALIZED / "klasifikasi_test.parquet")

    for df, name in [(df_clf_train, "klasifikasi_train"), (df_clf_test, "klasifikasi_test")]:
        df["waktu"] = pd.to_datetime(df["waktu"])
        df["id_transaksi"] = df["id_transaksi"].astype(str).str.strip()
        df["id_pengguna"] = df["id_pengguna"].astype(str).str.strip()
        df["id_merchant"] = df["id_merchant"].astype(str).str.strip()
        
        # Structural missingness for first-time users
        df["is_first_trx"] = df["menit_sejak_trx_terakhir"].isnull().astype(int)
        df["menit_sejak_trx_terakhir"] = df["menit_sejak_trx_terakhir"].fillna(-1.0).astype(float)
        df["rasio_nominal_rata2"] = df["rasio_nominal_rata2"].fillna(1.0).astype(float)
        
        out_path = DATA_PROCESSED_COMMON / f"cleaned_{name}.parquet"
        save_parquet(df, out_path)
        logger.info(f"✓ Saved {out_path.name}: {len(df):,} rows, {len(df.columns)} cols")
        
        summary["datasets"][name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_cells": int(df.isnull().sum().sum()),
            "path": str(out_path)
        }

    # ==========================================
    # 3. Clean Clustering Dataset
    # ==========================================
    logger.info("Cleaning clustering dataset...")
    df_clust = read_parquet(DATA_INTERIM_NORMALIZED / "clustering.parquet")
    df_clust["id_pelanggan"] = df_clust["id_pelanggan"].astype(str).str.strip()
    
    # Deterministic reconstruction of missing rata2_keranjang
    mask_missing = df_clust["rata2_keranjang"].isnull()
    reconstructed = np.where(
        df_clust["frekuensi_90h"] > 0,
        df_clust["belanja_90h"] / np.maximum(1.0, df_clust["frekuensi_90h"]),
        0.0
    )
    df_clust["rata2_keranjang"] = np.where(mask_missing, reconstructed, df_clust["rata2_keranjang"]).astype(float)
    
    out_path = DATA_PROCESSED_COMMON / "cleaned_clustering.parquet"
    save_parquet(df_clust, out_path)
    logger.info(f"✓ Saved {out_path.name}: {len(df_clust):,} rows, {len(df_clust.columns)} cols")
    
    summary["datasets"]["clustering"] = {
        "rows": len(df_clust),
        "columns": len(df_clust.columns),
        "missing_cells": int(df_clust.isnull().sum().sum()),
        "path": str(out_path)
    }

    # Save summary
    summary_path = DATA_PROCESSED_COMMON / "common_data_summary.json"
    save_json(summary, summary_path)
    logger.info(f"Saved common data summary to {summary_path}")
    logger.info("Phase 3: Shared Cleaned Data Generation completed.")

if __name__ == "__main__":
    clean_common_pipeline()
