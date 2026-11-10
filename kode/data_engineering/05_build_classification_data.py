import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kode.common.config import DATA_PROCESSED_COMMON, DATA_PROCESSED_CLASSIFICATION
from kode.common.io import read_parquet, save_parquet, save_json
from kode.common.logging_config import setup_logger

logger = setup_logger("05_build_classification_data", "05_build_classification_data.log")

CATEGORICAL_COLS = [
    'kategori_merchant',
    'kanal',
    'tipe_perangkat',
    'email_domain'
]

NUMERICAL_COLS = [
    'jam',
    'hari_libur',
    'umur_akun_hari',
    'level_kyc',
    'umur_pengguna_tahun',
    'umur_perangkat_hari',
    'perangkat_baru',
    'jumlah_akun_per_perangkat',
    'ganti_password_7h',
    'gagal_login_24j',
    'jarak_lokasi_km',
    'negara_ip_beda',
    'kartu_tidak_hadir',
    'trx_1j',
    'trx_24j',
    'trx_7h',
    'nominal_24j',
    'saldo_sebelum',
    'persen_saldo',
    'umur_merchant_hari',
    'refund_merchant_30h',
    'volume_merchant_30h',
    'menit_sejak_trx_terakhir',
    'rasio_nominal_rata2',
    'is_first_trx'
]

V_COLS = [f'v{i:02d}' for i in range(1, 51)]

def build_classification_dataset():
    logger.info("Starting Phase 5: Classification Feature Engineering & Splitting...")
    
    # 1. Load cleaned common classification datasets
    df_raw_train = read_parquet(DATA_PROCESSED_COMMON / "cleaned_klasifikasi_train.parquet")
    df_raw_test = read_parquet(DATA_PROCESSED_COMMON / "cleaned_klasifikasi_test.parquet")
    
    logger.info(f"Loaded cleaned_klasifikasi_train: {len(df_raw_train):,} rows")
    logger.info(f"Loaded cleaned_klasifikasi_test: {len(df_raw_test):,} rows")
    
    # 2. Chronological Split (August 2025 as validation)
    VAL_CUTOFF = pd.Timestamp("2025-08-01")
    train_mask = df_raw_train["waktu"] < VAL_CUTOFF
    val_mask = df_raw_train["waktu"] >= VAL_CUTOFF
    
    df_train_raw = df_raw_train[train_mask].copy().reset_index(drop=True)
    df_val_raw = df_raw_train[val_mask].copy().reset_index(drop=True)
    df_test_raw = df_raw_test.copy().reset_index(drop=True)
    
    logger.info(f"Chronological split sizes: Train={len(df_train_raw):,} (Jan-Jul 2025), Val={len(df_val_raw):,} (Aug 2025), Test={len(df_test_raw):,} (Sep-Oct 2025)")
    
    # 3. Fit Categorical Encoders strictly on Train split
    logger.info("Fitting categorical encoders strictly on Train split...")
    category_mappings = {}
    for col in CATEGORICAL_COLS:
        uniques = sorted(df_train_raw[col].dropna().unique().tolist())
        category_mappings[col] = {val: idx + 1 for idx, val in enumerate(uniques)}
        
    # 4. Transform Partitions
    def transform_df(df_in: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
        df_out = pd.DataFrame()
        # Keep essential IDs and business values
        df_out["id_transaksi"] = df_in["id_transaksi"].values
        df_out["nominal"] = df_in["nominal"].astype(float).values
        
        # Temporal & cyclical features
        dt = pd.to_datetime(df_in["waktu"])
        hours = dt.dt.hour.astype(float).values
        df_out["day_of_week"] = dt.dt.day_of_week.astype(int)
        df_out["is_weekend"] = (dt.dt.day_of_week >= 5).astype(int)
        df_out["sin_jam"] = np.sin(2 * np.pi * hours / 24.0)
        df_out["cos_jam"] = np.cos(2 * np.pi * hours / 24.0)
        
        # Financial ratio & log features
        nom = df_out["nominal"].values
        df_out["log_nominal"] = np.log1p(nom)
        saldo = np.maximum(0.0, df_in["saldo_sebelum"].astype(float).values)
        df_out["log_saldo_sebelum"] = np.log1p(saldo)
        df_out["nominal_to_saldo_ratio"] = nom / (saldo + 1000.0)
        
        # Velocity and burst features
        trx_1j = df_in["trx_1j"].astype(float).values
        trx_24j = df_in["trx_24j"].astype(float).values
        df_out["velocity_ratio_1j_24j"] = trx_1j / (trx_24j + 1.0)
        
        # Base numericals
        for c in NUMERICAL_COLS:
            df_out[c] = df_in[c].astype(float)
            
        # Categoricals (1-indexed; 0 reserved for unseen/missing)
        for col in CATEGORICAL_COLS:
            mapping = category_mappings[col]
            df_out[f"{col}_cat"] = df_in[col].map(mapping).fillna(0).astype(int)
            
        # 50 internal risk engine features
        for v in V_COLS:
            df_out[v] = df_in[v].astype(float)
            
        # Target if not test
        if not is_test and "is_fraud" in df_in.columns:
            df_out["is_fraud"] = df_in["is_fraud"].astype(int)
            
        return df_out

    logger.info("Transforming train, validation, and test partitions...")
    df_train_proc = transform_df(df_train_raw, is_test=False)
    df_val_proc = transform_df(df_val_raw, is_test=False)
    df_test_proc = transform_df(df_test_raw, is_test=True)
    
    # Feature columns (excluding id_transaksi, nominal, is_fraud)
    feature_cols = [c for c in df_train_proc.columns if c not in ("id_transaksi", "is_fraud")]
    categorical_cols = [c for c in feature_cols if c.endswith("_cat")]
    
    # Save Processed Partitions
    train_path = DATA_PROCESSED_CLASSIFICATION / "train.parquet"
    val_path = DATA_PROCESSED_CLASSIFICATION / "validation.parquet"
    test_path = DATA_PROCESSED_CLASSIFICATION / "test.parquet"
    
    save_parquet(df_train_proc, train_path)
    save_parquet(df_val_proc, val_path)
    save_parquet(df_test_proc, test_path)
    
    logger.info(f"✓ Saved classification train: {train_path.name} ({len(df_train_proc):,} rows, {len(df_train_proc.columns)} cols)")
    logger.info(f"✓ Saved classification val: {val_path.name} ({len(df_val_proc):,} rows, {len(df_val_proc.columns)} cols)")
    logger.info(f"✓ Saved classification test: {test_path.name} ({len(df_test_proc):,} rows, {len(df_test_proc.columns)} cols)")
    
    # Save Metadata and Class Distribution
    feature_meta = {
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "categorical_columns": categorical_cols,
        "id_column": "id_transaksi",
        "nominal_column": "nominal",
        "target_column": "is_fraud"
    }
    save_json(feature_meta, DATA_PROCESSED_CLASSIFICATION / "feature_columns.json")
    
    prep_meta = {
        "category_mappings": category_mappings,
        "split_cutoff": str(VAL_CUTOFF)
    }
    save_json(prep_meta, DATA_PROCESSED_CLASSIFICATION / "preprocessing_metadata.json")
    
    # Detailed class distribution
    train_pos = int(df_train_proc["is_fraud"].sum())
    train_neg = int(len(df_train_proc) - train_pos)
    val_pos = int(df_val_proc["is_fraud"].sum())
    val_neg = int(len(df_val_proc) - val_pos)
    
    class_dist = {
        "train": {
            "total": len(df_train_proc),
            "negative_legit": train_neg,
            "positive_fraud": train_pos,
            "fraud_rate": round(float(train_pos / len(df_train_proc)), 5),
            "imbalance_ratio": round(float(train_neg / max(1, train_pos)), 2)
        },
        "validation": {
            "total": len(df_val_proc),
            "negative_legit": val_neg,
            "positive_fraud": val_pos,
            "fraud_rate": round(float(val_pos / len(df_val_proc)), 5),
            "imbalance_ratio": round(float(val_neg / max(1, val_pos)), 2)
        },
        "test": {
            "total": len(df_test_proc),
            "is_labeled": False
        }
    }
    save_json(class_dist, DATA_PROCESSED_CLASSIFICATION / "class_distribution.json")
    
    summary = {
        "train_rows": len(df_train_proc),
        "validation_rows": len(df_val_proc),
        "test_rows": len(df_test_proc),
        "feature_count": len(feature_cols),
        "train_fraud_count": train_pos,
        "val_fraud_count": val_pos,
        "created_at": pd.Timestamp.now().isoformat()
    }
    save_json(summary, DATA_PROCESSED_CLASSIFICATION / "classification_data_summary.json")
    logger.info("Phase 5: Classification Feature Engineering completed.")

if __name__ == "__main__":
    build_classification_dataset()
