import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kode.common.config import DATA_PROCESSED_COMMON, DATA_PROCESSED_REGRESSION
from kode.common.io import read_parquet, save_parquet, save_json
from kode.common.logging_config import setup_logger

logger = setup_logger("04_build_regression_data", "04_build_regression_data.log")

LOW_CARDINALITY_CATEGORICALS = [
    'layanan',
    'mitra_kurir',
    'hub_asal',
    'kota_asal',
    'pulau_tujuan'
]

BASE_NUMERICAL_COLS = [
    'jarak_km',
    'berat_kg',
    'volume_liter',
    'jumlah_item',
    'nilai_barang',
    'asuransi',
    'cod',
    'alamat_skor',
    'pengalaman_kurir_bulan',
    'rating_penjual',
    'umur_akun_penjual_hari',
    'volume_penjual_30h',
    'antrean_hub_asal',
    'indeks_kemacetan',
    'hujan_asal_mm',
    'hujan_tujuan_mm'
]

F_COLS = [f'f{i:02d}' for i in range(1, 51)]

def build_regression_dataset():
    logger.info("Starting Phase 4: Regression Feature Engineering & Splitting...")
    
    # 1. Load cleaned common regression datasets
    df_raw_train = read_parquet(DATA_PROCESSED_COMMON / "cleaned_regresi_train.parquet")
    df_raw_test = read_parquet(DATA_PROCESSED_COMMON / "cleaned_regresi_test.parquet")
    
    logger.info(f"Loaded cleaned_regresi_train: {len(df_raw_train):,} rows")
    logger.info(f"Loaded cleaned_regresi_test: {len(df_raw_test):,} rows")
    
    # 2. Chronological Split for Train vs Validation (August 2025 as out-of-time validation)
    VAL_CUTOFF = pd.Timestamp("2025-08-01")
    train_mask = df_raw_train["tgl_pickup"] < VAL_CUTOFF
    val_mask = df_raw_train["tgl_pickup"] >= VAL_CUTOFF
    
    df_train_raw = df_raw_train[train_mask].copy().reset_index(drop=True)
    df_val_raw = df_raw_train[val_mask].copy().reset_index(drop=True)
    df_test_raw = df_raw_test.copy().reset_index(drop=True)
    
    logger.info(f"Chronological split sizes: Train={len(df_train_raw):,} (Jan-Jul 2025), Val={len(df_val_raw):,} (Aug 2025), Test={len(df_test_raw):,} (Sep-Oct 2025)")
    
    # 3. Fit Transformations strictly on Train partition
    logger.info("Fitting categorical mappings and target encodings on Train split...")
    
    # Categorical encoders (1-indexed; 0 reserved for unseen/missing)
    category_mappings = {}
    for col in LOW_CARDINALITY_CATEGORICALS:
        uniques = sorted(df_train_raw[col].dropna().unique().tolist())
        category_mappings[col] = {val: idx + 1 for idx, val in enumerate(uniques)}
        
    # High cardinality kota_tujuan: smoothed target encoding & frequency encoding
    global_target_median = float(df_train_raw["durasi_jam"].median())
    smoothing_m = 20.0
    city_counts = df_train_raw["kota_tujuan"].value_counts()
    total_train = len(df_train_raw)
    city_freq_map = (city_counts / total_train).to_dict()
    city_medians = df_train_raw.groupby("kota_tujuan")["durasi_jam"].median()
    
    city_target_map = {}
    for city, count in city_counts.items():
        med_c = city_medians.get(city, global_target_median)
        smoothed = (count * med_c + smoothing_m * global_target_median) / (count + smoothing_m)
        city_target_map[city] = float(smoothed)

    # 4. Transform Partitions
    def transform_df(df_in: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
        df_out = pd.DataFrame()
        df_out["id_paket"] = df_in["id_paket"].values
        
        # Temporal features
        tgl = pd.to_datetime(df_in["tgl_pickup"])
        df_out["pickup_dayofweek"] = tgl.dt.dayofweek.astype(int)
        df_out["pickup_is_weekend"] = (tgl.dt.dayofweek >= 5).astype(int)
        df_out["pickup_month"] = tgl.dt.month.astype(int)
        df_out["pickup_day"] = tgl.dt.day.astype(int)
        
        hours = df_in["jam_pickup"].astype(float).values
        df_out["jam_pickup"] = hours
        df_out["sin_jam_pickup"] = np.sin(2 * np.pi * hours / 24.0)
        df_out["cos_jam_pickup"] = np.cos(2 * np.pi * hours / 24.0)
        
        # Domain ratios
        weight = df_in["berat_kg"].astype(float).values
        vol = df_in["volume_liter"].astype(float).values
        items = df_in["jumlah_item"].astype(float).values
        nilai = df_in["nilai_barang"].astype(float).values
        
        df_out["densitas_kg_per_l"] = weight / (vol + 1e-4)
        df_out["nilai_per_kg"] = nilai / (weight + 1e-4)
        df_out["nilai_per_item"] = nilai / (items + 1e-4)
        df_out["is_same_city"] = (df_in["kota_asal"] == df_in["kota_tujuan"]).astype(int)
        
        # Base numericals
        for c in BASE_NUMERICAL_COLS:
            df_out[c] = df_in[c].astype(float)
            
        # Low-cardinality categoricals
        for col in LOW_CARDINALITY_CATEGORICALS:
            mapping = category_mappings[col]
            df_out[f"{col}_cat"] = df_in[col].map(mapping).fillna(0).astype(int)
            
        # Target and frequency encoding for kota_tujuan
        df_out["kota_tujuan_te"] = df_in["kota_tujuan"].map(city_target_map).fillna(global_target_median).astype(float)
        df_out["kota_tujuan_freq"] = df_in["kota_tujuan"].map(city_freq_map).fillna(0.0).astype(float)
        
        # 50 internal system features
        for f in F_COLS:
            df_out[f] = df_in[f].astype(float)
            
        # Add target if not test
        if not is_test and "durasi_jam" in df_in.columns:
            df_out["durasi_jam"] = df_in["durasi_jam"].astype(float)
            
        return df_out

    logger.info("Transforming train, validation, and test partitions...")
    df_train_proc = transform_df(df_train_raw, is_test=False)
    df_val_proc = transform_df(df_val_raw, is_test=False)
    df_test_proc = transform_df(df_test_raw, is_test=True)
    
    # Feature columns (excluding id_paket and target durasi_jam)
    feature_cols = [c for c in df_train_proc.columns if c not in ("id_paket", "durasi_jam")]
    categorical_cols = [c for c in feature_cols if c.endswith("_cat")]
    
    # Verify strict exclusion of leakage column 'jumlah_scan'
    assert "jumlah_scan" not in feature_cols, "CRITICAL ERROR: Leakage column 'jumlah_scan' found in feature set!"
    logger.info("✓ Verified complete exclusion of fatal leakage feature 'jumlah_scan'.")
    
    # 5. Save Processed Artifacts
    train_path = DATA_PROCESSED_REGRESSION / "train.parquet"
    val_path = DATA_PROCESSED_REGRESSION / "validation.parquet"
    test_path = DATA_PROCESSED_REGRESSION / "test.parquet"
    
    save_parquet(df_train_proc, train_path)
    save_parquet(df_val_proc, val_path)
    save_parquet(df_test_proc, test_path)
    
    logger.info(f"✓ Saved regression train: {train_path.name} ({len(df_train_proc):,} rows, {len(df_train_proc.columns)} cols)")
    logger.info(f"✓ Saved regression val: {val_path.name} ({len(df_val_proc):,} rows, {len(df_val_proc.columns)} cols)")
    logger.info(f"✓ Saved regression test: {test_path.name} ({len(df_test_proc):,} rows, {len(df_test_proc.columns)} cols)")
    
    # Save Metadata and Schema
    feature_meta = {
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "categorical_columns": categorical_cols,
        "excluded_features": ["jumlah_scan", "id_paket"],
        "target_column": "durasi_jam",
        "id_column": "id_paket"
    }
    save_json(feature_meta, DATA_PROCESSED_REGRESSION / "feature_columns.json")
    
    prep_meta = {
        "smoothing_m": smoothing_m,
        "global_target_median": global_target_median,
        "category_mappings": category_mappings,
        "city_target_map_size": len(city_target_map),
        "split_cutoff": str(VAL_CUTOFF)
    }
    save_json(prep_meta, DATA_PROCESSED_REGRESSION / "preprocessing_metadata.json")
    
    summary = {
        "train_rows": len(df_train_proc),
        "validation_rows": len(df_val_proc),
        "test_rows": len(df_test_proc),
        "feature_count": len(feature_cols),
        "train_target_p50": float(df_train_proc["durasi_jam"].median()),
        "val_target_p50": float(df_val_proc["durasi_jam"].median()),
        "created_at": pd.Timestamp.now().isoformat()
    }
    save_json(summary, DATA_PROCESSED_REGRESSION / "regression_data_summary.json")
    logger.info("Phase 4: Regression Feature Engineering completed.")

if __name__ == "__main__":
    build_regression_dataset()
