import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kode.common.config import DATA_PROCESSED_COMMON, DATA_PROCESSED_CLUSTERING
from kode.common.io import read_parquet, save_parquet, save_json
from kode.common.logging_config import setup_logger

logger = setup_logger("06_build_clustering_data", "06_build_clustering_data.log")

PORSI_KATEGORI_COLS = [
    'porsi_fashion', 'porsi_elektronik', 'porsi_kecantikan', 'porsi_rumah_tangga',
    'porsi_makanan_minuman', 'porsi_ibu_bayi', 'porsi_olahraga', 'porsi_hobi',
    'porsi_otomotif', 'porsi_kesehatan', 'porsi_buku_alat_tulis', 'porsi_voucher_digital',
    'porsi_gadget_aksesoris', 'porsi_perabot', 'porsi_pertanian', 'porsi_lainnya'
]

PORSI_JAM_COLS = [
    'porsi_jam_pagi', 'porsi_jam_siang', 'porsi_jam_sore', 'porsi_jam_malam'
]

N_COLS = [f'n{i:02d}' for i in range(1, 11)]

def build_clustering_dataset():
    logger.info("Starting Phase 6: Customer Clustering Feature Engineering...")
    
    # 1. Load cleaned common clustering data
    df_raw = read_parquet(DATA_PROCESSED_COMMON / "cleaned_clustering.parquet")
    logger.info(f"Loaded cleaned_clustering: {len(df_raw):,} rows, {len(df_raw.columns)} cols")
    
    # 2. Extract and preserve customer IDs separately
    df_ids = pd.DataFrame({"id_pelanggan": df_raw["id_pelanggan"].values})
    
    # Verify uniqueness of customer IDs
    assert df_ids["id_pelanggan"].nunique() == len(df_ids), "CRITICAL: Duplicate customer IDs detected!"
    logger.info(f"✓ Verified strict uniqueness for all {len(df_ids):,} customer IDs.")
    
    # 3. Feature Engineering
    df_feat = pd.DataFrame()
    
    # Log1p transforms for heavy-tailed RFM and spending variables
    df_feat["log_belanja_90h"] = np.log1p(np.maximum(0.0, df_raw["belanja_90h"].values))
    df_feat["log_frekuensi_90h"] = np.log1p(np.maximum(0.0, df_raw["frekuensi_90h"].values))
    df_feat["log_rata2_keranjang"] = np.log1p(np.maximum(0.0, df_raw["rata2_keranjang"].values))
    df_feat["log_recency_hari"] = np.log1p(np.maximum(0.0, df_raw["recency_hari"].values))
    df_feat["log_belanja_30h"] = np.log1p(np.maximum(0.0, df_raw["belanja_30h"].values))
    df_feat["log_frekuensi_30h"] = np.log1p(np.maximum(0.0, df_raw["frekuensi_30h"].values))
    
    # Behavioral momentum (30d run-rate vs 90d baseline)
    df_feat["momentum_belanja"] = (df_raw["belanja_30h"].values * 3.0) / (df_raw["belanja_90h"].values + 1000.0)
    df_feat["momentum_frekuensi"] = (df_raw["frekuensi_30h"].values * 3.0) / (df_raw["frekuensi_90h"].values + 1.0)
    
    # Promotion & channel sensitivity
    promo_cols = ['persen_voucher', 'persen_flash_sale', 'persen_paylater', 'persen_cod']
    for c in promo_cols:
        df_feat[c] = df_raw[c].astype(float).values
        
    # Customer experience & friction
    friction_cols = ['retur_90h', 'rating_beri_rata2', 'ulasan_90h']
    for c in friction_cols:
        df_feat[c] = df_raw[c].astype(float).values
        
    # App engagement & conversion
    engagement_cols = ['sesi_per_minggu', 'durasi_sesi_menit', 'rasio_konversi']
    for c in engagement_cols:
        df_feat[c] = df_raw[c].astype(float).values
        
    # Demographics & device
    df_feat['lama_member_bulan'] = df_raw['lama_member_bulan'].astype(float).values
    df_feat['umur'] = df_raw['umur'].astype(float).values
    df_feat['kota_tier'] = df_raw['kota_tier'].astype(float).values
    df_feat['is_android'] = (df_raw['perangkat'] == 'android').astype(int).values
    df_feat['is_ios'] = (df_raw['perangkat'] == 'ios').astype(int).values
    df_feat['is_web'] = (df_raw['perangkat'] == 'web').astype(int).values
    
    # Commercial intent & basket structure
    intent_cols = ['jumlah_alamat', 'wishlist', 'klik_iklan_30h', 'item_per_pesanan']
    for c in intent_cols:
        df_feat[c] = df_raw[c].astype(float).values
        
    # Compositional category shares (16 categories)
    for c in PORSI_KATEGORI_COLS:
        df_feat[c] = df_raw[c].astype(float).values
        
    # Compositional time-of-day shares (4 time bins)
    for c in PORSI_JAM_COLS:
        df_feat[c] = df_raw[c].astype(float).values
        
    # 10 internal CRM attributes
    for n in N_COLS:
        df_feat[n] = df_raw[n].astype(float).values
        
    # Verify no NaN or infinite values
    assert not df_feat.isnull().any().any(), "CRITICAL: NaN values present in clustering feature matrix!"
    assert np.all(np.isfinite(df_feat.values)), "CRITICAL: Infinite values present in clustering feature matrix!"
    
    # 4. Fit StandardScaler
    logger.info(f"Fitting StandardScaler on {len(df_feat):,} customer profiles across {len(df_feat.columns)} features...")
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df_feat.values)
    df_scaled = pd.DataFrame(scaled_matrix, columns=df_feat.columns)
    
    # 5. Save Artifacts
    feat_path = DATA_PROCESSED_CLUSTERING / "customer_features.parquet"
    scaled_path = DATA_PROCESSED_CLUSTERING / "customer_features_scaled.parquet"
    ids_path = DATA_PROCESSED_CLUSTERING / "customer_ids.parquet"
    
    save_parquet(df_feat, feat_path)
    save_parquet(df_scaled, scaled_path)
    save_parquet(df_ids, ids_path)
    
    logger.info(f"✓ Saved unscaled features: {feat_path.name} ({len(df_feat):,} rows, {len(df_feat.columns)} cols)")
    logger.info(f"✓ Saved scaled features: {scaled_path.name} ({len(df_scaled):,} rows, {len(df_scaled.columns)} cols)")
    logger.info(f"✓ Saved customer IDs: {ids_path.name} ({len(df_ids):,} rows)")
    
    # Metadata
    feature_meta = {
        "feature_count": len(df_feat.columns),
        "feature_names": list(df_feat.columns),
        "id_column": "id_pelanggan",
        "sample_size": len(df_feat),
        "category_count": len(PORSI_KATEGORI_COLS),
        "time_bin_count": len(PORSI_JAM_COLS)
    }
    save_json(feature_meta, DATA_PROCESSED_CLUSTERING / "feature_columns.json")
    
    prep_meta = {
        "scaler": "StandardScaler",
        "scaler_means": {col: float(m) for col, m in zip(df_feat.columns, scaler.mean_)},
        "scaler_scales": {col: float(s) for col, s in zip(df_feat.columns, scaler.scale_)},
        "reconstructed_missing_baskets": int(df_raw["rata2_keranjang"].isnull().sum())
    }
    save_json(prep_meta, DATA_PROCESSED_CLUSTERING / "preprocessing_metadata.json")
    
    summary = {
        "customer_count": len(df_feat),
        "feature_count": len(df_feat.columns),
        "unique_id_ratio": 1.0,
        "scaled_matrix_finite": bool(np.all(np.isfinite(scaled_matrix))),
        "created_at": pd.Timestamp.now().isoformat()
    }
    save_json(summary, DATA_PROCESSED_CLUSTERING / "clustering_data_summary.json")
    logger.info("Phase 6: Customer Clustering Feature Engineering completed.")

if __name__ == "__main__":
    build_clustering_dataset()
