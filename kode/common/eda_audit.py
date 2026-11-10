import os
import gzip
import json
import pandas as pd
import numpy as np

DATA_DIR = "/home/relyf/ML/ML-UTS/data/raw"

def get_file_info(filepath):
    stat = os.stat(filepath)
    size_mb = stat.st_size / (1024 * 1024)
    return size_mb

def audit_regression():
    train_path = os.path.join(DATA_DIR, "regresi_train.csv.gz")
    test_path = os.path.join(DATA_DIR, "regresi_test.csv.gz")
    
    print("=== AUDIT REGRESSION ===")
    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)
    
    print(f"Train Shape: {df_tr.shape}, Size: {get_file_info(train_path):.2f} MB")
    print(f"Test Shape: {df_te.shape}, Size: {get_file_info(test_path):.2f} MB")
    
    # ID check
    print(f"Train unique id_paket: {df_tr['id_paket'].nunique()} / {len(df_tr)}")
    print(f"Test unique id_paket: {df_te['id_paket'].nunique()} / {len(df_te)}")
    overlap = set(df_tr['id_paket']).intersection(set(df_te['id_paket']))
    print(f"ID Overlap: {len(overlap)}")
    
    # Columns comparison
    diff_cols = set(df_tr.columns) - set(df_te.columns)
    print(f"Columns in train but not test: {diff_cols}")
    
    # Dates
    print(f"Train tgl_pickup range: {df_tr['tgl_pickup'].min()} to {df_tr['tgl_pickup'].max()}")
    print(f"Test tgl_pickup range: {df_te['tgl_pickup'].min()} to {df_te['tgl_pickup'].max()}")
    
    # Missing values
    tr_null = df_tr.isnull().sum()
    te_null = df_te.isnull().sum()
    print("Train missing columns:", tr_null[tr_null > 0].to_dict())
    print("Test missing columns:", te_null[te_null > 0].to_dict())
    
    # Target stats
    target = df_tr['durasi_jam']
    print(f"durasi_jam stats: min={target.min()}, max={target.max()}, mean={target.mean():.2f}, "
          f"std={target.std():.2f}, median={target.median():.2f}, "
          f"p90={np.percentile(target, 90):.2f}, p99={np.percentile(target, 99):.2f}, p99.9={np.percentile(target, 99.9):.2f}")
    
    # Check LEAKAGE suspect: jumlah_scan
    print("\n--- LEAKAGE CHECK: jumlah_scan ---")
    print("Train jumlah_scan describe:\n", df_tr['jumlah_scan'].describe())
    print("Test jumlah_scan describe:\n", df_te['jumlah_scan'].describe())
    corr_scan = df_tr[['jumlah_scan', 'durasi_jam']].corr().iloc[0, 1]
    print(f"Correlation jumlah_scan with durasi_jam in train: {corr_scan:.4f}")
    
    # Categoricals
    cat_cols = df_tr.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCategorical columns ({len(cat_cols)}): {cat_cols}")
    for col in ['layanan', 'mitra_kurir', 'pulau_tujuan']:
        if col in df_tr.columns:
            print(f"Unique {col} in train: {df_tr[col].nunique()}, test: {df_te[col].nunique()}")
            unseen = set(df_te[col].dropna()) - set(df_tr[col].dropna())
            print(f"Unseen in test for {col}: {len(unseen)}")

def audit_classification():
    train_path = os.path.join(DATA_DIR, "klasifikasi_train.csv.gz")
    test_path = os.path.join(DATA_DIR, "klasifikasi_test.csv.gz")
    
    print("\n=== AUDIT CLASSIFICATION ===")
    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)
    
    print(f"Train Shape: {df_tr.shape}, Size: {get_file_info(train_path):.2f} MB")
    print(f"Test Shape: {df_te.shape}, Size: {get_file_info(test_path):.2f} MB")
    
    # ID check
    print(f"Train unique id_transaksi: {df_tr['id_transaksi'].nunique()} / {len(df_tr)}")
    print(f"Test unique id_transaksi: {df_te['id_transaksi'].nunique()} / {len(df_te)}")
    overlap = set(df_tr['id_transaksi']).intersection(set(df_te['id_transaksi']))
    print(f"ID Overlap: {len(overlap)}")
    
    diff_cols = set(df_tr.columns) - set(df_te.columns)
    print(f"Columns in train but not test: {diff_cols}")
    
    # Dates / times
    print(f"Train waktu range: {df_tr['waktu'].min()} to {df_tr['waktu'].max()}")
    print(f"Test waktu range: {df_te['waktu'].min()} to {df_te['waktu'].max()}")
    
    # Missing values
    tr_null = df_tr.isnull().sum()
    te_null = df_te.isnull().sum()
    print("Train missing columns:", tr_null[tr_null > 0].to_dict())
    print("Test missing columns:", te_null[te_null > 0].to_dict())
    
    # Target stats
    target = df_tr['is_fraud']
    print(f"is_fraud distribution:\n{target.value_counts(normalize=True).to_dict()}")
    print(f"Total fraud count: {target.sum()} out of {len(target)}")
    
    # Check LEAKAGE suspect: flag_review_manual
    print("\n--- LEAKAGE CHECK: flag_review_manual ---")
    print("Train flag_review_manual distribution:\n", df_tr['flag_review_manual'].value_counts().to_dict())
    print("Test flag_review_manual distribution:\n", df_te['flag_review_manual'].value_counts().to_dict())
    print("Crosstab flag_review_manual vs is_fraud in train:\n", pd.crosstab(df_tr['flag_review_manual'], df_tr['is_fraud'], margins=True))
    
    # Nominal by fraud
    print("\nNominal stats by fraud:")
    print("Fraud nominal:", df_tr.loc[df_tr['is_fraud'] == 1, 'nominal'].describe())
    print("Non-fraud nominal:", df_tr.loc[df_tr['is_fraud'] == 0, 'nominal'].describe())

def audit_clustering():
    clust_path = os.path.join(DATA_DIR, "clustering.csv.gz")
    print("\n=== AUDIT CLUSTERING ===")
    df = pd.read_csv(clust_path)
    print(f"Clustering Shape: {df.shape}, Size: {get_file_info(clust_path):.2f} MB")
    print(f"Unique id_pelanggan: {df['id_pelanggan'].nunique()} / {len(df)}")
    
    null_counts = df.isnull().sum()
    print("Missing columns:", null_counts[null_counts > 0].to_dict())
    
    # Category sum check
    porsi_kat_cols = [c for c in df.columns if c.startswith('porsi_') and not c.startswith('porsi_jam_')]
    print(f"Category proportion columns ({len(porsi_kat_cols)}): {porsi_kat_cols}")
    kat_sum = df[porsi_kat_cols].sum(axis=1)
    print(f"Porsi category row sum stats: min={kat_sum.min():.4f}, max={kat_sum.max():.4f}, mean={kat_sum.mean():.4f}")
    
    # Time sum check
    porsi_jam_cols = [c for c in df.columns if c.startswith('porsi_jam_')]
    print(f"Time proportion columns ({len(porsi_jam_cols)}): {porsi_jam_cols}")
    jam_sum = df[porsi_jam_cols].sum(axis=1)
    print(f"Porsi jam row sum stats: min={jam_sum.min():.4f}, max={jam_sum.max():.4f}, mean={jam_sum.mean():.4f}")

if __name__ == "__main__":
    audit_regression()
    audit_classification()
    audit_clustering()
