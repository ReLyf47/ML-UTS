import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = "/home/relyf/ML/ML-UTS"
RAW_DIR = os.path.join(ROOT_DIR, "data/raw")

def validate_regression():
    print("=== VALIDATING PREDIKSI_REGRESI.CSV ===")
    pred_path = os.path.join(ROOT_DIR, "prediksi_regresi.csv")
    test_path = os.path.join(RAW_DIR, "regresi_test.csv.gz")
    
    assert os.path.exists(pred_path), "File prediksi_regresi.csv does not exist!"
    df_pred = pd.read_csv(pred_path)
    df_test = pd.read_csv(test_path, usecols=['id_paket'])
    
    # Check shape
    assert len(df_pred) == 60000, f"Expected 60,000 rows, got {len(df_pred)}"
    assert list(df_pred.columns) == ['id_paket', 'p50', 'p90'], f"Invalid columns: {df_pred.columns}"
    
    # Check ID alignment
    assert (df_pred['id_paket'].values == df_test['id_paket'].values).all(), "id_paket does not match test set order!"
    
    # Check constraints
    assert df_pred['p50'].isnull().sum() == 0, "Missing values in p50!"
    assert df_pred['p90'].isnull().sum() == 0, "Missing values in p90!"
    assert (df_pred['p50'] > 0).all(), "Non-positive delivery duration in p50!"
    assert (df_pred['p90'] > 0).all(), "Non-positive delivery duration in p90!"
    
    # Non-crossing
    violations = (df_pred['p90'] < df_pred['p50']).sum()
    assert violations == 0, f"P90 non-crossing condition violated in {violations} rows!"
    
    print(f"PASSED! Rows: {len(df_pred):,}")
    print(f"P50 stats: min={df_pred['p50'].min():.2f}, median={df_pred['p50'].median():.2f}, max={df_pred['p50'].max():.2f}")
    print(f"P90 stats: min={df_pred['p90'].min():.2f}, median={df_pred['p90'].median():.2f}, max={df_pred['p90'].max():.2f}")
    return True

def validate_classification():
    print("\n=== VALIDATING PREDIKSI_KLASIFIKASI.CSV ===")
    pred_path = os.path.join(ROOT_DIR, "prediksi_klasifikasi.csv")
    test_path = os.path.join(RAW_DIR, "klasifikasi_test.csv.gz")
    
    assert os.path.exists(pred_path), "File prediksi_klasifikasi.csv does not exist!"
    df_pred = pd.read_csv(pred_path)
    df_test = pd.read_csv(test_path, usecols=['id_transaksi'])
    
    # Check shape
    assert len(df_pred) == 106674, f"Expected 106,674 rows, got {len(df_pred)}"
    assert list(df_pred.columns) == ['id_transaksi', 'prob', 'tandai'], f"Invalid columns: {df_pred.columns}"
    
    # Check ID alignment
    assert (df_pred['id_transaksi'].values == df_test['id_transaksi'].values).all(), "id_transaksi does not match test set order!"
    
    # Check constraints
    assert df_pred['prob'].isnull().sum() == 0, "Missing values in prob!"
    assert df_pred['tandai'].isnull().sum() == 0, "Missing values in tandai!"
    assert ((df_pred['prob'] >= 0.0) & (df_pred['prob'] <= 1.0)).all(), "Probabilities outside [0, 1]!"
    assert set(df_pred['tandai'].unique()).issubset({0, 1}), "Tandai contains values other than 0 and 1!"
    
    print(f"PASSED! Rows: {len(df_pred):,}")
    print(f"Prob stats: min={df_pred['prob'].min():.6f}, mean={df_pred['prob'].mean():.6f}, max={df_pred['prob'].max():.6f}")
    print(f"Flagged (tandai=1): {df_pred['tandai'].sum():,} ({df_pred['tandai'].mean()*100:.2f}%)")
    return True

def validate_clustering():
    print("\n=== VALIDATING CLUSTER.CSV ===")
    pred_path = os.path.join(ROOT_DIR, "cluster.csv")
    test_path = os.path.join(RAW_DIR, "clustering.csv.gz")
    
    assert os.path.exists(pred_path), "File cluster.csv does not exist!"
    df_pred = pd.read_csv(pred_path)
    df_test = pd.read_csv(test_path, usecols=['id_pelanggan'])
    
    # Check shape
    assert len(df_pred) == 120000, f"Expected 120,000 rows, got {len(df_pred)}"
    assert list(df_pred.columns) == ['id_pelanggan', 'cluster'], f"Invalid columns: {df_pred.columns}"
    
    # Check ID alignment
    assert (df_pred['id_pelanggan'].values == df_test['id_pelanggan'].values).all(), "id_pelanggan does not match dataset order!"
    
    # Check constraints
    assert df_pred['cluster'].isnull().sum() == 0, "Missing values in cluster!"
    assert pd.api.types.is_integer_dtype(df_pred['cluster']), "Cluster column is not integer!"
    assert -1 in df_pred['cluster'].values, "Anomalies (-1) missing from predictions!"
    
    print(f"PASSED! Rows: {len(df_pred):,}")
    print("Cluster value counts:\n" + str(df_pred['cluster'].value_counts()))
    return True

if __name__ == "__main__":
    v1 = validate_regression()
    v2 = validate_classification()
    v3 = validate_clustering()
    if v1 and v2 and v3:
        print("\nALL THREE SUBMISSION CSV FILES FULLY VALIDATED AND COMPLIANT!")
