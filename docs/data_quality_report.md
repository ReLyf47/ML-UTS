# Initial Data Quality Report

**Project**: UTS Machine Learning  
**Date**: November 10, 2026  
**Auditor**: Autonomous Machine Learning Engineering Team  

---

## 1. Executive Summary

Data quality across all five datasets (`regresi_train.csv.gz`, `regresi_test.csv.gz`, `klasifikasi_train.csv.gz`, `klasifikasi_test.csv.gz`, and `clustering.csv.gz`) is generally high with strictly unique primary keys and clean formatting. However, key distributional phenomena, extreme outliers, informative missingness, and one unseen category in the test set require structured preprocessing and mitigation.

---

## 2. Completeness & Missing Value Analysis

| Dataset | Column Name | Missing Count | Missing Rate | Missingness Mechanism & Engineering Action |
| :--- | :--- | :---: | :---: | :--- |
| `regresi_train` | `berat_kg` | 13,897 | 5.56% | Missing completely at random (MCAR/MAR). Impute using median weight grouped by (`volume_liter` quantile or `layanan`). |
| `regresi_test` | `berat_kg` | 3,265 | 5.44% | Identical missingness rate to train (~5.5%). Handled identically via train-fitted imputer. |
| `klasifikasi_train` | `menit_sejak_trx_terakhir` | 102,621 | 25.46% | **Structural Missingness**: Users executing their very first transaction have no prior transaction history. Impute with distinct high value (e.g. 99,999 or -1) and add binary indicator `is_first_transaction`. |
| `klasifikasi_train` | `rasio_nominal_rata2` | 102,621 | 25.46% | **Structural Missingness**: Matches `menit_sejak_trx_terakhir` row-for-row. First-time users have no historical average spend. Impute with neutral baseline (1.0) and flag with `is_first_transaction`. |
| `klasifikasi_test` | `menit_sejak_trx_terakhir` | 11,458 | 10.74% | Lower first-time user proportion in test (10.74% vs 25.46%), reflecting user base maturation over time. |
| `klasifikasi_test` | `rasio_nominal_rata2` | 11,458 | 10.74% | Exactly matches `menit_sejak_trx_terakhir`. Handled identically. |
| `clustering` | `rata2_keranjang` | 6,736 | 5.61% | Missing basket value. Reconstructed deterministically as `belanja_90h / frekuensi_90h` (if `frekuensi_90h > 0`) or 0. |

*All remaining 75 regression columns, 83 classification columns, and 59 clustering columns have 0 missing values.*

---

## 3. Identifiers & Deduplication

- **Regression**:
  - `id_paket` is 100% unique in `regresi_train` (250,000 / 250,000).
  - `id_paket` is 100% unique in `regresi_test` (60,000 / 60,000).
  - Intersection between train and test IDs: 0. Zero test contamination.
- **Classification**:
  - `id_transaksi` is 100% unique in `klasifikasi_train` (403,077 / 403,077).
  - `id_transaksi` is 100% unique in `klasifikasi_test` (106,674 / 106,674).
  - Intersection between train and test IDs: 0.
- **Clustering**:
  - `id_pelanggan` is 100% unique in `clustering` (120,000 / 120,000).

---

## 4. Target Distribution & Outliers

### 4.1 Regression Target (`durasi_jam`)
- **Summary**:
  - Min: 11.60 hours
  - 25th percentile: 43.80 hours
  - Median (50th percentile): 61.70 hours (~2.57 days)
  - Mean: 77.84 hours
  - 75th percentile: 86.40 hours
  - 90th percentile: 117.80 hours
  - 99th percentile: 224.70 hours (~9.36 days)
  - Max: 25,452.00 hours (~2.9 years!)
- **Findings**:
  - The target distribution exhibits a heavy positive skew with extreme outliers (top 0.1% reaches up to 25,452 hours, representing lost, stuck, or corrupted package delivery logs).
  - Mean (77.84) is significantly pulled above median (61.70) by the long tail.
  - Using MSE/RMSE loss directly would cause severe distortion from these massive outliers.
  - Pinball loss for median (P50) and 90th percentile (P90) is robust to extreme tail anomalies. For point estimates, log-transformed targets or Huber/MAE losses are strongly recommended.

### 4.2 Classification Target (`is_fraud`)
- **Summary**:
  - Total transactions: 403,077
  - Non-fraud (`0`): 400,455 (99.35%)
  - Fraud (`1`): 2,622 (0.65%)
  - Imbalance ratio: ~153 to 1.
- **Findings**:
  - Severe class imbalance. Accuracy is meaningless (a trivial all-zero predictor achieves 99.35% accuracy but catches 0 fraud and incurs massive financial losses).
  - Primary evaluation metric must be PR-AUC (Average Precision) and Business Cost Curve.
  - Fraud transaction amounts (`nominal`): Mean = Rp 1,209,456 (Median = Rp 370,000), compared to Non-fraud mean = Rp 297,283 (Median = Rp 127,600). Fraud ticket sizes are ~4x larger on average, making missed fraud financially devastating.

### 4.3 Clustering Composition & Dimensions
- The 16 category proportions (`porsi_fashion`, `porsi_elektronik`, etc.) sum precisely to 1.0 (mean = 1.0000, min = 0.9996, max = 1.0005, slight floating-point tolerance).
- The 4 time-of-day proportions (`porsi_jam_pagi`, etc.) sum precisely to 1.0 (mean = 1.0000).
- Skewed monetary variables (`belanja_90h`, `rata2_keranjang`, `belanja_30h`) show power-law distributions requiring log-scaling.

---

## 5. Domain Support & Unseen Categories

### 5.1 Regression Categoricals
- `layanan` (4 levels): Identical in train and test (`Instant`, `Same Day`, `Reguler`, `Kargo`).
- `pulau_tujuan` (7 levels): Identical in train and test.
- `hub_asal` (6 levels): Identical in train and test.
- `kota_asal` (30 levels): Identical in train and test.
- `kota_tujuan` (405 levels): Identical in train and test (all 405 test destination cities appeared in train).
- **`mitra_kurir` (12 levels in train, 13 levels in test)**:
  - Partner `'K13'` appears **only** in the test set!
  - **Remediation**: The categorical encoder must map `'K13'` to an `unseen_category` / `other` token or impute with global prior distributions to prevent inference crashes or out-of-vocabulary errors.

### 5.2 Classification Categoricals
- `kategori_merchant` (24 levels): 0 unseen in test.
- `kanal` (5 levels): 0 unseen in test.
- `tipe_perangkat` (3 levels): 0 unseen in test.
- `email_domain` (5 levels): 0 unseen in test.

### 5.3 Clustering Categoricals
- `perangkat`: 3 levels (`android`: 77,760, `ios`: 31,710, `web`: 10,530).
- `kota_tier`: 3 ordinal tiers (Tier 1: 47,961, Tier 2: 41,741, Tier 3: 30,298).
