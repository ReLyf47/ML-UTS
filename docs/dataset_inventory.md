# Dataset Inventory

**Project**: UTS Machine Learning - Regression, Classification, and Clustering  
**Audit Date**: November 10, 2026  
**Environment**: Linux x86_64, Python 3.14.8  

---

## 1. Overview and Summary Statistics

The project provides five (5) synthetic tabular dataset files stored as gzip-compressed CSV files in `data/raw/`, accompanied by `kamus_data.md`. Across all datasets, there are **939,751 total rows** (~940,000 rows as specified in the assignment).

| Dataset File | Task | Split / Role | Rows | Columns | Compressed Size | Uncompressed Memory | Temporal Coverage |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `regresi_train.csv.gz` | Regression | Training | 250,000 | 77 | 35.35 MB | ~147.2 MB | 2025-01-01 to 2025-08-31 |
| `regresi_test.csv.gz` | Regression | Hidden Test | 60,000 | 76 | 8.30 MB | ~34.8 MB | 2025-09-01 to 2025-10-31 |
| `klasifikasi_train.csv.gz` | Classification | Training | 403,077 | 85 | 65.42 MB | ~262.1 MB | 2025-01-01 to 2025-08-31 |
| `klasifikasi_test.csv.gz` | Classification | Hidden Test | 106,674 | 84 | 17.39 MB | ~68.4 MB | 2025-09-01 to 2025-10-31 |
| `clustering.csv.gz` | Clustering | Unsupervised | 120,000 | 60 | 14.96 MB | ~55.0 MB | 90-day aggregate snapshot |
| **Total** | | | **939,751** | | **141.42 MB** | **~567.5 MB** | |

---

## 2. Granular Inventory by Task

### 2.1 Task 1: Regression (Delivery Time Estimation)
- **Target Variable**: `durasi_jam` (Continuous, float hours from package pickup to delivery). Present ONLY in `regresi_train.csv.gz`.
- **Primary Identifier**: `id_paket` (String/alphanumeric identifier).
  - `regresi_train.csv.gz`: 250,000 rows, 250,000 unique `id_paket` (100% unique, zero duplicates).
  - `regresi_test.csv.gz`: 60,000 rows, 60,000 unique `id_paket` (100% unique, zero duplicates).
  - **ID Overlap**: 0 overlapping packages between train and test.
- **Temporal Alignment**:
  - Training window: January 1, 2025 to August 31, 2025 (8 months).
  - Test window: September 1, 2025 to October 31, 2025 (2 months).
  - Strictly non-overlapping chronological split.

### 2.2 Task 2: Classification (Fraud Detection)
- **Target Variable**: `is_fraud` (Binary integer: `1` = confirmed fraud, `0` = non-fraud). Present ONLY in `klasifikasi_train.csv.gz`.
- **Primary Identifier**: `id_transaksi`.
  - `klasifikasi_train.csv.gz`: 403,077 rows, 403,077 unique `id_transaksi` (100% unique, zero duplicates).
  - `klasifikasi_test.csv.gz`: 106,674 rows, 106,674 unique `id_transaksi` (100% unique, zero duplicates).
  - **ID Overlap**: 0 overlapping transactions between train and test.
- **Entities**:
  - `id_pengguna`: 120,000 unique users in train, users reappear across transactions.
  - `id_merchant`: 15,000 unique merchants in train.
- **Temporal Alignment**:
  - Training window: 2025-01-01 00:02:53 to 2025-08-31 23:58:04 (8 months).
  - Test window: 2025-09-01 00:01:51 to 2025-10-31 23:57:22 (2 months).
  - Strictly non-overlapping chronological split.

### 2.3 Task 3: Clustering (Customer Segmentation)
- **Target Variable**: None (Unsupervised learning task). The lecturer holds private hidden ground truth clusters for evaluation.
- **Primary Identifier**: `id_pelanggan`.
  - `clustering.csv.gz`: 120,000 rows, 120,000 unique `id_pelanggan` (100% unique, zero duplicates).
- **Temporal Nature**: A single cross-sectional behavioral aggregation representing the past 90 days.

---

## 3. Directory and Storage Constraints
- Original raw datasets are archived in `data/raw/` and marked read-only in pipeline logic.
- Processed feature matrices and intermediate artifacts will be stored in `data/processed/`.
- Predictions will be output at root:
  - `prediksi_regresi.csv` (60,000 rows, columns: `id_paket`, `p50`, `p90`)
  - `prediksi_klasifikasi.csv` (106,674 rows, columns: `id_transaksi`, `prob`, `tandai`)
  - `cluster.csv` (120,000 rows, columns: `id_pelanggan`, `cluster`)
