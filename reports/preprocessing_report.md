# Comprehensive Preprocessing Report

**Project**: UTS Machine Learning - Autonomous Data Engineering Pipeline  
**Date**: November 10, 2026  
**Status**: Executed & Validated  

---

## 1. Pipeline Architecture Overview

The data engineering pipeline processes approximately 940,000 tabular records across logistics, fintech fraud risk, and customer behavioral segmentation. The pipeline follows a strict modular flow:
1. `01_inspect_data.py`: Profiling, metadata discovery, initial data quality audit.
2. `02_extract_data.py`: Interim normalization, whitespace sanitization, Parquet storage.
3. `03_clean_common.py`: Shared missing value imputations, typing, and identifier sanitization.
4. `04_build_regression_data.py`: Feature engineering, smoothed target encoding, out-of-time chronological partitioning, and strict target leakage removal.
5. `05_build_classification_data.py`: Structural missingness handling, temporal and transaction velocity features, chronological out-of-time splitting.
6. `06_build_clustering_data.py`: RFM log1p transformations, behavioral momentum, compositional category shares, and StandardScaler normalization.
7. `07_validate_processed_data.py`: Comprehensive schema, row count, null count, and leakage validation.

---

## 2. Dataset-by-Dataset Transformation Design

### 2.1 Task 1: Regression (`durasi_jam`)

- **Source Datasets**: `regresi_train.csv` (250,000 rows) and `regresi_test.csv` (60,000 rows).
- **Primary Identifier**: `id_paket` (Retained as string metadata, dropped from training feature matrix).
- **Target Variable**: `durasi_jam` (Continuous hours, present only in train/validation).
- **Fatal Leakage Feature**: `jumlah_scan`
  - *Decision*: **EXCLUDED completely** from all regression feature sets.
  - *Reason*: Tracking scans accumulate over the delivery lifecycle (mean 9.65 in train vs 1.50 in test). Using this feature leaks post-pickup journey duration and causes catastrophic under-prediction at test time.
- **Missing Value Handling**:
  - `berat_kg`: 5.56% missing in train, 5.44% in test.
  - *Strategy*: Deterministic median imputation conditioned on service tier (`layanan`) and volume, fitted strictly on the training partition. A binary indicator `berat_kg_missing` is preserved.
- **Categorical Encoding Strategy**:
  - Low-cardinality categoricals (`layanan`, `mitra_kurir`, `hub_asal`, `kota_asal`, `pulau_tujuan`): 1-indexed integer encoding, reserving index `0` for unseen categories. This gracefully accommodates the test-only courier partner `'K13'`.
  - High-cardinality destination city (`kota_tujuan`, 405 categories): $m$-estimate smoothed target encoding ($m=20$) and empirical frequency encoding, fitted strictly on the train partition.
- **Temporal Transformations**:
  - `tgl_pickup` parsed to datetime. Extracted `pickup_dayofweek`, `pickup_is_weekend`, `pickup_month`, `pickup_day`.
  - `jam_pickup` transformed via sine and cosine cyclical encodings: $\sin(2\pi \cdot \text{hour} / 24)$ and $\cos(2\pi \cdot \text{hour} / 24)$.
- **Domain Ratios**:
  - Volumetric density: $\text{densitas\_kg\_per\_l} = \text{berat\_kg} / (\text{volume\_liter} + 10^{-4})$.
  - Value per kg: $\text{nilai\_barang} / (\text{berat\_kg} + 10^{-4})$.
  - Same-city indicator: $\mathbb{I}(\text{kota\_asal} == \text{kota\_tujuan})$.
- **Partitioning**:
  - Chronological split:
    - Train: `tgl_pickup < 2025-08-01` (215,832 rows, Jan-Jul 2025).
    - Validation: `tgl_pickup >= 2025-08-01` (34,168 rows, Aug 2025).
    - Test: 60,000 rows (Sep-Oct 2025).

---

### 2.2 Task 2: Classification (`is_fraud`)

- **Source Datasets**: `klasifikasi_train.csv` (403,077 rows) and `klasifikasi_test.csv` (106,674 rows).
- **Primary Identifier**: `id_transaksi` (Retained as string metadata, preserved for submission matching).
- **Target Variable**: `is_fraud` (Binary 0/1, 0.65% positive rate in train, present only in train/val).
- **Structural Missingness Handling**:
  - `menit_sejak_trx_terakhir` and `rasio_nominal_rata2`: Missing in 25.46% of train rows and 10.74% of test rows.
  - *Mechanism*: Structural zero history for first-time account transactions.
  - *Strategy*: Created explicit binary flag `is_first_trx = 1`. Imputed `menit_sejak_trx_terakhir` with `-1.0` and `rasio_nominal_rata2` with baseline `1.0`.
- **Financial Ratios & Log Scaling**:
  - `log_nominal = log1p(nominal)`.
  - `log_saldo_sebelum = log1p(max(0, saldo_sebelum))`.
  - `nominal_to_saldo_ratio = nominal / (saldo_sebelum + 1000)`.
  - `velocity_ratio_1j_24j = trx_1j / (trx_24j + 1)`.
- **Categorical Encoding Strategy**:
  - `kategori_merchant`, `kanal`, `tipe_perangkat`, `email_domain` mapped using 1-indexed integers with 0 reserved for unseen tokens.
- **Partitioning**:
  - Chronological split:
    - Train: `waktu < 2025-08-01` (350,173 rows, Jan-Jul 2025; 2,286 fraud cases, 0.653%).
    - Validation: `waktu >= 2025-08-01` (52,904 rows, Aug 2025; 336 fraud cases, 0.635%).
    - Test: 106,674 rows (Sep-Oct 2025).

---

### 2.3 Task 3: Clustering (Customer Segmentation)

- **Source Dataset**: `clustering.csv` (120,000 rows, 60 columns).
- **Primary Identifier**: `id_pelanggan` (Extracted and saved separately to `customer_ids.parquet`; excluded from clustering feature matrix to prevent identifier memorization).
- **Deterministic Reconstruction**:
  - `rata2_keranjang`: 5.61% missing. Reconstructed as `belanja_90h / max(1, frekuensi_90h)`.
- **Heavy-Tail Mitigations**:
  - `log1p` applied to: `belanja_90h`, `frekuensi_90h`, `rata2_keranjang`, `recency_hari`, `belanja_30h`, `frekuensi_30h`.
- **Behavioral Momentum**:
  - $\text{momentum\_belanja} = (\text{belanja\_30h} \times 3) / (\text{belanja\_90h} + 1000)$.
  - $\text{momentum\_frekuensi} = (\text{frekuensi\_30h} \times 3) / (\text{frekuensi\_90h} + 1)$.
- **Compositional Data**:
  - 16 category proportions (`porsi_fashion`, `porsi_elektronik`, etc.) and 4 time-of-day proportions (`porsi_jam_pagi`, etc.) preserved as bounded continuous features summing to 1.0.
- **Feature Scaling**:
  - `StandardScaler` applied to all 58 engineered numerical features, producing `customer_features_scaled.parquet` with zero mean and unit variance. Both unscaled and scaled matrices are persisted with exact row ordering matching `customer_ids.parquet`.

---

## 3. Summary of Processed Artifacts

| Task | Partition / File | Rows | Columns | Null Count | Target Available |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Common | `cleaned_regresi_train.parquet` | 250,000 | 78 | 0 | Yes (`durasi_jam`) |
| Common | `cleaned_regresi_test.parquet` | 60,000 | 77 | 0 | No |
| Common | `cleaned_klasifikasi_train.parquet` | 403,077 | 86 | 0 | Yes (`is_fraud`) |
| Common | `cleaned_klasifikasi_test.parquet` | 106,674 | 85 | 0 | No |
| Common | `cleaned_clustering.parquet` | 120,000 | 60 | 0 | N/A |
| Regression | `train.parquet` | 215,832 | 86 | 0 | Yes (`durasi_jam`) |
| Regression | `validation.parquet` | 34,168 | 86 | 0 | Yes (`durasi_jam`) |
| Regression | `test.parquet` | 60,000 | 85 | 0 | No |
| Classification | `train.parquet` | 350,173 | 90 | 0 | Yes (`is_fraud`) |
| Classification | `validation.parquet` | 52,904 | 90 | 0 | Yes (`is_fraud`) |
| Classification | `test.parquet` | 106,674 | 89 | 0 | No |
| Clustering | `customer_features.parquet` | 120,000 | 58 | 0 | N/A |
| Clustering | `customer_features_scaled.parquet` | 120,000 | 58 | 0 | N/A |
| Clustering | `customer_ids.parquet` | 120,000 | 1 | 0 | N/A |
