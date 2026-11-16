# Comprehensive Data Quality Report

**Generated**: 2026-11-16 11:21:46
**Pipeline**: Autonomous Data Engineering (Phase 1)

---

## 1. Executive Summary & Inventory

| Dataset | Task | Role | Rows | Columns | Size (MB) | ID Column | Unique IDs | Time Span |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `regresi_train.csv` | Regression | train | 250,000 | 77 | 99.36 | `id_paket` | 250,000 (100%) | 2025-01-01 to 2025-08-31 |
| `regresi_test.csv` | Regression | test | 60,000 | 76 | 23.54 | `id_paket` | 60,000 (100%) | 2025-09-01 to 2025-10-31 |
| `klasifikasi_train.csv` | Classification | train | 403,077 | 85 | 185.41 | `id_transaksi` | 403,077 (100%) | 2025-01-01 to 2025-08-31 |
| `klasifikasi_test.csv` | Classification | test | 106,674 | 84 | 49.05 | `id_transaksi` | 106,674 (100%) | 2025-09-01 to 2025-10-31 |
| `clustering.csv` | Clustering | unsupervised | 120,000 | 60 | 39.27 | `id_pelanggan` | 120,000 (100%) | Cross-sectional (90d) |

**Total Record Volume Across All Tasks**: **939,751 rows** (approx. 940,000 rows as specified in the assignment).

---

## 2. Completeness & Missing Value Audit

| Dataset | Column Name | Missing Count | Missing Pct | Mechanism & Preprocessing Strategy |
| :--- | :--- | :---: | :---: | :--- |
| `regresi_train` | `berat_kg` | 13,897 | 5.559% | MCAR/MAR. Median imputation grouped by `layanan` (service tier) and volume quantile. |
| `regresi_test` | `berat_kg` | 3,265 | 5.442% | Handled via imputer fitted strictly on training data. |
| `klasifikasi_train` | `menit_sejak_trx_terakhir` | 102,621 | 25.460% | Structural missingness (first-time users). Impute with -1.0; add explicit binary flag `is_first_trx=1`. |
| `klasifikasi_train` | `rasio_nominal_rata2` | 102,621 | 25.460% | Structural missingness (first-time users). Impute with baseline 1.0; flag with `is_first_trx=1`. |
| `klasifikasi_test` | `menit_sejak_trx_terakhir` | 11,458 | 10.741% | Structural missingness. Handled identically with `is_first_trx` flag. |
| `klasifikasi_test` | `rasio_nominal_rata2` | 11,458 | 10.741% | Structural missingness. Handled identically. |
| `clustering` | `rata2_keranjang` | 6,736 | 5.613% | Deterministic omission. Reconstructed as `belanja_90h / max(1, frekuensi_90h)`. |

All other 75 regression columns, 83 classification columns, and 59 clustering columns exhibit **0 missing values**.

---

## 3. Duplicate and Identifier Integrity Audit

- **Zero Exact Duplicate Rows**: Verified across all 5 raw datasets.
- **Strict Identifier Uniqueness**: Primary keys (`id_paket`, `id_transaksi`, `id_pelanggan`) are 100% unique within their respective partitions.
- **Zero Split Contamination**: 0 overlapping `id_paket` between regression train and test; 0 overlapping `id_transaksi` between classification train and test.

---

## 4. Target Distributions & Skewness

### 4.1 Regression Target (`durasi_jam`)
- **Sample Count**: 250,000
- **Quantiles**: Min=11.60h, P25=43.80h, P50 (Median)=61.70h, P75=86.40h, P90=117.80h, P99=224.70h, Max=25,452.00h.
- **Assessment**: Severe right-skew with extreme long tail (top 0.1% delayed/lost packages). Pinball quantile regression at P50 and P90 is mathematically immune to such tail perturbations.

### 4.2 Classification Target (`is_fraud`)
- **Sample Count**: 403,077
- **Class Breakdown**: Class 0 (Legitimate) = 400,455 (99.35%); Class 1 (Fraud) = 2,622 (0.65%).
- **Imbalance Ratio**: 1 : 152.7.
- **Financial Impact**: Mean nominal fraud ticket is Rp 1,209,456 vs legitimate ticket Rp 297,283 (4.07x higher).

---

## 5. Unseen Categorical Audit in Test Partitions

- **Courier Partner (`mitra_kurir`)**: Train has 12 unique couriers (`K01`-`K12`). Test contains an unseen partner `'K13'`.
- **Action**: Category mapping uses 1-based indexing, reserving index `0` for unseen tokens, preventing runtime crashes and gracefully routing unseen couriers to prior baselines.