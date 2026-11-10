# Comprehensive Leakage Audit & Feature Risk Assessment

**Project**: UTS Machine Learning  
**Date**: November 10, 2026  
**Auditor**: Autonomous Machine Learning Engineering Team  

---

## 1. Executive Summary

A critical component of production-grade Machine Learning is ensuring that models do not rely on post-outcome signals, future information, or artifacts that do not exist at the exact decision cutoff. 

Our deep audit of the provided datasets identified **one fatal post-outcome target leakage feature (`jumlah_scan` in Task 1)**, **one high-risk operational policy feature (`flag_review_manual` in Task 2)**, and established strict boundaries for temporal feature engineering.

---

## 2. Fatal Leakage in Task 1: `jumlah_scan`

### 2.1 What the Feature Represents
According to `kamus_data.md`, `jumlah_scan` represents:
> *"Jumlah scan pelacakan yang tercatat di sistem"* (Number of tracking scans recorded in the system).

The task description specifies:
> *"Satu baris per paket. Prediksi dibuat saat paket diambil kurir."* (One row per package. Predictions are made at the moment the package is picked up by the courier).

### 2.2 The Anomaly: Distributional Divergence
Empirical comparison of `jumlah_scan` between training and test sets reveals an irreconcilable discrepancy:

| Metric | `regresi_train.csv.gz` | `regresi_test.csv.gz` | Divergence Ratio |
| :--- | :---: | :---: | :---: |
| **Mean** | **9.65 scans** | **1.50 scans** | **6.4x lower in test** |
| **Standard Deviation** | 5.63 | 0.71 | 7.9x lower in test |
| **Minimum** | 1 scan | 1 scan | - |
| **25th Percentile** | 6.0 scans | 1.0 scan | - |
| **Median (50th)** | **8.0 scans** | **1.0 scan** | **8.0x lower in test** |
| **75th Percentile** | 12.0 scans | 2.0 scans | 6.0x lower in test |
| **Maximum** | **97.0 scans** | **7.0 scans** | 13.9x lower in test |
| Correlation with `durasi_jam` | +0.1887 | N/A | Positive correlation |

### 2.3 Why This Is Fatal Target Leakage
1. **Physical Reality**: At the exact moment of courier pickup, a package has only experienced the initial dispatch/pickup scan (1 to 2 scans at most). It is physically impossible for a newly picked-up package to have 8, 12, or 97 scans.
2. **Data Logging Failure in Training Set**: In `regresi_train.csv.gz`, `jumlah_scan` was logged retrospectively at the end of the package delivery lifecycle (accumulating all checkpoint scans through sorting hubs, transit stops, and delivery attempts). In contrast, in `regresi_test.csv.gz`, the snapshot correctly captures the system state at pickup time (mean = 1.50).
3. **Catastrophic Impact if Retained**:
   - If a model is trained using `jumlah_scan`, gradient boosting trees will allocate high feature importance to it.
   - When evaluated on the test set where 98%+ of packages have only 1 or 2 scans, the model will infer that all packages are in their earliest delivery stage or have ultra-short routes, leading to massive systematic under-prediction of delivery times.

### 2.4 Engineering Decision
**Action**: **COMPLETELY EXCLUDE `jumlah_scan` from all regression models.**  
We will document the empirical ablation in our experiment log, demonstrating how retaining `jumlah_scan` produces artificially inflated cross-validation performance but disastrous out-of-time generalizability.

---

## 3. High-Risk Operational Policy Feature in Task 2: `flag_review_manual`

### 3.1 What the Feature Represents
`kamus_data.md` defines `flag_review_manual`:
> *"1 jika transaksi pernah ditinjau manual oleh tim risiko"* (1 if the transaction was ever reviewed manually by the risk team).

### 3.2 Empirical Data Audit
- In `klasifikasi_train.csv.gz`:
  - `flag_review_manual == 1`: 6,898 transactions (1.71%).
  - `flag_review_manual == 0`: 396,179 transactions (98.29%).
  - **Crosstab against `is_fraud`**:
    - When `flag_review_manual == 1`: 2,095 are fraud (30.37% fraud rate).
    - When `flag_review_manual == 0`: 527 are fraud (0.13% fraud rate).
    - Almost 80% (2,095 / 2,622) of all confirmed frauds in the training set passed through manual review!
- In `klasifikasi_test.csv.gz`:
  - `flag_review_manual == 1`: 1,312 transactions (1.23%).
  - `flag_review_manual == 0`: 105,362 transactions (98.77%).

### 3.3 Leakage & Operational Dilemma
1. **Does it represent an upstream rule or a post-facto review?**
   Because `flag_review_manual` is provided in `klasifikasi_test.csv.gz`, it represents the output of a legacy heuristic triage rule (e.g. legacy rule-based system flagging suspicious transactions for human review).
2. **The Optimization Goal**:
   Section 6 and Section 9 instruct our team to build an autonomous ML model that outputs:
   - `prob`: Calibrated fraud probability.
   - `tandai`: A binary review decision based on the business cost objective:
     $$\text{tandai} = 1 \iff \text{prob} \times \text{nominal} > \text{Rp } 25,000$$
3. **Risk of Feature Dependency**:
   - If our model relies exclusively on `flag_review_manual`, it will simply learn to copy the legacy rule engine rather than discovering novel, subtle fraud vectors from transaction velocity, device binding, and behavioral patterns.
   - Furthermore, missed fraud under the legacy rule (`flag_review_manual == 0`, containing 527 fraud cases totaling tens of millions in nominal losses) would be systematically ignored by the model.

### 3.4 Engineering Decision
**Action**:
1. Treat `flag_review_manual` as an available operational context signal, but evaluate model performance both **with and without** `flag_review_manual`.
2. Ensure the primary feature space is dominated by objective behavioral signals (`v01-v50`, velocity, device history, IP anomaly, nominal ratio).
3. The final decision `tandai` in `prediksi_klasifikasi.csv` must strictly optimize expected financial cost ($P(\text{fraud}) \times \text{nominal} > 25,000$), completely independent of whether the legacy team manually flagged it.

---

## 4. Entity Aggregation & Historical Feature Leakage

### 4.1 Strict Temporal Precedence Rule
In fraud detection, user and merchant behaviors evolve dynamically. When engineering historical aggregates (e.g., user transaction frequency, merchant fraud rate, past failed logins):
- **NEVER** include the current transaction in historical aggregates.
- **NEVER** use global or batch groupby aggregates across the full dataset (which leaks future transactions into past records).
- All aggregations must be point-in-time (`timestamp < current_transaction_timestamp`).

### 4.2 Merchant Fraud History
Any feature summarizing merchant fraud rate must **only** count fraud cases that were confirmed *prior* to the transaction cutoff. Using contemporaneous or future fraud labels is a severe violation of causal time ordering.

---

## 5. Anonymized Features Audit (`f01-f50`, `v01-v50`, `n01-n10`)

- **Regression (`f01-f50`)**:
  - `kamus_data.md` explicitly states: *"dihitung saat paket diambil"* (calculated at package pickup).
  - All 50 features are non-null continuous floats with valid variances.
  - Zero leakage detected; fully valid for training and inference.
- **Classification (`v01-v50`)**:
  - Generated by internal real-time risk scoring engine at transaction time.
  - All 50 features have zero missingness and valid distributions.
  - Zero leakage detected; fully valid for inference.
- **Clustering (`n01-n10`)**:
  - CRM static customer profile attributes. Zero missingness, valid distributions.

---

## 6. Summary of Feature Actions

| Feature | Task | Status | Prescribed Action |
| :--- | :--- | :---: | :--- |
| `jumlah_scan` | Regression | **FATAL LEAKAGE** | **EXCLUDE completely from feature space.** |
| `id_paket` | Regression | Metadata ID | Drop before model training. |
| `berat_kg` | Regression | 5.5% Missing | Retain; impute via median grouped by volume/service. |
| `mitra_kurir` | Regression | 1 Unseen Category | Retain; encode unseen 'K13' as fallback/unknown. |
| `flag_review_manual` | Classification | Operational Heuristic | Benchmark with/without; prioritize independent behavioral features. |
| `id_transaksi` | Classification | Metadata ID | Drop before model training. |
| `menit_sejak_trx_terakhir` | Classification | 25.5% Missing (Structural) | Impute with -1 / 99,999; add `is_first_trx` boolean flag. |
| `rasio_nominal_rata2` | Classification | 25.5% Missing (Structural) | Impute with 1.0; add `is_first_trx` boolean flag. |
| `rata2_keranjang` | Clustering | 5.6% Missing | Reconstruct via `belanja_90h / frekuensi_90h`. |
