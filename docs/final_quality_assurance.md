# Final Quality Assurance & Compliance Report

**Project**: Hands-on Machine Learning: Regression, Classification, and Clustering  
**Audit Date**: November 10, 2026  
**Auditor**: Autonomous Machine Learning Engineering Team  
**Status**: **ALL 12 CHECKS FULLY PASSED (100% COMPLIANT)**  

---

## 1. Compliance Checklist (Section 12 Verification)

| No | Verification Item | Status | Verification Evidence / Command Output |
| :-: | :--- | :---: | :--- |
| **1** | **All three pipelines execute successfully** | **PASSED** | Executed via `kode/run_all.py` without errors. Run time: ~3.2 minutes total. |
| **2** | **No target leakage exists** | **PASSED** | Fatal post-outcome leakage `jumlah_scan` completely excluded from feature space. All features restricted to values available at pickup/transaction cutoff. |
| **3** | **No future information enters historical features** | **PASSED** | All behavioral aggregations, rolling velocities, and point-in-time security counters evaluated using strictly preceding transactions ($t < t_{\text{current}}$). |
| **4** | **Temporal validation correctly implemented** | **PASSED** | Task 1: Train Jan–Jul 2025, Val Aug 2025. Task 2: Train Jan–Jun 2025, Val Jul–Aug 2025 (2-month holdout mirroring the 2-month hidden test window). |
| **5** | **Hidden test labels were never used** | **PASSED** | Verified: Hidden test targets (`durasi_jam`, `is_fraud`) do not exist in test files. Test data was never used for imputation, encoding, calibration, or model fitting. |
| **6** | **Calibration uses valid data** | **PASSED** | Conformal quantile adjustment (Task 1) and Isotonic/Sigmoid probability calibrators (Task 2) fitted strictly on out-of-time chronological validation slices. |
| **7** | **Prediction IDs match test IDs exactly** | **PASSED** | `id_paket`: Exactly 60,000 rows matching `regresi_test.csv.gz` order.<br>`id_transaksi`: Exactly 106,674 rows matching `klasifikasi_test.csv.gz` order.<br>`id_pelanggan`: Exactly 120,000 rows matching `clustering.csv.gz` order. |
| **8** | **All prediction constraints are satisfied** | **PASSED** | Regression: $P_{90} \ge P_{50}$ across 100% of rows (0 violations), no NaNs.<br>Classification: $\text{prob} \in [0.0, 1.0]$, $\text{tandai} \in \{0, 1\}$, no NaNs.<br>Clustering: Integer labels, anomaly label `-1` present (1,800 anomalies / 1.50%). |
| **9** | **Every reported metric can be reproduced** | **PASSED** | All metrics logged deterministically into `outputs/metrics/` and `experiments/experiments.csv` using fixed random seed (`seed=42`).<br>Clustering: 5 core clusters, stability ARI 0.8711 ± 0.1414 (mean ± standard deviation of 10 pairwise ARIs from 5 subsampling runs without replacement). |
| **10** | **README instructions work** | **PASSED** | Tested in virtual environment: `python kode/run_all.py` executes full end-to-end pipeline cleanly. |
| **11** | **All required files exist** | **PASSED** | Verified existence of `prediksi_regresi.csv`, `prediksi_klasifikasi.csv`, `cluster.csv`, `kontribusi.md`, `requirements.txt`, `README.md`, and `laporan/`. |
| **12** | **Report page constraint satisfied** | **PASSED** | Final academic report `laporan/laporan.pdf` adheres strictly to the $\le 18$ pages ceiling excluding appendices. |

---

## 2. Output File Verification Audit

Executed via `kode/common/validate_submissions.py`:

```text
=== VALIDATING PREDIKSI_REGRESI.CSV ===
PASSED! Rows: 60,000
P50 stats: min=15.82, median=63.54, max=288.29
P90 stats: min=20.93, median=83.70, max=427.59
Non-crossing violations: 0 (100% compliant)

=== VALIDATING PREDIKSI_KLASIFIKASI.CSV ===
PASSED! Rows: 106,674
Prob stats: min=0.000001, mean=0.007210, max=0.763889
Flagged (tandai=1): 1,674 (1.57%)
Constraints: Probabilities bounded in [0, 1], flags strictly in {0, 1}

=== VALIDATING CLUSTER.CSV ===
PASSED! Rows: 120,000
Cluster value counts:
  Cluster 3: 41,414 (34.51%)
  Cluster 2: 39,102 (32.59%)
  Cluster 0: 29,405 (24.50%)
  Cluster 4:  6,547 ( 5.46%)
  Cluster -1: 1,800 ( 1.50%) [Anomalies]
  Cluster 1:  1,732 ( 1.44%)
Constraints: Integer dtype, -1 anomalies present.
```

Persona labels and dominant categories from `outputs/metrics/clustering_metrics.json`:

| Label | Persona | Dominant category |
| ---: | :--- | :--- |
| 0 | VIP Champions | `makanan_minuman` |
| 1 | Hyperactive Bargain Hunters | `makanan_minuman` |
| 2 | Casual Shoppers | `makanan_minuman` |
| 3 | Discount Hunters | `fashion` |
| 4 | Lapsed / At-Risk Dormant | `makanan_minuman` |
| -1 | Anomalous Outliers / Extreme Accounts | `multivariate_extreme` |

---

## 3. Remaining Limitations & Risk Disclosures

1. **Synthetic Tabular Artifacts**:
   - The datasets are synthetic representations of Indonesian logistics and fintech ecosystems. While realistic, certain edge cases (such as the extreme 25,452-hour delivery duration) represent synthetically injected tails rather than standard physical logistics operations.
2. **Dynamic Concept Drift**:
   - In production fintech deployments, fraud vectors adapt adversarially. While our out-of-time chronological validation confirmed robust generalizability over a 2-month horizon (PR-AUC = 0.4932 without policy flag, PR-AUC = 0.7439 with policy flag), active monitoring of feature drift using Population Stability Index (PSI) is recommended for production deployment.
3. **Hardware / Runtime Considerations**:
   - The entire pipeline executes in approximately 3 minutes on modern x86_64 CPUs using scikit-learn's optimized multi-threaded C/Cython backends. No GPU acceleration or deep learning frameworks are required.
