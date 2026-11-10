# Autonomous Data Engineering & Machine Learning System

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean](https://img.shields.io/badge/Code%20Style-Production%20Grade-brightgreen.svg)]()
[![Data Validation: 100% Passed](https://img.shields.io/badge/Data%20Validation-100%25%20Passed-brightgreen.svg)]()

This repository contains the complete, autonomous data engineering pipeline, machine learning models, and validated submission predictions for the **UTS Machine Learning** project: **"Hands-on Machine Learning: Regression, Classification, and Clustering"**.

The project implements a strict, leakage-free data processing and model training pipeline across approximately **940,000 tabular records** in Indonesian logistics, fintech fraud risk, and e-commerce customer behavior.

---

## 📌 Project Overview & Key Highlights

1. **Task 1: Delivery Time Estimation (Quantile Regression)**
   - **Target**: Delivery duration (`durasi_jam`) predicting median ($P_{50}$) and upper bound ($P_{90}$).
   - **Leakage Prevention**: Discovered and eliminated fatal post-outcome tracking scan leakage (`jumlah_scan`, which diverged 6.4x between train and test).
   - **Model**: `HistGradientBoostingRegressor` with pinball quantile loss, non-crossing enforcement ($P_{90} \ge P_{50}$ guaranteed), and split conformal calibration ensuring $\ge 90.0\%$ empirical coverage.
   - **Results**: $P_{50}$ Pinball Loss = **10.52**, $P_{90}$ Pinball Loss = **11.15**, Empirical Coverage = **90.00%** (vs Global Median baseline loss = 17.15 / 15.96).

2. **Task 2: Financial Fraud Detection (Cost-Optimized Classification)**
   - **Target**: Confirmed fraud (`is_fraud`, 0.65% extreme class imbalance).
   - **Cost Function**: $\text{Cost} = \sum_{\text{missed fraud}} \text{nominal} + 25,000 \times N_{\text{flagged}}$.
   - **Model**: Balanced `HistGradientBoostingClassifier` with out-of-time Isotonic Probability Calibration.
   - **Decision Optimization**: Expected loss decision rule ($P(\text{fraud}) \times \text{nominal} > 25,000$), saving over **Rp 441,000,000** compared to unflagged transactions.
   - **Results**: PR-AUC = **0.4932** (0.7439 with legacy operational triage flag), Brier Score = **0.0033**, Total Cost = **Rp 156,366,200** (vs Flag Nothing cost of Rp 597,604,200).

3. **Task 3: Customer Segmentation (Unsupervised Behavioral Clustering)**
   - **Target**: Discover natural customer segments and isolate anomalies among 120,000 accounts.
   - **Methodology**: RFM logarithmic transformation, compositional category preservation, and robust feature scaling.
   - **Algorithm Comparison**: K-Means ($k=3\dots8$), Gaussian Mixture Models (BIC/AIC), and HDBSCAN density estimation.
   - **Results**: Optimal $k=5$ with minimum Davies-Bouldin index (**2.9177**), high bootstrap stability (Mean ARI = **0.8711**), and 1,800 multivariate anomalies (1.50%) isolated with label `-1`.

---

## 🗂 Project Architecture

```text
ML-UTS/
├── data/
│   ├── raw/                       # Original raw datasets & kamus_data.md (Read-Only)
│   │   ├── clustering.csv
│   │   ├── klasifikasi_train.csv
│   │   ├── klasifikasi_test.csv
│   │   ├── regresi_train.csv
│   │   ├── regresi_test.csv
│   │   └── kamus_data.md
│   │
│   ├── interim/                   # Interim extracted & normalized parquet files
│   │   ├── extracted/
│   │   └── normalized/
│   │
│   ├── processed/                 # Partitioned and validated parquet artifacts
│   │   ├── common/                # Shared cleaned parquet files
│   │   ├── regression/            # train.parquet, validation.parquet, test.parquet
│   │   ├── classification/        # train.parquet, validation.parquet, test.parquet
│   │   └── clustering/            # customer_features.parquet, customer_features_scaled.parquet, customer_ids.parquet
│   │
│   └── manifests/                 # Automated JSON audit manifests
│       ├── source_manifest.json
│       ├── processing_manifest.json
│       └── dataset_schemas.json
│
├── kode/
│   ├── common/                    # Shared modular utilities
│   │   ├── config.py
│   │   ├── io.py
│   │   ├── validation.py
│   │   ├── preprocessing.py
│   │   ├── logging_config.py
│   │   ├── seeds.py
│   │   ├── metrics.py
│   │   └── validate_submissions.py
│   │
│   ├── data_engineering/          # Modular Data Engineering Pipeline
│   │   ├── 01_inspect_data.py
│   │   ├── 02_extract_data.py
│   │   ├── 03_clean_common.py
│   │   ├── 04_build_regression_data.py
│   │   ├── 05_build_classification_data.py
│   │   ├── 06_build_clustering_data.py
│   │   └── 07_validate_processed_data.py
│   │
│   ├── regression/                # Quantile regression modeling & calibration
│   ├── classification/            # Balanced fraud classification & cost thresholding
│   ├── clustering/                # Customer segmentation & anomaly isolation
│   └── run_all.py                 # End-to-end master orchestrator
│
├── reports/
│   ├── data_quality_report.md     # Granular missingness, distributions & cardinality
│   ├── preprocessing_report.md    # Preprocessing rationale and data transformation log
│   ├── leakage_audit.md           # Leakage risk assessment & feature cutoff defense
│   └── figures/                   # 9 Publication-grade diagnostic charts
│
├── outputs/
│   ├── predictions/               # Validated final submission CSV files
│   │   ├── prediksi_regresi.csv
│   │   ├── prediksi_klasifikasi.csv
│   │   └── cluster.csv
│   ├── models/                    # Model serialization artifacts
│   └── metrics/                   # JSON evaluation summaries
│
├── logs/                          # Execution logs for each data engineering step
├── requirements.txt               # Frozen dependencies
└── README.md                      # Pipeline execution instructions
```

---

## 🚀 Quickstart & One-Command Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ReLyf47/ML-UTS.git
cd ML-UTS

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the Complete End-to-End Pipeline
To run the complete data engineering pipeline, validate all processed datasets, train all models, generate diagnostic figures, and output validated submission CSV files:

```bash
python kode/run_all.py
```

### 3. Step-by-Step Data Engineering Pipeline Execution
If you wish to run each data engineering step independently:

```bash
# 1. Inspect raw data and build source manifest & schemas
python kode/data_engineering/01_inspect_data.py

# 2. Extract and create normalized interim parquet datasets
python kode/data_engineering/02_extract_data.py

# 3. Clean common datasets and handle structural nulls
python kode/data_engineering/03_clean_common.py

# 4. Build regression processed datasets (drops jumlah_scan leakage)
python kode/data_engineering/04_build_regression_data.py

# 5. Build classification processed datasets (temporal split)
python kode/data_engineering/05_build_classification_data.py

# 6. Build clustering processed datasets (log1p & scaling)
python kode/data_engineering/06_build_clustering_data.py

# 7. Validate all processed parquet artifacts and generate processing manifest
python kode/data_engineering/07_validate_processed_data.py
```

### 4. Validate Final Submission Files
To verify that all three submission files adhere strictly to submission schemas, non-crossing constraints, and ID ordering:

```bash
python kode/common/validate_submissions.py
```
