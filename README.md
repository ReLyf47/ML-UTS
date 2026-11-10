# Hands-on Machine Learning: Regression, Classification, and Clustering

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean](https://img.shields.io/badge/Code%20Style-Production%20Grade-brightgreen.svg)]()

This repository contains the complete, reproducible, and leakage-free implementation for the **UTS Machine Learning** project: **"Hands-on Machine Learning: Regression, Classification, and Clustering"**.

The project implements three classical machine learning tasks on approximately **940,000 tabular records** across Indonesian logistics, fintech risk, and e-commerce customer behavior.

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
│   ├── raw/                       # Original gzipped datasets & kamus_data.md (Read-Only)
│   └── processed/                 # Cached feature matrices
├── docs/
│   ├── dataset_inventory.md       # Complete inventory table, memory, and shapes
│   ├── data_dictionary_analysis.md# Column-by-column interpretation & semantics
│   ├── data_quality_report.md     # Missingness, outliers, and distribution check
│   ├── leakage_audit.md           # Leakage audit & feature risk assessment
│   └── final_quality_assurance.md # 12-point QA compliance report
├── experiments/
│   ├── experiments.csv            # Structured audit log of all experimental runs
│   └── experiments.jsonl          # JSON Lines experiment tracking
├── kode/
│   ├── common/                    # Shared loaders, metrics, logging, seed controllers
│   ├── regression/                # Task 1 quantile regression pipeline
│   ├── classification/            # Task 2 cost-optimal classification pipeline
│   ├── clustering/                # Task 3 RFM customer segmentation pipeline
│   └── run_all.py                 # One-command full pipeline execution
├── laporan/
│   ├── figures/                   # 9 Publication-grade diagnostic charts
│   ├── laporan.md                 # Markdown source of the academic report
│   └── laporan.pdf                # Compiled final academic report (<= 18 pages)
├── outputs/
│   ├── metrics/                   # Serialized JSON evaluation logs
│   ├── models/                    # Model serialization artifacts
│   └── figures/                   # Generated evaluation plots
├── prediksi_regresi.csv           # Final regression submission (60,000 rows)
├── prediksi_klasifikasi.csv       # Final classification submission (106,674 rows)
├── cluster.csv                    # Final clustering submission (120,000 rows)
├── kontribusi.md                  # Honest team contribution sheet & AI disclosure
├── requirements.txt               # Exact frozen dependencies
└── README.md                      # Pipeline execution instructions
```

---

## 🚀 Quickstart & One-Command Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository_url>
cd ML-UTS

# Create and activate Python virtual environment (Python 3.10+ supported, developed on 3.14.8)
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the Complete End-to-End Pipeline
To reproduce all data audits, train all baseline and advanced models, generate 9 diagnostic figures, output all 3 prediction files, and run the automated compliance auditor, run:

```bash
python kode/run_all.py
```

*Expected runtime: ~3.2 minutes on modern x86_64 CPU.*

### 3. Run Individual Task Pipelines
Each task pipeline can also be executed independently:

```bash
# Run Task 1 (Delivery Duration Quantile Regression)
python kode/regression/train_and_evaluate.py

# Run Task 2 (Fraud Detection Classification)
python kode/classification/train_and_evaluate.py

# Run Task 3 (Customer Segmentation Clustering)
python kode/clustering/train_and_evaluate.py

# Run Submission Integrity Audit
python kode/common/validate_submissions.py
```

---

## 📊 Summary of Experimental Results

### Task 1: Regression (Holdout August 2025)
| Model | Features | P50 Pinball Loss | P90 Pinball Loss | P90 Coverage | Non-crossing Violations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Global Median Baseline | None | 17.15 | 15.96 | 89.92% | 0 |
| Service Median Baseline | `layanan` | 15.86 | 15.10 | 89.80% | 0 |
| Log-Ridge Quantile | 84 Clean Features | 13.88 | 12.85 | 90.26% | 0 |
| **HistGBM Calibrated (Ours)** | **84 Clean Features** | **10.52** | **11.15** | **90.00%** | **0** |
| *HistGBM (Leakage Ablation)* | *With `jumlah_scan`* | *9.18* | *9.80* | *89.46%* | *0* |

### Task 2: Classification (Holdout July–August 2025)
| Model | Strategy | PR-AUC | ROC-AUC | Brier Score | Total Financial Cost |
| :--- | :--- | :---: | :---: | :---: | :---: |
| *Baseline: Flag Nothing* | No Review | - | - | - | Rp 597,604,200 |
| *Baseline: Flag Everything*| Review All | - | - | - | Rp 2,630,950,000 |
| Logistic Regression | Balanced Weights | 0.4004 | 0.9332 | 0.0541 | Rp 1,044,486,100 |
| HistGBM (Uncalibrated) | Raw Balanced | 0.4888 | 0.9612 | 0.0196 | Rp 388,757,700 |
| **HistGBM Isotonic (Ours)**| **Cost Rule ($P \times \text{nom} > 25\text{k}$)** | **0.4932** | **0.9628** | **0.0033** | **Rp 156,366,200** |
| *HistGBM (Policy Ablation)*| *With `flag_review_manual`* | *0.7439* | *0.9785* | *0.0028* | *Rp 68,946,700* |

### Task 3: Clustering (Full N=120,000 Customers)
| Cluster ID | Persona Name | Customer Count (%) | Average 90d Spend (IDR) | Avg Frequency | Dominant Category | Actionable Strategy |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | VIP Champions | 29,405 (24.50%) | Rp 12,450,000 | 18.4 orders | Fashion / Gadget | VIP loyalty tiers & concierge service |
| **1** | Discount Hunters | 1,732 (1.44%) | Rp 3,210,000 | 8.2 orders | Electronics | Flash sale alerts & coupon thresholds |
| **2** | Daily Impulse Shoppers | 39,102 (32.59%) | Rp 4,890,000 | 14.1 orders | Food & Beverage | Gamified checkout & free shipping clubs |
| **3** | Lapsed / At-Risk Dormant | 41,414 (34.51%) | Rp 1,940,000 | 3.1 orders | Home & Living | 'We Miss You' win-back voucher campaigns |
| **4** | Selective Pragmatists | 6,547 (5.46%) | Rp 5,620,000 | 6.8 orders | Automotive / Hobbies | Personalized category replenishment |
| **-1** | **Anomalous Outliers** | **1,800 (1.50%)** | **Rp 48,900,000** | **42.5 orders** | Extreme Spikes | Security, fraud & reseller compliance audit |

---

## 🔒 Verification & Quality Assurance

All final prediction CSV files conform strictly to assignment constraints:
- `prediksi_regresi.csv`: 60,000 rows; $P_{90} \ge P_{50}$ across 100% of rows; 0 missing values.
- `prediksi_klasifikasi.csv`: 106,674 rows; $\text{prob} \in [0.0, 1.0]$; $\text{tandai} \in \{0, 1\}$; 0 missing values.
- `cluster.csv`: 120,000 rows; integer cluster labels; $-1$ represents isolated anomalies.
