# Comprehensive Leakage Audit & Feature Risk Assessment

**Project**: UTS Machine Learning - Autonomous Data Engineering Pipeline  
**Date**: November 10, 2026  
**Status**: Executed & Verified  

---

## 1. Executive Summary

Leakage prevention is the primary structural safeguard in modern ML systems. This audit documents all direct target leakage, indirect proxy leakage, temporal leakage, post-event signals, and identifier memorization risks identified across the assignment datasets, along with their enforced technical mitigations.

---

## 2. Leakage Categorization & Audited Decisions

### 2.1 Fatal Post-Event Target Leakage: `jumlah_scan` in Task 1

- **Feature Definition**: *"Jumlah scan pelacakan yang tercatat di sistem"* (Number of tracking scans recorded in system).
- **Prediction Cutoff**: *"Prediksi dibuat saat paket diambil kurir"* (Prediction is made at package pickup).
- **Empirical Failure Discovered**:
  - In `regresi_train`: Mean = **9.65 scans**, Median = **8.0 scans**, Max = **97.0 scans**.
  - In `regresi_test`: Mean = **1.50 scans**, Median = **1.0 scan**, Max = **7.0 scans**.
  - *Divergence Ratio*: 6.4x lower scan counts in the test partition.
- **Root Cause**:
  In training data, tracking scans were logged retrospectively at delivery completion, accumulating checkpoints across transit hubs. In test data, the snapshot was logged correctly at initial pickup.
- **Consequence If Retained**:
  A model using `jumlah_scan` would associate high scan counts with longer delivery durations. At test time, where 98%+ of packages have only 1–2 scans, the model would severely under-predict delivery times, ruining SLA compliance.
- **Enforced Mitigation**:
  **COMPLETELY EXCLUDED** from `04_build_regression_data.py`. Verified by automated assertion in `07_validate_processed_data.py`.

---

### 2.2 Operational Policy Triage Flag: `flag_review_manual` in Task 2

- **Feature Definition**: *"1 jika transaksi pernah ditinjau manual oleh tim risiko"* (1 if transaction was ever manually reviewed by risk team).
- **Audit Findings**:
  - Training: 6,898 flagged (1.71%). Among flagged, fraud rate is 30.37% (vs 0.13% when unflagged).
  - Test: 1,312 flagged (1.23%).
- **Leakage / Policy Risk**:
  - Relying exclusively on this feature turns the ML model into an emulator of the legacy heuristic engine rather than a fraud detector.
  - Furthermore, 527 confirmed fraud transactions in the training set slipped past the legacy manual review (`flag_review_manual == 0`). A model over-relying on this flag would fail to protect against these uncaught frauds.
- **Enforced Mitigation**:
  The classification pipeline derives risk signals primarily from behavioral velocity (`trx_1j`, `trx_24j`, `velocity_ratio_1j_24j`), device history (`umur_perangkat_hari`, `perangkat_baru`), IP location consistency (`negara_ip_beda`, `jarak_lokasi_km`), and 50 point-in-time scoring features (`v01-v50`).
  The decision flag `tandai` strictly optimizes the objective business cost:
  $$\text{tandai} = 1 \iff P(\text{fraud}) \times \text{nominal} > \text{Rp } 25,000$$

---

### 2.3 Identifier Memorization & Entity Leakage

- **Regression**:
  - `id_paket`: Distinct random package IDs. Excluded from feature matrix.
- **Classification**:
  - `id_transaksi`: Unique transaction primary key. Excluded from feature matrix.
  - `id_pengguna` & `id_merchant`: High-cardinality IDs. High-cardinality label memorization is prevented by relying on behavioral aggregate features (`trx_1j`, `trx_24j`, `rasio_nominal_rata2`, `refund_merchant_30h`) rather than raw entity identifiers.
- **Clustering**:
  - `id_pelanggan`: Excluded from feature matrix and isolated in `customer_ids.parquet` to prevent spurious clustering on alphanumeric hashes.

---

### 2.4 Temporal Leakage Prevention (Lookahead Bias)

- **Causal Time Ordering**:
  All transformations (imputations, categorical mappings, frequency encodings, target encodings) are fitted **strictly on the training split** (Jan 1, 2025 to Jul 31, 2025).
- **Out-of-Time Validation**:
  - Validation set: August 1, 2025 to August 31, 2025.
  - Test set: September 1, 2025 to October 31, 2025.
  - Zero random shuffling across temporal boundaries.
- **Label Leakage in Test**:
  - `durasi_jam` is absent from regression test data.
  - `is_fraud` is absent from classification test data.

---

## 3. Defense Table: Feature Availability at Prediction Cutoff

| Task | Feature Group | Availability at Cutoff | Defensible Explanation |
| :--- | :--- | :---: | :--- |
| **Regression** | `tgl_pickup`, `jam_pickup` | Available | Logged at the instant courier scans the package for pickup. |
| **Regression** | `layanan`, `mitra_kurir`, `hub_asal`, `kota_asal`, `pulau_tujuan`, `kota_tujuan` | Available | Fixed dispatch order attributes declared before courier dispatch. |
| **Regression** | `jarak_km`, `berat_kg`, `volume_liter`, `nilai_barang`, `jumlah_item` | Available | Manifest package properties measured at hub intake / merchant handoff. |
| **Regression** | `asuransi`, `cod`, `alamat_skor`, `pengalaman_kurir_bulan` | Available | System profile and courier assignment metadata at assignment time. |
| **Regression** | `rating_penjual`, `umur_akun_penjual_hari`, `volume_penjual_30h` | Available | Historical seller performance logged before package pickup. |
| **Regression** | `antrean_hub_asal`, `indeks_kemacetan`, `hujan_asal_mm`, `hujan_tujuan_mm` | Available | Ambient weather forecasts and hub queue lengths at time of dispatch. |
| **Regression** | `f01 - f50` | Available | System-computed features explicitly calculated at package pickup time. |
| **Regression** | `jumlah_scan` | **UNAVAILABLE / POST-EVENT** | **EXCLUDED**. Represents cumulative journey scans logged post-pickup. |
| **Classification** | `nominal`, `jam`, `hari_libur`, `waktu` | Available | Real-time payment gateway payload attributes. |
| **Classification** | `kategori_merchant`, `kanal`, `tipe_perangkat`, `email_domain` | Available | Request headers and merchant profile at transaction initiation. |
| **Classification** | `umur_akun_hari`, `level_kyc`, `umur_perangkat_hari`, `perangkat_baru` | Available | Point-in-time account and device registry attributes. |
| **Classification** | `trx_1j`, `trx_24j`, `trx_7h`, `nominal_24j`, `menit_sejak_trx_terakhir` | Available | Pre-transaction sliding window counts strictly preceding the event. |
| **Classification** | `v01 - v50` | Available | Real-time scoring vector produced by fraud engine before authorization. |
| **Clustering** | 90-day RFM, categories, CRM attributes | Available | Static 90-day historical snapshot for quarterly customer segmentation. |
