# Sistem Rekayasa Data & Pembelajaran Mesin Otonom

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gaya Kode: Bersih](https://img.shields.io/badge/Code%20Style-Production%20Grade-brightgreen.svg)]()
[![Validasi Data: 100% Lulus](https://img.shields.io/badge/Data%20Validation-100%25%20Passed-brightgreen.svg)]()

Repositori ini berisi pipeline rekayasa data otonom lengkap, model pembelajaran mesin, dan prediksi tervalidasi untuk proyek **Pembelajaran Mesin UTS**: **"Pembelajaran Mesin Praktis: Regresi, Klasifikasi, dan Klasterisasi"**.

Proyek ini menerapkan pipeline pemrosesan data dan pelatihan model yang ketat serta bebas kebocoran data pada sekitar **940.000 baris data tabular** di bidang logistik Indonesia, risiko penipuan fintech, dan perilaku pelanggan e-commerce.

---

## 📌 Ringkasan Proyek & Sorotan Utama

1. **Tugas 1: Estimasi Waktu Pengiriman (Regresi Kuantil)**
   - **Target**: Durasi pengiriman (`durasi_jam`) untuk memprediksi median ($P_{50}$) dan batas atas ($P_{90}$).
   - **Pencegahan Kebocoran Data**: Menemukan dan menghapus kebocoran fatal dari pemindaian pelacakan pasca-hasil (`jumlah_scan`), yang berbeda 6,4x antara data train dan test.
   - **Model**: `HistGradientBoostingRegressor` dengan loss kuantil pinball, penegakan non-crossing ($P_{90} \ge P_{50}$ dijamin), dan kalibrasi split conformal untuk memastikan cakupan empiris $\ge 90,0\%$.
   - **Hasil**: Pinball Loss $P_{50}$ = **10,52**, Pinball Loss $P_{90}$ = **11,15**, Cakupan Empiris = **90,00%** (dibandingkan loss baseline Median Global = 17,15 / 15,96).

2. **Tugas 2: Deteksi Penipuan Finansial (Klasifikasi Berbasis Optimasi Biaya)**
   - **Target**: Penipuan terkonfirmasi (`is_fraud`, ketidakseimbangan kelas ekstrem sebesar 0,65%).
   - **Fungsi Biaya**: $\text{Biaya} = \sum_{\text{penipuan yang terlewat}} \text{nominal} + 25.000 \times N_{\text{ditandai}}$.
   - **Model**: `HistGradientBoostingClassifier` seimbang dengan kalibrasi probabilitas isotonic out-of-time.
   - **Optimasi Keputusan**: Aturan keputusan expected loss ($P(\text{penipuan}) \times \text{nominal} > 25.000$), menghemat lebih dari **Rp 441.000.000** dibandingkan transaksi yang tidak ditandai.
   - **Hasil**: PR-AUC = **0,4932** (0,7439 dengan flag triase operasional lama), Brier Score = **0,0033**, Total Biaya = **Rp 156.366.200** (dibandingkan biaya tanpa penandaan sebesar Rp 597.604.200).

3. **Tugas 3: Segmentasi Pelanggan (Klasterisasi Perilaku Tanpa Supervisi)**
   - **Target**: Menemukan segmen pelanggan alami dan mengisolasi anomali pada 120.000 akun.
   - **Metodologi**: Transformasi logaritmik RFM, pemeliharaan komposisi kategori, dan penskalaan fitur yang tangguh.
   - **Perbandingan Algoritma**: K-Means ($k=3\dots8$), Gaussian Mixture Models (BIC/AIC), dan estimasi kepadatan HDBSCAN.
   - **Hasil**: $k=5$ optimal dengan indeks Davies-Bouldin minimum (**2,9177**), stabilitas bootstrap tinggi (ARI Rata-rata = **0,8711**), dan 1.800 anomali multivariat (1,50%) yang diisolasi dengan label `-1`.

---

## 🗂 Arsitektur Proyek

```text
ML-UTS/
├── data/
│   ├── raw/                       # Dataset mentah asli & kamus_data.md (Hanya Baca)
│   │   ├── clustering.csv
│   │   ├── klasifikasi_train.csv
│   │   ├── klasifikasi_test.csv
│   │   ├── regresi_train.csv
│   │   ├── regresi_test.csv
│   │   └── kamus_data.md
│   │
│   ├── interim/                   # Berkas parquet interim yang diekstrak & dinormalisasi
│   │   ├── extracted/
│   │   └── normalized/
│   │
│   ├── processed/                 # Artefak parquet yang dipartisi & divalidasi
│   │   ├── common/                # Berkas parquet bersih bersama
│   │   ├── regression/            # train.parquet, validation.parquet, test.parquet
│   │   ├── classification/        # train.parquet, validation.parquet, test.parquet
│   │   └── clustering/            # customer_features.parquet, customer_features_scaled.parquet, customer_ids.parquet
│   │
│   └── manifests/                 # Manifest audit JSON otomatis
│       ├── source_manifest.json
│       ├── processing_manifest.json
│       └── dataset_schemas.json
│
├── kode/
│   ├── common/                    # Utilitas modular bersama
│   │   ├── config.py
│   │   ├── io.py
│   │   ├── validation.py
│   │   ├── preprocessing.py
│   │   ├── logging_config.py
│   │   ├── seeds.py
│   │   ├── metrics.py
│   │   └── validate_submissions.py
│   │
│   ├── data_engineering/          # Pipeline Rekayasa Data Modular
│   │   ├── 01_inspect_data.py
│   │   ├── 02_extract_data.py
│   │   ├── 03_clean_common.py
│   │   ├── 04_build_regression_data.py
│   │   ├── 05_build_classification_data.py
│   │   ├── 06_build_clustering_data.py
│   │   └── 07_validate_processed_data.py
│   │
│   ├── regression/                # Pemodelan & kalibrasi regresi kuantil
│   ├── classification/            # Klasifikasi penipuan seimbang & threshold biaya
│   ├── clustering/                # Segmentasi pelanggan & isolasi anomali
│   └── run_all.py                 # Orkestrator utama end-to-end
│
├── reports/
│   ├── data_quality_report.md     # Missingness, distribusi & kardinalitas terperinci
│   ├── preprocessing_report.md    # Alasan preprocessing dan log transformasi data
│   ├── leakage_audit.md           # Penilaian risiko kebocoran & pertahanan batas fitur
│   └── figures/                   # 9 grafik diagnostik berkualitas publikasi
│
├── outputs/
│   ├── predictions/               # Berkas CSV submission akhir tervalidasi
│   │   ├── prediksi_regresi.csv
│   │   ├── prediksi_klasifikasi.csv
│   │   └── cluster.csv
│   ├── models/                    # Artefak serialisasi model
│   └── metrics/                   # Ringkasan evaluasi JSON
│
├── logs/                          # Log eksekusi untuk setiap langkah rekayasa data
├── requirements.txt               # Dependensi yang dibekukan
└── README.md                      # Instruksi eksekusi pipeline
```

---

## 🚀 Quickstart & Reproduksi Satu Perintah

### 1. Pengaturan Lingkungan

> ⚠️ **Peringatan:** Pastikan **Git LFS** sudah terinstal sebelum melakukan clone repositori ini karena beberapa dataset berukuran besar dikelola menggunakan Git LFS. Setelah Git LFS terinstal, jalankan `git lfs install`.

```bash
# Clone repositori
git clone https://github.com/ReLyf47/ML-UTS.git
cd ML-UTS

# Buat dan aktifkan virtual environment Python
python3 -m venv .venv
source .venv/bin/activate

# Instal dependensi yang diperlukan
pip install -r requirements.txt
```

### 2. Menjalankan Pipeline End-to-End Lengkap

Untuk menjalankan pipeline rekayasa data lengkap, memvalidasi semua dataset yang telah diproses, melatih semua model, membuat grafik diagnostik, dan menghasilkan berkas CSV submission yang tervalidasi:

```bash
python kode/run_all.py
```

### 3. Menjalankan Pipeline Rekayasa Data Langkah demi Langkah

Jika ingin menjalankan setiap langkah rekayasa data secara terpisah:

```bash
# 1. Memeriksa data mentah dan membuat manifest sumber & skema
python kode/data_engineering/01_inspect_data.py

# 2. Mengekstrak dan membuat dataset interim parquet yang dinormalisasi
python kode/data_engineering/02_extract_data.py

# 3. Membersihkan dataset umum dan menangani null struktural
python kode/data_engineering/03_clean_common.py

# 4. Membuat dataset regresi yang telah diproses (menghapus kebocoran jumlah_scan)
python kode/data_engineering/04_build_regression_data.py

# 5. Membuat dataset klasifikasi yang telah diproses (pemisahan temporal)
python kode/data_engineering/05_build_classification_data.py

# 6. Membuat dataset klasterisasi yang telah diproses (log1p & scaling)
python kode/data_engineering/06_build_clustering_data.py

# 7. Memvalidasi semua artefak parquet yang telah diproses dan membuat manifest pemrosesan
python kode/data_engineering/07_validate_processed_data.py
```

### 4. Memvalidasi Berkas Submission Akhir

Untuk memastikan ketiga berkas submission mematuhi skema submission, batasan non-crossing, dan urutan ID secara ketat:

```bash
python kode/common/validate_submissions.py
```
