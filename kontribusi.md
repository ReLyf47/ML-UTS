# Lembar Kontribusi & Deklarasi AI (Contribution Sheet & AI Disclosure)

**Judul Proyek**: Hands-on Machine Learning: Regression, Classification, and Clustering  
**Mata Kuliah**: Machine Learning (UTS) — Tahun Akademik 2025/2026  
**Kelompok**: ML-UTS (3 orang)  
**Repositori**: https://github.com/ReLyf47/ML-UTS  

---

## 1. Pembagian Tugas Anggota Kelompok (Team Member Contributions)

| No | Nama Anggota | NIM | Peran Utama | Rincian Kontribusi | Bukti di GitHub |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | **ILMA JULIANTI** | 101032330052 | Data Engineer / Regression Lead | Audit kualitas data & kebocoran (`jumlah_scan`), perancangan validasi temporal Tugas 1, *feature pipeline* regresi (target-encoding `kota_tujuan`, penanganan kategori unseen `K13`), eksperimen model kuantil P50/P90 dan kalibrasi konformal. | Commit [`ede9645`](https://github.com/ReLyf47/ML-UTS/commit/ede9645), [`2a04854`](https://github.com/ReLyf47/ML-UTS/commit/2a04854), [`3f49e69`](https://github.com/ReLyf47/ML-UTS/commit/3f49e69) |
| 2 | **ALIF AKBAR AR RAIYAN** | 101032330121 | ML Engineer / Classification Lead | Pipeline deteksi fraud end-to-end, penanganan ketidakseimbangan kelas (class weight, *under-sampling*, *threshold tuning*), kalibrasi probabilitas Isotonik, optimasi aturan biaya finansial, runner satu-perintah `kode/run_all.py`, dan README repositori. | Commit [`638257f`](https://github.com/ReLyf47/ML-UTS/commit/638257f), [`c271db1`](https://github.com/ReLyf47/ML-UTS/commit/c271db1), [`870dc18`](https://github.com/ReLyf47/ML-UTS/commit/870dc18) |
| 3 | **PRICILIA APRIANA** | 101032300056 | Analyst / Clustering & Reporting Lead | Segmentasi pelanggan (benchmark K-Means, GMM, Agglomerative, HDBSCAN), uji stabilitas *bootstrap* ARI, isolasi anomali `-1`, *profiling* persona & rekomendasi pemasaran, penyusunan `laporan.pdf`, serta dokumen reprodusibilitas. | Commit [`d10173d`](https://github.com/ReLyf47/ML-UTS/commit/d10173d), [`c8f0f96`](https://github.com/ReLyf47/ML-UTS/commit/c8f0f96), [`d4b0dd2`](https://github.com/ReLyf47/ML-UTS/commit/d4b0dd2) |

**Alur kerja kolaborasi:** setiap anggota bekerja pada *fork* repositori kelompok di akun GitHub masing-masing, membuka *pull request* (PR) ke branch `main`, dan semua PR di-review silang oleh minimal satu anggota lain sebelum di-merge. *Main branch* pada deadline adalah sumber penilaian.

### 1.1 Commit Tambahan (Refactor / Optimasi Pasca-Rilis)

Setelah implementasi awal, masing-masing anggota melanjutkan dengan perbaikan kualitas kode agar riwayat kontribusi tercatat lengkap di GitHub:

| Anggota | Branch | Judul PR | Lingkup |
| :--- | :--- | :--- | :--- |
| ILMA JULIANTI | `refactor/classification-pipeline` | `refactor(classification): deduplicate experiment logging, add type hints` | Merapikan blok eksperimen yang berulang di `kode/classification/train_and_evaluate.py` tanpa mengubah logika/angka; menambahkan *type hint*. |
| PRICILIA APRIANA | `perf/clustering-stability` | `perf(clustering): vectorize stability ARI + input guards` | Mengoptimalkan perhitungan ARI pasangan di `kode/clustering/models.py` dan menambahkan validasi input; hasil ARI identik. |

*(Tautan PR dilampirkan setelah PR di-merge ke `main`.)*

---

## 2. Deklarasi Penggunaan Kecerdasan Buatan (AI Assistance Disclosure)

Sesuai prinsip integritas akademik dan transparansi profesional, kelompok kami secara jujur mendokumentasikan pemanfaatan asisten kecerdasan buatan (AI) dalam pengerjaan proyek ini:

### 2.1 Alat AI yang Digunakan
- **Antigravity AI Coding Assistant** — eksplorasi data, penyusunan *boilerplate* kode, dan draf awal laporan.
- **Hermes Agent (Nous Research)** — audit reproduksibilitas, perbaikan *path* portable, penambahan analisis yang hilang (partial dependence, *resampling*, benchmark Agglomerative/HDBSCAN, bukti seleksi fitur), serta penyelarasan angka laporan dengan metrik pipeline.

### 2.2 Lingkup dan Tugas yang Dibantu AI
1. **Eksplorasi dan Audit Data**:
   - Membantu inspeksi skema 5 berkas tabular mentah (~940.000 baris) dan `kamus_data.md`.
   - Mendeteksi anomali fatal disparitas distribusi `jumlah_scan` antara data latih (rata-rata 9.65 scan) dan data uji (rata-rata 1.50 scan) sebagai *post-outcome target leakage*.
2. **Penyusunan Boilerplate & Modularitas Kode**:
   - *Scaffolding* modul-modul modular Python di bawah `kode/common/`, `kode/regression/`, `kode/classification/`, dan `kode/clustering/`.
   - Skrip visualisasi Matplotlib/Seaborn untuk kurva kalibrasi, batas kuantil, elbow/silhouette, PDP, dan *heatmap* persona.
3. **Dokumentasi & Script Validasi**:
   - Penyusunan draf laporan teknis, kamus data, dan script pengujian kepatuhan berkas prediksi (`prediksi_regresi.csv`, `prediksi_klasifikasi.csv`, `cluster.csv`).

### 2.3 Metode Verifikasi dan Validasi Manusia
Setiap kode, logika analitis, dan angka eksperimental yang dihasilkan telah melalui tahapan verifikasi ketat:
- **Reproduksi Satu Perintah**: seluruh pipeline dijalankan ulang dari nol dengan `python kode/run_all.py` pada environment bersih (Python 3.14, dependensi ter-*pin* di `requirements.txt`, seed 42) dan seluruh berkas prediksi tervalidasi oleh `kode/common/validate_submissions.py`.
- **Audit Tanpa Kebocoran (Zero Leakage Audit)**: memastikan fitur `jumlah_scan` dibuang dari data latih regresi dan tidak ada data uji tersembunyi yang digunakan dalam rekayasa fitur maupun pelatihan.
- **Kepatuhan Batasan Matematis**: memverifikasi secara independen bahwa $P_{90} \ge P_{50}$ terpenuhi 100%, probabilitas klasifikasi berada pada rentang $[0, 1]$, dan anomali klasterisasi dilabeli `-1`.
- **Penyelarasan Angka**: setiap angka yang tampil di laporan dibandingkan silang dengan keluaran `outputs/metrics/*.json` dari pipeline yang sama.

---

## 3. Persetujuan Anggota (Member Approval)

Lembar kontribusi ini disusun bersama dan **disetujui oleh seluruh anggota kelompok**:

| No | Nama | NIM | Persetujuan |
| :-: | :--- | :--- | :---: |
| 1 | ILMA JULIANTI | 101032330052 | ☑ |
| 2 | ALIF AKBAR AR RAIYAN | 101032330121 | ☑ |
| 3 | PRICILIA APRIANA | 101032300056 | ☑ |

*Tanggal persetujuan: 16 November 2026*
