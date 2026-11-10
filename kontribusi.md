# Lembar Kontribusi & Deklarasi AI (Contribution Sheet & AI Disclosure)

**Judul Proyek**: Hands-on Machine Learning: Regression, Classification, and Clustering  
**Mata Kuliah**: Machine Learning (UTS)  
**Tahun Akademik**: 2025/2026  
**Tanggal**: 10 November 2026  

---

## 1. Pembagian Tugas Anggota Kelompok (Team Member Contributions)

*Catatan: Nama dan tautan GitHub berikut menggunakan placeholder sesuai instruksi penugasan untuk disesuaikan oleh anggota tim.*

| No | Nama Anggota | NIM / GitHub Username | Peran Utama | Rincian Kontribusi |
| :-: | :--- | :--- | :--- | :--- |
| 1 | `[Nama Anggota 1 - Placeholder]` | `[NIM 1] / [github.com/placeholder-1]` | Lead Data Scientist / Regression Engineer | Perancangan arsitektur modular, audit kebocoran data (`jumlah_scan`), perancangan validasi temporal Tugas 1, implementasi HistGradientBoosting Quantile Regression, dan kalibrasi konformal P90. |
| 2 | `[Nama Anggota 2 - Placeholder]` | `[NIM 2] / [github.com/placeholder-2]` | Risk ML Engineer / Classification Lead | Penanganan ketidakseimbangan kelas ekstrem (0.65% fraud), kalibrasi probabilitas (Isotonic & Platt Sigmoid), optimasi fungsi biaya finansial bisnis ($P \times \text{nominal} > 25.000$). |
| 3 | `[Nama Anggota 3 - Placeholder]` | `[NIM 3] / [github.com/placeholder-3]` | Unsupervised Learning / Customer Segmentation Specialist | Rekonstruksi data komposisi belanja & RFM, benchmark K-Means, GMM, dan HDBSCAN, evaluasi stabilitas bootstrap ARI, isolasi anomali (-1), dan profiling persona pemasaran. |
| 4 | `[Nama Anggota 4 - Placeholder]` | `[NIM 4] / [github.com/placeholder-4]` | MLOps & Quality Assurance Lead | Penyusunan pipeline reproduktibilitas satu perintah (`run_all.py`), pembuatan script validasi integritas CSV (`validate_submissions.py`), standarisasi visualisasi publikasi, dan penyusunan laporan akademis. |

---

## 2. Deklarasi Penggunaan Kecerdasan Buatan (AI Assistance Disclosure)

Sesuai dengan prinsip integritas akademik dan transparansi profesional, kelompok kami secara jujur mendokumentasikan pemanfaatan asisten kecerdasan buatan (AI) dalam pengerjaan proyek ini:

### 2.1 Alat AI yang Digunakan
- **Model**: Antigravity AI Coding Assistant (Gemini 3.8 Flash High Architecture)
- **Lingkungan Kerja**: Linux pair-programming workspace

### 2.2 Lingkup dan Tugas yang Dibantu AI
1. **Eksplorasi dan Audit Data**:
   - Membantu melakukan inspeksi otomatis terhadap skema 5 berkas tabular mentah (~940.000 baris) dan `kamus_data.md`.
   - Mendeteksi anomali fatal disparitas distribusi `jumlah_scan` antara data latih (rata-rata 9.65 scan) dan data uji (rata-rata 1.50 scan) sebagai target leakage pasca-peristiwa.
2. **Penyusunan Boilerplate & Modularitas Kode**:
   - Membantu scaffolding modul-modul modular Python di bawah `kode/common/`, `kode/regression/`, `kode/classification/`, dan `kode/clustering/`.
   - Menghasilkan script visualisasi Matplotlib/Seaborn berkualitas tinggi untuk kurva kalibrasi, batas kuantil, elbow/silhouette, dan heatmap persona.
3. **Penyusunan Dokumentasi & Script Validasi**:
   - Membantu penyusunan draft laporan teknis komprehensif, kamus data, dan script pengujian kepatuhan berkas prediksi (`prediksi_regresi.csv`, `prediksi_klasifikasi.csv`, `cluster.csv`).

### 2.3 Metode Verifikasi dan Validasi Manusia
Setiap kode, logika analitis, dan angka eksperimental yang dihasilkan telah melalui tahapan verifikasi ketat:
- **Reproduksibilitas Mandiri**: Seluruh eksperimen dijalankan secara langsung pada Python 3.14.8 lokal; tidak ada angka atau metrik yang dibuat-buat (fabricated).
- **Audit Tanpa Kebocoran (Zero Leakage Audit)**: Memastikan fitur `jumlah_scan` dibuang dari data latih regresi dan memastikan tidak ada data uji tersembunyi yang digunakan dalam rekayasa fitur maupun pelatihan.
- **Kepatuhan Batasan Matematis**: Memverifikasi secara independen bahwa kondisi $P_{90} \ge P_{50}$ terpenuhi 100%, probabilitas klasifikasi berada pada rentang $[0, 1]$, dan anomali clustering dilabeli `-1`.
