# Lembar Kontribusi & Deklarasi AI (Contribution Sheet & AI Disclosure)

**Judul Proyek**: Hands-on Machine Learning: Regression, Classification, and Clustering  
**Mata Kuliah**: Machine Learning (UTS), Tahun Akademik 2025/2026  
**Kelompok**: ML-UTS (3 orang)  
**Repositori**: https://github.com/ReLyf47/ML-UTS  

---

## 1. Kontribusi Anggota (Team Member Contributions)

Semua bukti di bawah dapat dicek pada riwayat commit dan pull request repositori.

| No | Nama Anggota | NIM | Peran | Rincian Kontribusi | Bukti di GitHub |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | **ILMA JULIANTI** | 101032330052 | Refactor pipeline klasifikasi | Merapikan blok eksperimen yang berulang di `kode/classification/train_and_evaluate.py` dan menambahkan *type hint*, tanpa mengubah logika maupun angka. | PR [#3](https://github.com/ReLyf47/ML-UTS/pull/3), commit [`975d0d3`](https://github.com/ReLyf47/ML-UTS/commit/975d0d3) |
| 2 | **ALIF AKBAR AR RAIYAN** | 101032330121 | Implementasi pipeline dan laporan awal | Struktur proyek dan audit data awal; utilitas bersama (seed, metrik, data loader); pipeline regresi kuantil dengan kalibrasi konformal (Tugas 1); pipeline deteksi fraud dengan kalibrasi isotonic dan aturan biaya (Tugas 2); pipeline segmentasi pelanggan dengan uji stabilitas dan isolasi anomali (Tugas 3); runner `kode/run_all.py`; laporan awal beserta gambar; README; penyempurnaan pipeline, metrik, gambar, dan laporan (commit `polish` dari akun SECULAB milik Alif); merge pull request anggota. | Commit [`ede9645`](https://github.com/ReLyf47/ML-UTS/commit/ede9645), [`2a04854`](https://github.com/ReLyf47/ML-UTS/commit/2a04854), [`3f49e69`](https://github.com/ReLyf47/ML-UTS/commit/3f49e69), [`638257f`](https://github.com/ReLyf47/ML-UTS/commit/638257f), [`d10173d`](https://github.com/ReLyf47/ML-UTS/commit/d10173d), [`c271db1`](https://github.com/ReLyf47/ML-UTS/commit/c271db1), [`c8f0f96`](https://github.com/ReLyf47/ML-UTS/commit/c8f0f96), [`870dc18`](https://github.com/ReLyf47/ML-UTS/commit/870dc18), [`d4b0dd2`](https://github.com/ReLyf47/ML-UTS/commit/d4b0dd2), [`7b4c00d`](https://github.com/ReLyf47/ML-UTS/commit/7b4c00d), [`c12fe33`](https://github.com/ReLyf47/ML-UTS/commit/c12fe33), [`1fa2d19`](https://github.com/ReLyf47/ML-UTS/commit/1fa2d19) (akun SECULAB) |
| 3 | **PRICILIA APRIANA** | 101032300056 | Optimasi kode clustering, revisi dan verifikasi laporan | Optimasi `evaluate_stability` dengan `np.intersect1d` dan validasi NaN/inf pada pipeline clustering (PR #1); perbaikan `laporan.pdf` menjadi satu versi terbaru, identitas anggota, istilah subsampling, typo, dan README Windows (PR #2); revisi laporan Tugas 3: tabel perbandingan metrik per k, narasi pemilihan k, stabilitas, anomali, dan bukti fitur sesuai data, serta strategi persona (PR #4); revisi laporan Tugas 1 dan 2: rincian galat per pulau, kurir, dan kategori merchant, gambar partial dependence, catatan keterbatasan evaluasi, serta pembaruan dokumen QA (PR #5). | PR [#1](https://github.com/ReLyf47/ML-UTS/pull/1), [#2](https://github.com/ReLyf47/ML-UTS/pull/2), [#4](https://github.com/ReLyf47/ML-UTS/pull/4), [#5](https://github.com/ReLyf47/ML-UTS/pull/5) |

**Alur kerja kolaborasi:** implementasi awal (10-16 November 2026) di-commit langsung ke branch `main`. Setelah itu, setiap perubahan masuk lewat *pull request* dari *fork* masing-masing anggota dan di-merge ke `main`. *Main branch* pada deadline adalah sumber penilaian.

### 1.1 Daftar Pull Request

| PR | Anggota | Branch | Lingkup |
| :-: | :--- | :--- | :--- |
| [#1](https://github.com/ReLyf47/ML-UTS/pull/1) | PRICILIA APRIANA | `perf/clustering-stability` | Perhitungan ARI pasangan di `evaluate_stability` memakai `np.intersect1d`, validasi NaN/inf di `fit_transform`, sinkronisasi dokumen QA. ARI tetap 0,8711 ± 0,1414 (subsampling 80% tanpa pengembalian). |
| [#2](https://github.com/ReLyf47/ML-UTS/pull/2) | PRICILIA APRIANA | `fix/laporan-quick` | `laporan.pdf` di root menjadi satu-satunya versi dan versi terbaru, identitas anggota, istilah subsampling, typo, judul kolom terpotong, README Windows. |
| [#3](https://github.com/ReLyf47/ML-UTS/pull/3) | ILMA JULIANTI | `refactor/classification-pipeline` | Deduplikasi logging eksperimen klasifikasi dan *type hint*. |
| [#4](https://github.com/ReLyf47/ML-UTS/pull/4) | PRICILIA APRIANA | `docs/task3-clustering` | Revisi laporan Tugas 3. |
| [#5](https://github.com/ReLyf47/ML-UTS/pull/5) | PRICILIA APRIANA | `docs/task1-task2-breakdown` | Revisi laporan Tugas 1 dan 2, pembaruan `docs/final_quality_assurance.md`. |

---

## 2. Deklarasi Penggunaan Kecerdasan Buatan (AI Assistance Disclosure)

Sesuai prinsip integritas akademik dan transparansi profesional, kelompok kami mendokumentasikan pemanfaatan asisten kecerdasan buatan (AI) dalam pengerjaan proyek ini.

### 2.1 Alat AI yang Digunakan
- **Antigravity AI Coding Assistant**: eksplorasi data, penyusunan *boilerplate* kode, dan draf awal laporan.
- **Hermes Agent (Nous Research)**: audit reproduksibilitas, perbaikan *path* portable, penambahan analisis yang hilang (partial dependence, *resampling*, benchmark Agglomerative/HDBSCAN, bukti seleksi fitur), serta penyelarasan angka laporan dengan metrik pipeline.
- **Codex CLI (OpenAI)**: dipakai Pricilia untuk mengedit kode clustering, `laporan/build_laporan_pdf.py`, `docs/final_quality_assurance.md`, dan README pada PR #1, #2, #4, dan #5.
- **Claude (Anthropic)**: dipakai Pricilia untuk mereview setiap perintah dan diff sebelum disetujui, memeriksa kode sumber yang menjadi dasar klaim laporan (misalnya cara kalibrasi dan partial dependence dihitung), serta mencocokkan angka di PDF dengan berkas metrik.

### 2.2 Lingkup dan Tugas yang Dibantu AI
1. **Eksplorasi dan Audit Data**:
   - Membantu inspeksi skema 5 berkas tabular mentah (~940.000 baris) dan `kamus_data.md`.
   - Mendeteksi disparitas distribusi `jumlah_scan` antara data latih (rata-rata 9,65 scan) dan data uji (rata-rata 1,50 scan) sebagai *post-outcome target leakage*.
2. **Penyusunan Boilerplate & Modularitas Kode**:
   - *Scaffolding* modul Python di bawah `kode/common/`, `kode/regression/`, `kode/classification/`, dan `kode/clustering/`.
   - Skrip visualisasi Matplotlib/Seaborn untuk kurva kalibrasi, batas kuantil, elbow/silhouette, PDP, dan *heatmap* persona.
3. **Dokumentasi & Script Validasi**:
   - Penyusunan draf laporan teknis, kamus data, dan script pengujian kepatuhan berkas prediksi.
4. **Revisi Laporan dan Dokumen QA (PR #1, #2, #4, #5)**:
   - Codex mengetikkan perubahan sesuai instruksi; Claude membantu menemukan klaim yang tidak didukung data dan memeriksa diff. Isi akhir diputuskan dan diverifikasi oleh anggota.

### 2.3 Metode Verifikasi dan Validasi Manusia
- **Reproduksi Satu Perintah**: seluruh pipeline dijalankan ulang dengan `python kode/run_all.py` (seed 42) dan seluruh berkas prediksi tervalidasi oleh `kode/common/validate_submissions.py`.
- **Audit Tanpa Kebocoran**: memastikan fitur `jumlah_scan` dibuang dan tidak ada data uji yang digunakan dalam rekayasa fitur maupun pelatihan.
- **Kepatuhan Batasan Matematis**: memverifikasi bahwa P90 >= P50 terpenuhi 100%, probabilitas klasifikasi berada pada rentang [0, 1], dan anomali klasterisasi dilabeli `-1`.
- **Penyelarasan Angka**: setiap angka di laporan dihasilkan oleh `laporan/build_laporan_pdf.py` dari `outputs/metrics/*.json`.
- **Review Perubahan (PR #1, #2, #4, #5)**: setiap diff dibaca sebelum commit; laporan di-build ulang dan angka baru dicocokkan dengan berkas metrik; `git status` diperiksa untuk memastikan `outputs/` dan berkas prediksi tidak berubah.

---

## 3. Persetujuan Anggota (Member Approval)

Persetujuan diberikan melalui komentar pada pull request yang memperbarui berkas ini. Kotak dicentang setelah ketiga anggota menyetujui.

| No | Nama | NIM | Persetujuan |
| :-: | :--- | :--- | :---: |
| 1 | ILMA JULIANTI | 101032330052 | ☐ |
| 2 | ALIF AKBAR AR RAIYAN | 101032330121 | ☐ |
| 3 | PRICILIA APRIANA | 101032300056 | ☑ |

*Tanggal persetujuan: (diisi setelah ketiga anggota menyetujui)*
