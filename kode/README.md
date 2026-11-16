# kode/ — Pipeline Machine Learning UTS

Seluruh kode sumber untuk tugas UTS Machine Learning (Regresi, Klasifikasi, Klasterisasi) ada di direktori ini.

## Struktur

```
kode/
├── common/                 # Utilitas bersama (config, I/O, seed, metrik, logger, validasi)
│   ├── config.py           #   Semua path direktori — satu-satunya tempat path didefinisikan
│   ├── io.py               #   Pembaca/penulis CSV & Parquet
│   ├── seeds.py            #   Kontrol seed global (seed = 42)
│   ├── metrics.py          #   Pinball loss, coverage, PR-AUC, biaya fraud
│   ├── figure_saver.py     #   Simpan grafik ke outputs/figures + laporan/figures
│   ├── experiment_tracker.py  # Log eksperimen ke experiments/experiments.{csv,jsonl}
│   └── validate_submissions.py# Validasi 3 berkas prediksi sebelum dikumpulkan
├── data_engineering/       # Langkah 01–07: inspeksi → ekstraksi →bersih → dataset per tugas
├── regression/             # Tugas 1: quantile regression P50/P90 + conformal
├── classification/         # Tugas 2: fraud detection + kalibrasi + aturan biaya
├── clustering/             # Tugas 3: segmentasi + stabilitas + anomali
└── run_all.py              # Orkestrator end-to-end (satu perintah)
```

## Menjalankan

Dari root repositori, setelah dependencies terpasang (`pip install -r requirements.txt`):

```bash
python kode/run_all.py
```

Perintah ini menjalankan seluruh pipeline (data engineering → pelatihan 3 tugas →validasi berkas) dan menuliskan hasil ke:

- `outputs/predictions/prediksi_regresi.csv`
- `outputs/predictions/prediksi_klasifikasi.csv`
- `outputs/predictions/cluster.csv`
- `outputs/metrics/*.json` — semua angka yang dipakai laporan
- `reports/figures/` & `laporan/figures/` — grafik diagnostik
- `experiments/experiments.{csv,jsonl}` — riwayat eksperimen

Untuk validasi berkas pengumpulan saja:

```bash
python kode/common/validate_submissions.py
```

## Reproduktibilitas

- Seed global: `42` (lihat `kode/common/seeds.py`).
- Semua dependensi di-*pin* di `requirements.txt`.
- Tidak ada path absolut tertulis di kode; semua path diturunkan dari `kode/common/config.py` relatif terhadap root repositori, sehingga pipeline bisa dijalankan di mesin/OS mana pun.
- Data mentah dibaca dari `data/raw/` (Git LFS).
