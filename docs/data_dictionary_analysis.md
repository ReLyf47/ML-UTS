# Data Dictionary Analysis & Column Interpretation

**Project**: UTS Machine Learning  
**Reference Document**: `data/raw/kamus_data.md`  

---

## 1. Task 1: Regression (Delivery Duration Estimation)

**Dataset Files**: `regresi_train.csv.gz` (77 cols), `regresi_test.csv.gz` (76 cols)  
**Prediction Cutoff Context**: Predictions must be made at the exact instant the package is picked up by the courier (`tgl_pickup`, `jam_pickup`).

| Column Name | Semantic Meaning | Data Type | Physical Range / Values | Availability at Pickup | Analytical Role & Handling |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `id_paket` | Unique package ID | String | Unique hash/ID | Yes | Identifier only; drop before model training. |
| `tgl_pickup` | Pickup date | String (Date) | 2025-01-01 to 2025-10-31 | Yes | Temporal splitting; extract day-of-week, month, day. |
| `jam_pickup` | Pickup hour of day | Integer | 0 to 23 | Yes | Time feature; cyclical or ordinal encoding. |
| `layanan` | Shipping service tier | Categorical (str) | 4 values: Instant, Same Day, Regular, Cargo | Yes | High-impact signal; Target/frequency/one-hot encoding. |
| `mitra_kurir` | Courier partner entity | Categorical (str) | Train: 12 partners, Test: 13 (K13 unseen) | Yes | Group unseen test partner into 'Other' / global default. |
| `hub_asal` | Origin sorting hub | Categorical (str) | 6 hubs (H01-H06) | Yes | Categorical feature; hub congestion / routing signal. |
| `kota_asal` | Origin city name | Categorical (str) | 30 unique origin cities | Yes | Categorical feature; geographic origin. |
| `pulau_tujuan` | Destination island | Categorical (str) | 7 islands (Jawa, Sumatera, etc.) | Yes | Strong geographic signal; inter-island vs intra-island. |
| `kota_tujuan` | Destination city name | Categorical (str) | 405 destination cities | Yes | Target/frequency encoding; high cardinality. |
| `jarak_km` | System routing distance | Float | Numeric km (>0) | Yes | Continuous numerical; non-linear relationship with time. |
| `berat_kg` | Seller declared weight | Float | kg (>0), 5.5% missing | Yes | Missing value imputation (median by service/volume). |
| `volume_liter` | Package volume in liters | Float | Liters (>0) | Yes | Continuous numerical; volumetric parcel size. |
| `jumlah_item` | Item count in parcel | Integer | >= 1 | Yes | Numerical count; parcel complexity. |
| `nilai_barang` | Declared item value | Float / Int | Indonesian Rupiah (IDR) | Yes | Monetary value; correlated with caution / service. |
| `asuransi` | Insurance flag | Binary (0/1) | 0 = No, 1 = Insured | Yes | Binary indicator. |
| `cod` | Cash on Delivery flag | Binary (0/1) | 0 = No, 1 = COD | Yes | Binary indicator (COD often adds delivery friction). |
| `alamat_skor` | Address quality score | Float | [0.0, 1.0] | Yes | Address completeness; low scores imply delay risk. |
| `pengalaman_kurir_bulan`| Courier experience | Float / Int | Months on duty | Yes | Driver productivity / routing familiarity. |
| `rating_penjual` | Seller average rating | Float | [1.0, 5.0] | Yes | Seller fulfillment reliability. |
| `umur_akun_penjual_hari`| Seller tenure | Integer | Days since onboarding | Yes | Seller maturity. |
| `volume_penjual_30h` | Seller 30d parcel volume| Integer | Count | Yes | Seller scale / packaging speed. |
| `antrean_hub_asal` | Parcels in queue at hub | Integer | Count at pickup | Yes | Sorting facility congestion at departure. |
| `indeks_kemacetan` | City traffic index | Float | [0.0, 1.0] | Yes | Origin congestion at pickup. |
| `hujan_asal_mm` | Forecast rainfall origin | Float | mm rain | Yes | Weather impediment at origin. |
| `hujan_tujuan_mm` | Forecast rainfall dest. | Float | mm rain | Yes | Weather impediment at destination. |
| `jumlah_scan` | Tracking scan count | Integer | Train: 1-97, Test: 1-7 | **NO (LEAKAGE)** | **CRITICAL LEAKAGE**: In train, represents total lifetime scans until delivery. In test, represents scans at pickup. **MUST EXCLUDE**. |
| `f01` - `f50` | Logistics system features | Float | Continuous normalized | Yes | Anonymized internal logistics features. Keep all. |
| `durasi_jam` | Total delivery hours | Float | Target: 11.6 to 25,452.0 | Post-event | **Target variable**. Only present in training set. |

---

## 2. Task 2: Classification (Fraud Detection)

**Dataset Files**: `klasifikasi_train.csv.gz` (85 cols), `klasifikasi_test.csv.gz` (84 cols)  
**Prediction Cutoff Context**: Real-time transaction scoring. Evaluated when the transaction is attempted.

| Column Name | Semantic Meaning | Data Type | Physical Range / Values | Availability at Scoring | Analytical Role & Handling |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `id_transaksi` | Transaction ID | String | Unique ID | Yes | Primary ID; drop before training. |
| `waktu` | Transaction timestamp | Datetime str | 2025-01-01 to 2025-10-31 | Yes | Temporal ordering; extract hour, day-of-week, elapsed. |
| `id_pengguna` | User account ID | String | 120,000 unique users | Yes | Entity key for historical rolling aggregations. |
| `id_merchant` | Merchant ID | String | 15,000 unique merchants | Yes | Entity key for historical risk profiles. |
| `kategori_merchant`| Merchant category | Categorical (str) | 24 categories | Yes | Categorical risk signal. |
| `kanal` | Payment channel | Categorical (str) | 5 channels | Yes | Payment instrument risk profile. |
| `tipe_perangkat` | Device category | Categorical (str) | 3 types (mobile, web, tablet) | Yes | Device risk profile. |
| `nominal` | Transaction amount | Float | IDR (1,000 to 25.46M) | Yes | Primary monetary exposure for cost optimization. |
| `jam` | Transaction hour | Integer | 0 to 23 | Yes | Time-of-day risk (late night spikes). |
| `hari_libur` | National holiday flag | Binary (0/1) | 0 = Workday, 1 = Holiday | Yes | Contextual calendar feature. |
| `umur_akun_hari` | User account age | Integer | Days since registration | Yes | New account fraud vector. |
| `level_kyc` | KYC verification tier | Integer | 0, 1, 2, ... | Yes | Identity verification assurance. |
| `umur_pengguna_tahun`| User age | Integer | Years | Yes | Demographic profile. |
| `email_domain` | Email provider domain | Categorical (str) | 5 domains | Yes | Disposable vs reputable email provider. |
| `umur_perangkat_hari`| Days device seen on account| Integer | >= 0 | Yes | Device binding security signal. |
| `perangkat_baru` | First seen <= 3 days | Binary (0/1) | 0 or 1 | Yes | Account takeover / credential stuffing indicator. |
| `jumlah_akun_per_perangkat`| Accounts on this device| Integer | >= 1 | Yes | Device fingerprinting fraud farm signal. |
| `ganti_password_7h` | Password changed past 7d | Binary (0/1) | 0 or 1 | Yes | Account takeover precursor. |
| `gagal_login_24j` | Failed logins past 24h | Integer | >= 0 | Yes | Brute-force / unauthorized access signal. |
| `jarak_lokasi_km` | Distance from habitual loc | Float | km | Yes | Geographic anomaly detection. |
| `negara_ip_beda` | IP country differs from KYC | Binary (0/1) | 0 or 1 | Yes | VPN / proxy / foreign IP indicator. |
| `kartu_tidak_hadir` | Card-not-present flag | Binary (0/1) | 0 or 1 | Yes | High-risk payment modality. |
| `menit_sejak_trx_terakhir`| Minutes since prior txn | Float | >= 0, NaN for 1st txn | Yes | Velocity indicator; NaN denotes first-time user. |
| `trx_1j`, `trx_24j`, `trx_7h`| Prior txn frequency counts| Integer | Counts | Yes | Frequency bursts / velocity anomalies. |
| `nominal_24j` | Prior 24h cumulative amount| Float | IDR | Yes | Monetary velocity. |
| `rasio_nominal_rata2`| Nominal / prior user avg | Float | Ratio, NaN for 1st txn | Yes | Ticket size surge indicator; NaN for first-time user. |
| `saldo_sebelum` | Wallet balance before txn | Float | IDR | Yes | Wallet emptying / cash-out risk. |
| `persen_saldo` | Fraction of wallet spent | Float | [0.0, 1.0] | Yes | Drain proportion. |
| `umur_merchant_hari` | Merchant account age | Integer | Days | Yes | Merchant maturity. |
| `refund_merchant_30h`| Merchant refund rate (30d) | Float | Percentage | Yes | Merchant chargeback/dispute risk. |
| `volume_merchant_30h`| Merchant txn volume (30d) | Integer | Count | Yes | Merchant throughput. |
| `flag_review_manual` | Historical review flag | Binary (0/1) | 0 = No, 1 = Reviewed | Available (Policy)| Legacy review rule indicator. (See Leakage Audit). |
| `v01` - `v50` | Risk model features | Float | Continuous normalized | Yes | Anonymized internal risk features. Keep all. |
| `is_fraud` | Ground truth fraud label | Binary (0/1) | Target: 0 or 1 | Post-investigation | **Target variable**. Only present in training set. |

---

## 3. Task 3: Clustering (Customer Segmentation)

**Dataset File**: `clustering.csv.gz` (120,000 rows, 60 cols)  
**Nature**: Static customer-level 90-day RFM, behavioral, and engagement summary.

| Column Group | Feature Names | Description | Analytical Role & Transformation |
| :--- | :--- | :--- | :--- |
| **Identity** | `id_pelanggan` | Customer unique identifier | Key only; excluded from distance / clustering metrics. |
| **RFM Core** | `recency_hari`<br>`frekuensi_90h`<br>`belanja_90h`<br>`rata2_keranjang` | Days since purchase, total 90d purchases, gross 90d spend (IDR), average basket value (IDR). | Core RFM behavioral dimensions. Heavy right-skew: apply `np.log1p` and Robust/Standard scaling. `rata2_keranjang` has 5.61% missing (impute via `belanja_90h / frekuensi_90h`). |
| **Order Dynamics** | `item_per_pesanan` | Average units per basket | Order size behavior. Standardized. |
| **Product Composition** | `porsi_fashion`, `porsi_elektronik`, `porsi_kecantikan`, `porsi_rumah_tangga`, `porsi_makanan_minuman`, `porsi_ibu_bayi`, `porsi_olahraga`, `porsi_hobi`, `porsi_otomotif`, `porsi_kesehatan`, `porsi_buku_alat_tulis`, `porsi_voucher_digital`, `porsi_gadget_aksesoris`, `porsi_perabot`, `porsi_pertanian`, `porsi_lainnya` (16 cols) | Category spend proportion of customer (sum = 1.0). | Compositional data. Captures lifestyle and catalog affinity. Can use Centered Log-Ratio (CLR) or PCA/direct proportions with simplex preservation. |
| **Temporal Habits** | `porsi_jam_pagi`, `porsi_jam_siang`, `porsi_jam_sore`, `porsi_jam_malam` (4 cols) | Shopping hour proportion (sum = 1.0). | Time-of-day behavioral pattern (night owl vs office hours). |
| **Promotion / Channel** | `persen_voucher`, `persen_flash_sale`, `persen_paylater`, `persen_cod` | Discount sensitivity and payment choice percentages [0, 1]. | Price sensitivity and financial modality indicators. |
| **Satisfaction & CS** | `retur_90h`, `rating_beri_rata2`, `ulasan_90h` | Returns count, avg review rating, review count. | Customer friction, satisfaction, and feedback engagement. |
| **Engagement** | `sesi_per_minggu`, `durasi_sesi_menit`, `rasio_konversi` | App visits/wk, session length, conversion rate. | Digital engagement and intent-to-buy efficiency. |
| **Demographics** | `lama_member_bulan`, `umur`, `kota_tier`, `perangkat` | Membership tenure, age, city tier (1/2/3), device OS (android/ios/web). | Customer lifecycle, purchasing power, and platform preference. |
| **Intent & Momentum** | `jumlah_alamat`, `wishlist`, `klik_iklan_30h`, `belanja_30h`, `frekuensi_30h` | Saved addresses, wishlist size, ad clicks (30d), recent spend (30d), recent orders (30d). | Momentum vs churn: comparing 30d vs 90d spend captures accelerating vs disengaging customers. |
| **CRM Latent** | `n01` - `n10` (10 cols) | Continuous latent CRM attributes. | Normalized CRM features; dimension reduction or direct standardized features. |
