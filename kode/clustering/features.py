import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler

PORSI_KATEGORI_COLS = [
    'porsi_fashion', 'porsi_elektronik', 'porsi_kecantikan', 'porsi_rumah_tangga',
    'porsi_makanan_minuman', 'porsi_ibu_bayi', 'porsi_olahraga', 'porsi_hobi',
    'porsi_otomotif', 'porsi_kesehatan', 'porsi_buku_alat_tulis', 'porsi_voucher_digital',
    'porsi_gadget_aksesoris', 'porsi_perabot', 'porsi_pertanian', 'porsi_lainnya'
]

PORSI_JAM_COLS = [
    'porsi_jam_pagi', 'porsi_jam_siang', 'porsi_jam_sore', 'porsi_jam_malam'
]

N_COLS = [f'n{i:02d}' for i in range(1, 11)]

class CustomerClusteringPipeline:
    """Feature engineering and scaling pipeline for Customer Segmentation.
    
    Transforms skewed RFM variables, handles basket size missingness,
    preserves compositional category profiles, and generates momentum signals.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        df_clean = df.copy()
        
        # 1. Deterministic reconstruction of missing rata2_keranjang
        mask_missing_basket = df_clean['rata2_keranjang'].isnull()
        reconstructed_basket = np.where(
            df_clean['frekuensi_90h'] > 0,
            df_clean['belanja_90h'] / np.maximum(1.0, df_clean['frekuensi_90h']),
            0.0
        )
        df_clean['rata2_keranjang_clean'] = np.where(
            mask_missing_basket,
            reconstructed_basket,
            df_clean['rata2_keranjang']
        )
        
        # 2. Log1p transformation of heavy-tailed monetary and frequency variables
        df_clean['log_belanja_90h'] = np.log1p(df_clean['belanja_90h'])
        df_clean['log_frekuensi_90h'] = np.log1p(df_clean['frekuensi_90h'])
        df_clean['log_rata2_keranjang'] = np.log1p(df_clean['rata2_keranjang_clean'])
        df_clean['log_recency_hari'] = np.log1p(df_clean['recency_hari'])
        df_clean['log_belanja_30h'] = np.log1p(df_clean['belanja_30h'])
        df_clean['log_frekuensi_30h'] = np.log1p(df_clean['frekuensi_30h'])
        
        # 3. Behavioral momentum (30d vs 90d normalized ratio)
        # Ratio of recent 30d run-rate to overall 90d rate (>1.0 means accelerating engagement)
        df_clean['momentum_belanja'] = (df_clean['belanja_30h'] * 3.0) / (df_clean['belanja_90h'] + 1000.0)
        df_clean['momentum_frekuensi'] = (df_clean['frekuensi_30h'] * 3.0) / (df_clean['frekuensi_90h'] + 1.0)
        
        # 4. Promotion & discount sensitivity
        promo_cols = ['persen_voucher', 'persen_flash_sale', 'persen_paylater', 'persen_cod']
        
        # 5. Satisfaction & Friction
        friction_cols = ['retur_90h', 'rating_beri_rata2', 'ulasan_90h']
        
        # 6. App Engagement
        engagement_cols = ['sesi_per_minggu', 'durasi_sesi_menit', 'rasio_konversi']
        
        # 7. Demographics
        df_clean['is_android'] = (df_clean['perangkat'] == 'android').astype(int)
        df_clean['is_ios'] = (df_clean['perangkat'] == 'ios').astype(int)
        df_clean['is_web'] = (df_clean['perangkat'] == 'web').astype(int)
        demo_cols = ['lama_member_bulan', 'umur', 'kota_tier', 'is_android', 'is_ios', 'is_web']
        
        # 8. Intent
        intent_cols = ['jumlah_alamat', 'wishlist', 'klik_iklan_30h', 'item_per_pesanan']
        
        # Assemble feature list
        selected_cols = (
            ['log_belanja_90h', 'log_frekuensi_90h', 'log_rata2_keranjang', 'log_recency_hari',
             'log_belanja_30h', 'log_frekuensi_30h', 'momentum_belanja', 'momentum_frekuensi'] +
            promo_cols +
            friction_cols +
            engagement_cols +
            demo_cols +
            intent_cols +
            PORSI_KATEGORI_COLS +
            PORSI_JAM_COLS +
            N_COLS
        )
        
        self.feature_names = selected_cols
        X_scaled = self.scaler.fit_transform(df_clean[selected_cols].values)
        self.is_fitted = True
        
        return X_scaled, selected_cols, df_clean
