import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional

CATEGORICAL_COLS = [
    'kategori_merchant',
    'kanal',
    'tipe_perangkat',
    'email_domain'
]

NUMERICAL_COLS = [
    'nominal',
    'jam',
    'hari_libur',
    'umur_akun_hari',
    'level_kyc',
    'umur_pengguna_tahun',
    'umur_perangkat_hari',
    'perangkat_baru',
    'jumlah_akun_per_perangkat',
    'ganti_password_7h',
    'gagal_login_24j',
    'jarak_lokasi_km',
    'negara_ip_beda',
    'kartu_tidak_hadir',
    'trx_1j',
    'trx_24j',
    'trx_7h',
    'nominal_24j',
    'saldo_sebelum',
    'persen_saldo',
    'umur_merchant_hari',
    'refund_merchant_30h',
    'volume_merchant_30h'
]

V_COLS = [f'v{i:02d}' for i in range(1, 51)]

class ClassificationFeaturePipeline:
    """Production feature engineering pipeline for Fraud Detection.
    
    Prevents leakage:
    - Never uses future transactions or test labels.
    - Handles structural missingness for first-time users cleanly.
    - Optionally includes or excludes operational policy flag (flag_review_manual).
    """
    def __init__(self, include_policy_flag: bool = False):
        self.include_policy_flag = include_policy_flag
        self.category_mappings: Dict[str, Dict[str, int]] = {}
        self.feature_names: List[str] = []
        self.categorical_indices: List[int] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame):
        """Fit categorical mappings strictly on training split."""
        for col in CATEGORICAL_COLS:
            unique_vals = sorted(df[col].dropna().unique().tolist())
            mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            self.category_mappings[col] = mapping
            
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transform dataframe into feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training data before transform.")
            
        df_out = df.copy()
        
        # 1. Structural missingness handling for first-time users
        is_first_trx = df_out['menit_sejak_trx_terakhir'].isnull().astype(int)
        df_out['is_first_trx'] = is_first_trx
        df_out['menit_sejak_trx_terakhir_clean'] = df_out['menit_sejak_trx_terakhir'].fillna(-1.0)
        df_out['rasio_nominal_rata2_clean'] = df_out['rasio_nominal_rata2'].fillna(1.0)
        
        # 2. Time & cyclical features
        if 'waktu' in df_out.columns:
            dt = pd.to_datetime(df_out['waktu'])
            df_out['trx_dayofweek'] = dt.dt.dayofweek
            df_out['trx_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        else:
            df_out['trx_dayofweek'] = 0
            df_out['trx_is_weekend'] = 0
            
        hour = df_out['jam'].values
        df_out['sin_jam'] = np.sin(2 * np.pi * hour / 24.0)
        df_out['cos_jam'] = np.cos(2 * np.pi * hour / 24.0)
        
        # 3. Behavioral velocity & risk interactions
        # Velocity ratios
        df_out['velocity_1j_to_24j'] = df_out['trx_1j'] / (df_out['trx_24j'] + 1e-4)
        df_out['velocity_24j_to_7h'] = df_out['trx_24j'] / ((df_out['trx_7h'] / 7.0) + 1e-4)
        
        # Nominal burst vs 24h average
        avg_24h_ticket = df_out['nominal_24j'] / (df_out['trx_24j'] + 1e-4)
        df_out['nominal_burst_ratio'] = df_out['nominal'] / (avg_24h_ticket + 1e-4)
        
        # Security alert combination
        df_out['security_alert_score'] = (
            df_out['ganti_password_7h'] * 2.0 +
            (df_out['gagal_login_24j'] > 0).astype(float) * 1.5 +
            df_out['negara_ip_beda'] * 2.5 +
            df_out['perangkat_baru'] * 1.0 +
            (df_out['jumlah_akun_per_perangkat'] > 1).astype(float) * 1.5
        )
        
        # Cross-modality risk: Card Not Present + Foreign IP
        df_out['cnp_foreign_ip'] = df_out['kartu_tidak_hadir'] * df_out['negara_ip_beda']
        
        # Log nominal for linear model support
        df_out['log_nominal'] = np.log1p(df_out['nominal'])
        
        # 4. Categorical encoding
        cat_encoded_cols = []
        for col in CATEGORICAL_COLS:
            mapping = self.category_mappings[col]
            encoded_col = f'{col}_cat'
            df_out[encoded_col] = df_out[col].map(mapping).fillna(0).astype(int)
            cat_encoded_cols.append(encoded_col)
            
        engineered_cols = [
            'is_first_trx',
            'menit_sejak_trx_terakhir_clean',
            'rasio_nominal_rata2_clean',
            'trx_dayofweek',
            'trx_is_weekend',
            'sin_jam',
            'cos_jam',
            'velocity_1j_to_24j',
            'velocity_24j_to_7h',
            'nominal_burst_ratio',
            'security_alert_score',
            'cnp_foreign_ip',
            'log_nominal'
        ]
        
        selected_cols = cat_encoded_cols + engineered_cols + NUMERICAL_COLS + V_COLS
        
        if self.include_policy_flag and 'flag_review_manual' in df_out.columns:
            selected_cols = ['flag_review_manual'] + selected_cols
            
        self.feature_names = selected_cols
        self.categorical_indices = [i for i, c in enumerate(selected_cols) if c.endswith('_cat')]
        
        X = df_out[selected_cols].values
        return X, selected_cols
