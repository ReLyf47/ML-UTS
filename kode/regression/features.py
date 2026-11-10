import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional

# Low-cardinality categoricals (cardinality <= 30, fully supported by HistGBM uint8 bins)
LOW_CARDINALITY_CATEGORICALS = [
    'layanan',
    'mitra_kurir',
    'hub_asal',
    'kota_asal',
    'pulau_tujuan'
]

NUMERICAL_COLS = [
    'jarak_km',
    'berat_kg',
    'volume_liter',
    'jumlah_item',
    'nilai_barang',
    'asuransi',
    'cod',
    'alamat_skor',
    'pengalaman_kurir_bulan',
    'rating_penjual',
    'umur_akun_penjual_hari',
    'volume_penjual_30h',
    'antrean_hub_asal',
    'indeks_kemacetan',
    'hujan_asal_mm',
    'hujan_tujuan_mm'
]

# 50 internal system features
F_COLS = [f'f{i:02d}' for i in range(1, 51)]

class RegressionFeaturePipeline:
    """Production feature engineering pipeline for Delivery Duration Quantile Regression.
    
    Prevents leakage:
    - Never uses test data for imputation or category encoding.
    - Strictly excludes post-outcome tracking scan leakage (jumlah_scan).
    - Robustly handles unseen test categories (e.g. courier partner K13).
    - Target-encodes high-cardinality kota_tujuan (405 categories) with m-estimate smoothing.
    """
    def __init__(self, include_leakage_feature: bool = False, smoothing_m: float = 20.0):
        self.include_leakage_feature = include_leakage_feature
        self.smoothing_m = smoothing_m
        self.category_mappings: Dict[str, Dict[str, int]] = {}
        self.global_weight_median: float = 2.0
        self.service_weight_medians: Dict[str, float] = {}
        self.city_dest_target_map: Dict[str, float] = {}
        self.city_dest_freq_map: Dict[str, float] = {}
        self.global_target_median: float = 61.7
        self.feature_names: List[str] = []
        self.categorical_indices: List[int] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, y: Optional[np.ndarray] = None):
        """Fit imputation statistics, smoothed target encodings, and categorical mappings strictly on training data."""
        # 1. Fit weight imputation
        self.global_weight_median = float(df['berat_kg'].dropna().median())
        self.service_weight_medians = df.groupby('layanan')['berat_kg'].median().to_dict()
        
        # 2. Fit low-cardinality categorical encodings (reserve 0 for unseen/unknown)
        for col in LOW_CARDINALITY_CATEGORICALS:
            unique_vals = sorted(df[col].dropna().unique().tolist())
            # 1-indexed; 0 represents unseen/missing
            mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            self.category_mappings[col] = mapping
            
        # 3. Fit high-cardinality kota_tujuan via smoothed target encoding & frequency encoding
        if y is not None:
            target_series = pd.Series(y, index=df.index)
        elif 'durasi_jam' in df.columns:
            target_series = df['durasi_jam']
        else:
            target_series = pd.Series(61.7, index=df.index)
            
        self.global_target_median = float(target_series.median())
        
        city_counts = df['kota_tujuan'].value_counts()
        total_rows = len(df)
        self.city_dest_freq_map = (city_counts / total_rows).to_dict()
        
        city_medians = target_series.groupby(df['kota_tujuan']).median()
        
        m = self.smoothing_m
        for city, count in city_counts.items():
            median_c = city_medians.get(city, self.global_target_median)
            # m-estimate smoothed target encoding
            smoothed_val = (count * median_c + m * self.global_target_median) / (count + m)
            self.city_dest_target_map[city] = float(smoothed_val)
            
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transform dataframe into feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training data before transform.")
            
        df_out = df.copy()
        
        # 1. Weight imputation
        imputed_weight = df_out['berat_kg'].copy()
        for service, median_val in self.service_weight_medians.items():
            mask = (df_out['layanan'] == service) & (imputed_weight.isnull())
            imputed_weight.loc[mask] = median_val
        imputed_weight = imputed_weight.fillna(self.global_weight_median)
        df_out['berat_kg_clean'] = imputed_weight
        
        # 2. Temporal feature engineering from tgl_pickup and jam_pickup
        tgl = pd.to_datetime(df_out['tgl_pickup'])
        df_out['pickup_dayofweek'] = tgl.dt.dayofweek
        df_out['pickup_is_weekend'] = (tgl.dt.dayofweek >= 5).astype(int)
        df_out['pickup_month'] = tgl.dt.month
        df_out['pickup_day'] = tgl.dt.day
        
        # Cyclical hour features
        hour = df_out['jam_pickup'].values
        df_out['sin_jam_pickup'] = np.sin(2 * np.pi * hour / 24.0)
        df_out['cos_jam_pickup'] = np.cos(2 * np.pi * hour / 24.0)
        
        # 3. Domain ratio features
        df_out['densitas_kg_per_l'] = df_out['berat_kg_clean'] / (df_out['volume_liter'] + 1e-4)
        df_out['nilai_per_kg'] = df_out['nilai_barang'] / (df_out['berat_kg_clean'] + 1e-4)
        df_out['nilai_per_item'] = df_out['nilai_barang'] / (df_out['jumlah_item'] + 1e-4)
        df_out['is_same_city'] = (df_out['kota_asal'] == df_out['kota_tujuan']).astype(int)
        
        # 4. Low-cardinality categorical encodings (mapping unseen test categories like K13 to 0)
        cat_encoded_cols = []
        for col in LOW_CARDINALITY_CATEGORICALS:
            mapping = self.category_mappings[col]
            encoded_col = f'{col}_cat'
            df_out[encoded_col] = df_out[col].map(mapping).fillna(0).astype(int)
            cat_encoded_cols.append(encoded_col)
            
        # 5. High-cardinality kota_tujuan features
        df_out['kota_tujuan_te'] = df_out['kota_tujuan'].map(self.city_dest_target_map).fillna(self.global_target_median).astype(float)
        df_out['kota_tujuan_freq'] = df_out['kota_tujuan'].map(self.city_dest_freq_map).fillna(0.0).astype(float)
        
        # 6. Assemble final feature columns list
        engineered_num_cols = [
            'pickup_dayofweek',
            'pickup_is_weekend',
            'pickup_month',
            'pickup_day',
            'jam_pickup',
            'sin_jam_pickup',
            'cos_jam_pickup',
            'densitas_kg_per_l',
            'nilai_per_kg',
            'nilai_per_item',
            'is_same_city',
            'berat_kg_clean',
            'kota_tujuan_te',
            'kota_tujuan_freq'
        ]
        
        base_num_cols = [c for c in NUMERICAL_COLS if c != 'berat_kg']
        
        selected_cols = cat_encoded_cols + engineered_num_cols + base_num_cols + F_COLS
        
        # Optional: leakage ablation feature
        if self.include_leakage_feature and 'jumlah_scan' in df_out.columns:
            selected_cols = ['jumlah_scan'] + selected_cols
            
        self.feature_names = selected_cols
        # Track indices of low-cardinality categoricals
        self.categorical_indices = [i for i, col in enumerate(selected_cols) if col.endswith('_cat')]
        
        X = df_out[selected_cols].values
        return X, selected_cols
