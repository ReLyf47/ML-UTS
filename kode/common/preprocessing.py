import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

def clean_string_columns(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Trim leading/trailing whitespaces and standardize casing for string columns."""
    df_clean = df.copy()
    if cols is None:
        cols = df_clean.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
    return df_clean

def add_cyclical_time_features(df: pd.DataFrame, hour_col: str) -> pd.DataFrame:
    """Add sine and cosine cyclical encodings for hour of day."""
    df_out = df.copy()
    hours = df_out[hour_col].astype(float).values
    df_out[f"sin_{hour_col}"] = np.sin(2 * np.pi * hours / 24.0)
    df_out[f"cos_{hour_col}"] = np.cos(2 * np.pi * hours / 24.0)
    return df_out

def apply_log1p(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Apply log1p transformation to positive skewed numeric variables."""
    df_out = df.copy()
    for col in cols:
        if col in df_out.columns:
            # Clip negative values if any to 0 before log1p
            val = np.maximum(0.0, df_out[col].astype(float).values)
            df_out[f"log_{col}"] = np.log1p(val)
    return df_out

class CategoryIndexEncoder:
    """Deterministic integer encoder for low-to-medium cardinality categoricals with unseen category handling."""
    def __init__(self, reserve_zero_for_unseen: bool = True):
        self.reserve_zero = reserve_zero_for_unseen
        self.mapping: Dict[str, int] = {}
        self.is_fitted: bool = False

    def fit(self, series: pd.Series):
        uniques = sorted(series.dropna().unique().tolist())
        start_idx = 1 if self.reserve_zero else 0
        self.mapping = {val: idx + start_idx for idx, val in enumerate(uniques)}
        self.is_fitted = True
        return self

    def transform(self, series: pd.Series, unseen_val: int = 0) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("CategoryIndexEncoder must be fitted before transform.")
        return series.map(self.mapping).fillna(unseen_val).astype(int).values

    def fit_transform(self, series: pd.Series, unseen_val: int = 0) -> np.ndarray:
        return self.fit(series).transform(series, unseen_val=unseen_val)
