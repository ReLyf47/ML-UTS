import os
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from kode.common.io import find_raw_file, read_raw_dataset

def get_regression_data(sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load regression training and test datasets safely from .csv or .csv.gz."""
    df_train = read_raw_dataset("regresi_train", sample_size=sample_size)
    df_test = read_raw_dataset("regresi_test", sample_size=sample_size)
    return df_train, df_test

def get_classification_data(sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load classification training and test datasets safely from .csv or .csv.gz."""
    df_train = read_raw_dataset("klasifikasi_train", sample_size=sample_size)
    df_test = read_raw_dataset("klasifikasi_test", sample_size=sample_size)
    return df_train, df_test

def get_clustering_data(sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load clustering dataset safely from .csv or .csv.gz."""
    return read_raw_dataset("clustering", sample_size=sample_size)
