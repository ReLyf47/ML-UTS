import os
import pandas as pd
from typing import Tuple, Optional

DATA_RAW_DIR = "/home/relyf/ML/ML-UTS/data/raw"

def get_regression_data(sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load regression training and test datasets.
    
    Excludes jumlah_scan immediately if desired or leaves it for pipeline exclusion.
    """
    train_path = os.path.join(DATA_RAW_DIR, "regresi_train.csv.gz")
    test_path = os.path.join(DATA_RAW_DIR, "regresi_test.csv.gz")
    
    if sample_size:
        df_train = pd.read_csv(train_path, nrows=sample_size)
        df_test = pd.read_csv(test_path, nrows=sample_size)
    else:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
    return df_train, df_test

def get_classification_data(sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load classification training and test datasets."""
    train_path = os.path.join(DATA_RAW_DIR, "klasifikasi_train.csv.gz")
    test_path = os.path.join(DATA_RAW_DIR, "klasifikasi_test.csv.gz")
    
    if sample_size:
        df_train = pd.read_csv(train_path, nrows=sample_size)
        df_test = pd.read_csv(test_path, nrows=sample_size)
    else:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
    return df_train, df_test

def get_clustering_data(sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load clustering dataset."""
    clust_path = os.path.join(DATA_RAW_DIR, "clustering.csv.gz")
    if sample_size:
        df = pd.read_csv(clust_path, nrows=sample_size)
    else:
        df = pd.read_csv(clust_path)
    return df
