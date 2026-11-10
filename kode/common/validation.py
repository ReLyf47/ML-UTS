import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

def validate_file_exists(path: Union[str, Path]) -> bool:
    """Check that file exists and has non-zero size."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file does not exist: {p}")
    if p.stat().st_size == 0:
        raise ValueError(f"File is empty (0 bytes): {p}")
    return True

def validate_unique_ids(df: pd.DataFrame, id_col: str) -> bool:
    """Verify that the primary identifier column has 0 duplicates and 0 nulls."""
    if id_col not in df.columns:
        raise KeyError(f"Identifier column '{id_col}' not found in dataframe.")
    
    null_count = df[id_col].isnull().sum()
    if null_count > 0:
        raise ValueError(f"Identifier column '{id_col}' contains {null_count} null values.")
        
    unique_count = df[id_col].nunique()
    total_count = len(df)
    if unique_count != total_count:
        dupes = total_count - unique_count
        raise ValueError(f"Identifier column '{id_col}' has {dupes} duplicate values ({unique_count} unique vs {total_count} total).")
    return True

def validate_row_count(df: pd.DataFrame, expected_rows: int, dataset_name: str = "dataset") -> bool:
    """Verify exact row count match."""
    actual_rows = len(df)
    if actual_rows != expected_rows:
        raise ValueError(f"Row count mismatch for {dataset_name}: expected {expected_rows}, got {actual_rows}.")
    return True

def validate_no_crossing(p50: np.ndarray, p90: np.ndarray) -> bool:
    """Ensure non-crossing quantiles: p90 >= p50 for all predictions."""
    violations = np.sum(p90 < p50)
    if violations > 0:
        max_diff = np.max(p50 - p90)
        raise ValueError(f"Quantile crossing detected: {violations} instances where P90 < P50 (max violation: {max_diff:.4f}).")
    return True

def validate_probabilities(probs: np.ndarray) -> bool:
    """Ensure all probabilities are finite and bounded in [0.0, 1.0]."""
    if not np.all(np.isfinite(probs)):
        raise ValueError("Probabilities contain NaN, inf, or non-finite values.")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError(f"Probabilities out of bounds [0, 1]: min={np.min(probs)}, max={np.max(probs)}")
    return True

def get_dataset_quality_stats(df: pd.DataFrame, id_col: Optional[str] = None) -> Dict[str, Any]:
    """Calculate summary statistics for manifests and audit reports."""
    total_rows = len(df)
    missing_dict = df.isnull().sum().to_dict()
    missing_cols = {k: int(v) for k, v in missing_dict.items() if v > 0}
    
    stats = {
        "row_count": total_rows,
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(dtype) for c, dtype in df.dtypes.items()},
        "missing_counts": missing_cols,
        "total_missing_cells": int(sum(missing_cols.values())),
        "duplicate_rows": int(df.duplicated().sum())
    }
    
    if id_col and id_col in df.columns:
        stats["unique_ids"] = int(df[id_col].nunique())
        stats["is_id_unique"] = bool(stats["unique_ids"] == total_rows)
        
    return stats
