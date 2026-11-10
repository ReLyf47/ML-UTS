import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from kode.common.config import DATA_RAW_DIR

def find_raw_file(base_name: str) -> Path:
    """Find a raw data file by base name, checking for .csv and .csv.gz extensions."""
    # Check direct name
    candidates = [
        DATA_RAW_DIR / base_name,
        DATA_RAW_DIR / f"{base_name}.csv",
        DATA_RAW_DIR / f"{base_name}.csv.gz"
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Raw dataset '{base_name}' not found in {DATA_RAW_DIR}. Checked: {[str(c) for c in candidates]}")

def read_raw_dataset(base_name: str, sample_size: Optional[int] = None, low_memory: bool = False) -> pd.DataFrame:
    """Read raw dataset seamlessly whether gzipped or plain CSV."""
    path = find_raw_file(base_name)
    if sample_size:
        return pd.read_csv(path, nrows=sample_size, low_memory=low_memory)
    return pd.read_csv(path, low_memory=low_memory)

def save_parquet(df: pd.DataFrame, path: Union[str, Path], compression: str = "snappy") -> Path:
    """Save dataframe to Parquet with index dropped and directory auto-created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression=compression)
    return path

def read_parquet(path: Union[str, Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read Parquet file efficiently."""
    return pd.read_parquet(path, columns=columns)

def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> Path:
    """Save dictionary/list to formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    return path

def read_json(path: Union[str, Path]) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_csv(df: pd.DataFrame, path: Union[str, Path], index: bool = False) -> Path:
    """Save dataframe to CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path

def read_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read CSV file."""
    return pd.read_csv(path, **kwargs)
