import numpy as np
import pandas as pd
from typing import Dict, Tuple

class GlobalMedianBaseline:
    """Simple non-parametric baseline predicting global median and 90th percentile."""
    def __init__(self):
        self.p50_val: float = 0.0
        self.p90_val: float = 0.0

    def fit(self, y: pd.Series):
        self.p50_val = float(np.percentile(y, 50))
        self.p90_val = float(np.percentile(y, 90))
        return self

    def predict(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        p50 = np.full(n_samples, self.p50_val)
        p90 = np.full(n_samples, self.p90_val)
        return p50, p90

class ServiceMedianBaseline:
    """Service-conditional baseline predicting median and 90th percentile per delivery tier."""
    def __init__(self):
        self.service_p50: Dict[str, float] = {}
        self.service_p90: Dict[str, float] = {}
        self.global_p50: float = 0.0
        self.global_p90: float = 0.0

    def fit(self, services: pd.Series, y: pd.Series):
        self.global_p50 = float(np.percentile(y, 50))
        self.global_p90 = float(np.percentile(y, 90))
        
        df = pd.DataFrame({'layanan': services, 'durasi_jam': y})
        for s, group in df.groupby('layanan'):
            self.service_p50[s] = float(np.percentile(group['durasi_jam'], 50))
            self.service_p90[s] = float(np.percentile(group['durasi_jam'], 90))
        return self

    def predict(self, services: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        p50 = services.map(self.service_p50).fillna(self.global_p50).values
        p90 = services.map(self.service_p90).fillna(self.global_p90).values
        # Non-crossing guarantee
        p90 = np.maximum(p90, p50)
        return p50, p90
