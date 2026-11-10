import numpy as np
from typing import Tuple, Optional, List
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class HistGradientBoostingQuantileModel:
    """Non-crossing Quantile Gradient Boosting for Delivery Duration Estimation.
    
    Fits separate quantile trees for P50 (median) and P90 (upper bound).
    Guarantees non-crossing predictions (P90 >= P50).
    Supports empirical conformal calibration to guarantee >=90% test coverage.
    """
    def __init__(
        self,
        max_iter: int = 150,
        learning_rate: float = 0.08,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 50,
        random_state: int = 42,
        categorical_features: Optional[List[int]] = None
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.categorical_features = categorical_features
        
        self.model_p50 = HistGradientBoostingRegressor(
            loss='quantile',
            quantile=0.50,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            categorical_features=self.categorical_features,
            random_state=self.random_state
        )
        
        self.model_p90 = HistGradientBoostingRegressor(
            loss='quantile',
            quantile=0.90,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            categorical_features=self.categorical_features,
            random_state=self.random_state
        )
        
        self.conformal_adjustment: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit both quantile models."""
        self.model_p50.fit(X, y)
        self.model_p90.fit(X, y)
        self.is_fitted = True
        return self

    def calibrate_conformal(self, X_val: np.ndarray, y_val: np.ndarray, target_coverage: float = 0.90):
        """Perform split conformal quantile calibration to guarantee nominal coverage."""
        raw_p90 = self.model_p90.predict(X_val)
        residuals = y_val - raw_p90
        # Compute the quantile of the residuals needed to hit target coverage
        q_level = 100 * target_coverage
        self.conformal_adjustment = max(0.0, float(np.percentile(residuals, q_level)))
        return self.conformal_adjustment

    def predict(self, X: np.ndarray, apply_conformal: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Predict median (P50) and upper bound (P90) with non-crossing guarantee."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
            
        p50 = self.model_p50.predict(X)
        p90 = self.model_p90.predict(X)
        
        if apply_conformal and self.conformal_adjustment > 0:
            p90 = p90 + self.conformal_adjustment
            
        # Non-crossing guarantee: P90 must be >= P50
        p90 = np.maximum(p90, p50)
        
        # Delivery durations cannot be negative: physical floor of 0.5 hours
        p50 = np.maximum(0.5, p50)
        p90 = np.maximum(0.5, p90)
        
        return p50, p90

class LogRidgeQuantileModel:
    """Linear regression baseline using Ridge on log-transformed durations with empirical quantile intervals."""
    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.log_residuals_p50: float = 0.0
        self.log_residuals_p90: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        log_y = np.log1p(np.maximum(0.1, y))
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, log_y)
        
        # Fit empirical quantiles of log residuals
        preds_log = self.model.predict(X_scaled)
        res = log_y - preds_log
        self.log_residuals_p50 = float(np.percentile(res, 50))
        self.log_residuals_p90 = float(np.percentile(res, 90))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        preds_log = self.model.predict(X_scaled)
        
        p50 = np.expm1(preds_log + self.log_residuals_p50)
        p90 = np.expm1(preds_log + self.log_residuals_p90)
        
        p90 = np.maximum(p90, p50)
        p50 = np.maximum(0.5, p50)
        p90 = np.maximum(0.5, p90)
        return p50, p90
