import numpy as np
from typing import Optional, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

class LogisticRegressionFraudBaseline:
    """Class-weighted Logistic Regression baseline for Fraud Detection."""
    def __init__(self, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs'
        )
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

class HistGradientBoostingFraudModel:
    """Production Gradient Boosting Classifier for Extreme Imbalance Fraud Detection.
    
    Includes native categorical support, balanced class weighting, and probability calibration.
    """
    def __init__(
        self,
        max_iter: int = 150,
        learning_rate: float = 0.08,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 50,
        class_weight: Optional[str] = 'balanced',
        categorical_features: Optional[List[int]] = None,
        random_state: int = 42
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.categorical_features = categorical_features
        self.random_state = random_state
        
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            categorical_features=self.categorical_features,
            random_state=self.random_state
        )
        
        self.calibrator_isotonic: Optional[IsotonicRegression] = None
        self.calibrator_sigmoid: Optional[LogisticRegression] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray):
        """Fit Isotonic and Sigmoid calibrators on out-of-time validation probabilities."""
        raw_val_probs = self.model.predict_proba(X_val)[:, 1]
        
        # 1. Isotonic calibrator
        self.calibrator_isotonic = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        self.calibrator_isotonic.fit(raw_val_probs, y_val)
        
        # 2. Sigmoid (Platt) calibrator
        self.calibrator_sigmoid = LogisticRegression(solver='lbfgs', random_state=self.random_state)
        # Reshape for single-feature logistic regression
        self.calibrator_sigmoid.fit(raw_val_probs.reshape(-1, 1), y_val)
        return self

    def predict_proba(self, X: np.ndarray, calibration: str = 'isotonic') -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
            
        raw_probs = self.model.predict_proba(X)[:, 1]
        
        if calibration == 'isotonic' and self.calibrator_isotonic is not None:
            cal_probs = self.calibrator_isotonic.predict(raw_probs)
        elif calibration == 'sigmoid' and self.calibrator_sigmoid is not None:
            cal_probs = self.calibrator_sigmoid.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        else:
            cal_probs = raw_probs
            
        return np.clip(cal_probs, 1e-6, 1.0 - 1e-6)

    def optimize_decision_threshold(
        self,
        probs: np.ndarray,
        nominal: np.ndarray,
        cost_per_review: float = 25000.0,
        multiplier: float = 1.0
    ) -> np.ndarray:
        """Apply cost-optimal decision rule:
        Flag if expected fraud loss > review cost.
        P(fraud) * nominal > multiplier * cost_per_review
        """
        expected_fraud_loss = probs * nominal
        threshold = multiplier * cost_per_review
        return (expected_fraud_loss > threshold).astype(int)
