import numpy as np
from sklearn.metrics import (
    mean_pinball_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    auc,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

# ----------------- REGRESSION METRICS -----------------

def calculate_pinball_loss(y_true, y_pred, alpha: float) -> float:
    """Compute pinball / quantile loss for a given quantile alpha."""
    return float(mean_pinball_loss(y_true, y_pred, alpha=alpha))

def calculate_coverage(y_true, y_pred_upper) -> float:
    """Compute empirical coverage rate: fraction of y_true <= y_pred_upper."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred_upper)
    return float(np.mean(y_true_arr <= y_pred_arr))

def evaluate_regression_predictions(y_true, p50_pred, p90_pred) -> dict:
    """Comprehensive evaluation of P50 and P90 quantile regression."""
    p50_arr = np.asarray(p50_pred)
    p90_arr = np.asarray(p90_pred)
    y_arr = np.asarray(y_true)
    
    # Monotonicity check
    crossing_violations = int(np.sum(p90_arr < p50_arr))
    crossing_rate = float(crossing_violations / len(y_arr))
    
    p50_loss = calculate_pinball_loss(y_arr, p50_arr, 0.50)
    p90_loss = calculate_pinball_loss(y_arr, p90_arr, 0.90)
    p90_cov = calculate_coverage(y_arr, p90_arr)
    p50_mae = float(mean_absolute_error(y_arr, p50_arr))
    
    return {
        "pinball_loss_p50": p50_loss,
        "pinball_loss_p90": p90_loss,
        "coverage_p90": p90_cov,
        "mae_p50": p50_mae,
        "crossing_violations": crossing_violations,
        "crossing_rate": crossing_rate
    }

# ----------------- CLASSIFICATION METRICS -----------------

def calculate_pr_auc(y_true, y_prob) -> float:
    """Compute Average Precision / PR-AUC."""
    return float(average_precision_score(y_true, y_prob))

def calculate_classification_metrics(y_true, y_prob, nominal, flag_decision=None, cost_per_review=25000.0) -> dict:
    """Evaluate fraud classification ranking, calibration, and business financial cost."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    nominal_arr = np.asarray(nominal)
    
    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    
    # Decision flags: if not provided, compute using optimal expected cost rule
    if flag_decision is None:
        flag_decision = (y_prob_arr * nominal_arr > cost_per_review).astype(int)
    else:
        flag_decision = np.asarray(flag_decision).astype(int)
        
    flagged_count = int(np.sum(flag_decision))
    flagged_rate = float(flagged_count / len(y_true_arr))
    
    # Cost calculation:
    # Missed fraud: true fraud that was not flagged (flag_decision == 0 and y_true == 1)
    missed_fraud_mask = (flag_decision == 0) & (y_true_arr == 1)
    missed_fraud_count = int(np.sum(missed_fraud_mask))
    missed_fraud_cost = float(np.sum(nominal_arr[missed_fraud_mask]))
    
    review_cost = float(flagged_count * cost_per_review)
    total_cost = missed_fraud_cost + review_cost
    
    # Baseline comparison 1: Flag nothing
    cost_flag_nothing = float(np.sum(nominal_arr[y_true_arr == 1]))
    # Baseline comparison 2: Flag everything
    cost_flag_everything = float(len(y_true_arr) * cost_per_review)
    
    # Caught fraud stats
    caught_fraud_mask = (flag_decision == 1) & (y_true_arr == 1)
    caught_fraud_count = int(np.sum(caught_fraud_mask))
    total_fraud = int(np.sum(y_true_arr == 1))
    fraud_recall = float(caught_fraud_count / total_fraud) if total_fraud > 0 else 0.0
    
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "flagged_count": flagged_count,
        "flagged_rate": flagged_rate,
        "caught_fraud_count": caught_fraud_count,
        "missed_fraud_count": missed_fraud_count,
        "fraud_recall": fraud_recall,
        "review_cost": review_cost,
        "missed_fraud_cost": missed_fraud_cost,
        "total_cost": total_cost,
        "cost_flag_nothing": cost_flag_nothing,
        "cost_flag_everything": cost_flag_everything,
        "savings_vs_flag_nothing": cost_flag_nothing - total_cost
    }

# ----------------- CLUSTERING METRICS -----------------

def evaluate_clustering(X, labels, sample_size=10000, random_state=42) -> dict:
    """Compute intrinsic cluster validity metrics, excluding noise (-1) if present."""
    X_arr = np.asarray(X)
    labels_arr = np.asarray(labels)
    
    n_total = len(labels_arr)
    n_anomalies = int(np.sum(labels_arr == -1))
    anomaly_rate = float(n_anomalies / n_total)
    
    # Filter noise for geometric cluster metrics
    core_mask = labels_arr != -1
    n_clusters = len(np.unique(labels_arr[core_mask]))
    
    if n_clusters < 2:
        return {
            "n_clusters": n_clusters,
            "n_anomalies": n_anomalies,
            "anomaly_rate": anomaly_rate,
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None
        }
        
    X_core = X_arr[core_mask]
    labels_core = labels_arr[core_mask]
    
    # Subsampled silhouette for efficiency on 120k rows
    if len(X_core) > sample_size:
        np.random.seed(random_state)
        sample_idx = np.random.choice(len(X_core), size=sample_size, replace=False)
        sil = float(silhouette_score(X_core[sample_idx], labels_core[sample_idx]))
    else:
        sil = float(silhouette_score(X_core, labels_core))
        
    db = float(davies_bouldin_score(X_core, labels_core))
    ch = float(calinski_harabasz_score(X_core, labels_core))
    
    return {
        "n_clusters": n_clusters,
        "n_anomalies": n_anomalies,
        "anomaly_rate": anomaly_rate,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch
    }
