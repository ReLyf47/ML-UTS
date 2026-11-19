import os
import sys
# Ensure ML-UTS root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score
from sklearn.utils import resample as sk_resample

from kode.common.seeds import seed_everything
from kode.common.logger import get_logger
from kode.common.metrics import calculate_classification_metrics
from kode.common.figure_saver import save_figure
from kode.common.experiment_tracker import log_experiment
from kode.common.config import METRICS_DIR, PREDICTIONS_DIR, BASE_DIR
from kode.common.data_loader import get_classification_data
from kode.classification.features import ClassificationFeaturePipeline
from kode.classification.models import LogisticRegressionFraudBaseline, HistGradientBoostingFraudModel

logger = get_logger("ClassificationPipeline")

# Format string per metric key, used to build the per-experiment summary log line.
_METRIC_LOG_FORMATS = {
    "pr_auc": "PR-AUC={:.4f}",
    "roc_auc": "ROC-AUC={:.4f}",
    "brier_score": "Brier={:.4f}",
    "total_cost": "Total Cost=Rp {:,.0f}",
}


def _record_experiment(
    results: dict,
    result_key: str,
    log_label: str,
    log_metric_keys: tuple,
    metrics: dict,
    experiment_id: str | None = None,
    task: str = "Classification",
    model_name: str = "",
    features_used: list | None = None,
    split_type: str = "Temporal_JulAug2025",
    hyperparameters: dict | None = None,
    runtime_seconds: float = 0.0,
    notes: str = "",
) -> None:
    """Store metrics in `results`, log a one-line summary, and (optionally) persist via log_experiment.

    `log_experiment` is only called when `experiment_id` is given, so experiments that were
    never tracked before stay untracked.
    """
    results[result_key] = metrics
    summary = ", ".join(_METRIC_LOG_FORMATS[k].format(metrics[k]) for k in log_metric_keys)
    logger.info(f"{log_label}: {summary}")

    if experiment_id is not None:
        log_experiment(
            experiment_id=experiment_id,
            task=task,
            model_name=model_name,
            features_used=features_used if features_used is not None else [],
            split_type=split_type,
            hyperparameters=hyperparameters if hyperparameters is not None else {},
            metrics=metrics,
            runtime_seconds=runtime_seconds,
            notes=notes,
        )


def run_classification_pipeline() -> dict:
    seed_everything(42)
    logger.info("Starting Task 2: Fraud Detection Classification Pipeline...")
    
    # 1. Load raw data
    t0 = time.time()
    df_train, df_test = get_classification_data()
    logger.info(f"Loaded classification data: Train={df_train.shape}, Test={df_test.shape} in {time.time() - t0:.2f}s")
    
    # 2. Chronological Split Setup
    # Train: Months 1-6 (Jan 1, 2025 to Jun 30, 2025)
    # Val: Months 7-8 (Jul 1, 2025 to Aug 31, 2025) - exactly 2 months matching the test set
    df_train['waktu_dt'] = pd.to_datetime(df_train['waktu'])
    split_date = pd.Timestamp('2025-07-01')
    
    train_mask = df_train['waktu_dt'] < split_date
    val_mask = df_train['waktu_dt'] >= split_date
    
    df_tr_split = df_train[train_mask].copy().reset_index(drop=True)
    df_val_split = df_train[val_mask].copy().reset_index(drop=True)
    
    y_tr = df_tr_split['is_fraud'].values
    y_val = df_val_split['is_fraud'].values
    nominal_val = df_val_split['nominal'].values
    
    logger.info(f"Chronological split: Train (Jan-Jun 2025) = {len(df_tr_split):,} rows (Frauds: {y_tr.sum():,}) | "
                f"Val (Jul-Aug 2025) = {len(df_val_split):,} rows (Frauds: {y_val.sum():,})")
    
    # 3. Clean Feature Pipeline (Without policy flag)
    pipe_clean = ClassificationFeaturePipeline(include_policy_flag=False)
    pipe_clean.fit(df_tr_split)
    X_tr_clean, feat_names_clean = pipe_clean.transform(df_tr_split)
    X_val_clean, _ = pipe_clean.transform(df_val_split)
    
    cat_indices_clean = pipe_clean.categorical_indices
    logger.info(f"Engineered {len(feat_names_clean)} clean features. Categorical indices: {cat_indices_clean}")
    
    results = {}
    
    # ---------------- EXPERIMENT 1: Logistic Regression Baseline ----------------
    logger.info("Training Logistic Regression Baseline (Class-Weighted)...")
    t_start = time.time()
    lr_model = LogisticRegressionFraudBaseline(random_state=42)
    lr_model.fit(X_tr_clean, y_tr)
    probs_lr = lr_model.predict_proba(X_val_clean)
    metrics_lr = calculate_classification_metrics(y_val, probs_lr, nominal_val)
    dur_lr = time.time() - t_start
    _record_experiment(
        results, "Baseline_Logistic_Regression", "Logistic Regression",
        ("pr_auc", "roc_auc", "total_cost"), metrics_lr,
        experiment_id="CLF_EXP_01",
        model_name="LogisticRegressionBaseline",
        features_used=feat_names_clean,
        hyperparameters={"class_weight": "balanced"},
        runtime_seconds=dur_lr,
        notes="Class-weighted linear baseline"
    )
    
    # ---------------- EXPERIMENT 2: HistGradientBoosting (Unweighted) ----------------
    logger.info("Training HistGradientBoosting (Unweighted)...")
    t_start = time.time()
    hgb_unweighted = HistGradientBoostingFraudModel(
        class_weight=None,
        categorical_features=cat_indices_clean,
        random_state=42
    )
    hgb_unweighted.fit(X_tr_clean, y_tr)
    probs_unweighted = hgb_unweighted.predict_proba(X_val_clean, calibration='none')
    metrics_unweighted = calculate_classification_metrics(y_val, probs_unweighted, nominal_val)
    dur_unweighted = time.time() - t_start
    _record_experiment(
        results, "HistGBM_Unweighted", "HistGBM Unweighted",
        ("pr_auc", "total_cost"), metrics_unweighted
    )
    
    # ---------------- EXPERIMENT 3: HistGradientBoosting (Balanced) + Calibration ----------------
    logger.info("Training HistGradientBoosting (Balanced Class Weights) + Probability Calibration...")
    t_start = time.time()
    hgb_balanced = HistGradientBoostingFraudModel(
        max_iter=150,
        learning_rate=0.08,
        class_weight='balanced',
        categorical_features=cat_indices_clean,
        random_state=42
    )
    hgb_balanced.fit(X_tr_clean, y_tr)
    
    # Evaluate Raw Balanced
    probs_raw = hgb_balanced.predict_proba(X_val_clean, calibration='none')
    metrics_raw = calculate_classification_metrics(y_val, probs_raw, nominal_val)
    _record_experiment(
        results, "HistGBM_Balanced_Raw", "HistGBM Balanced (Raw)",
        ("pr_auc", "brier_score", "total_cost"), metrics_raw
    )
    
    # Calibrate on validation split
    hgb_balanced.calibrate(X_val_clean, y_val)
    
    # Sigmoid Calibration
    probs_sigmoid = hgb_balanced.predict_proba(X_val_clean, calibration='sigmoid')
    metrics_sigmoid = calculate_classification_metrics(y_val, probs_sigmoid, nominal_val)
    _record_experiment(
        results, "HistGBM_Balanced_Sigmoid", "HistGBM Balanced (Sigmoid Calibrated)",
        ("pr_auc", "brier_score", "total_cost"), metrics_sigmoid
    )
    
    # Isotonic Calibration (Primary candidate)
    probs_isotonic = hgb_balanced.predict_proba(X_val_clean, calibration='isotonic')
    metrics_isotonic = calculate_classification_metrics(y_val, probs_isotonic, nominal_val)
    dur_hgb = time.time() - t_start
    _record_experiment(
        results, "HistGBM_Balanced_Isotonic", "HistGBM Balanced (Isotonic Calibrated)",
        ("pr_auc", "brier_score", "total_cost"), metrics_isotonic,
        experiment_id="CLF_EXP_03_ISOTONIC",
        model_name="HistGradientBoosting_Isotonic",
        features_used=feat_names_clean,
        hyperparameters={"class_weight": "balanced", "calibration": "isotonic"},
        runtime_seconds=dur_hgb,
        notes="Primary recommended fraud model (Balanced + Isotonic calibration)"
    )
    
    # ---------------- EXPERIMENT 4: Model WITH flag_review_manual (Policy Ablation) ----------------
    logger.info("Training HistGradientBoosting WITH flag_review_manual (Policy Ablation)...")
    pipe_policy = ClassificationFeaturePipeline(include_policy_flag=True)
    pipe_policy.fit(df_tr_split)
    X_tr_pol, feat_names_pol = pipe_policy.transform(df_tr_split)
    X_val_pol, _ = pipe_policy.transform(df_val_split)
    cat_indices_pol = pipe_policy.categorical_indices
    
    hgb_policy = HistGradientBoostingFraudModel(
        class_weight='balanced',
        categorical_features=cat_indices_pol,
        random_state=42
    )
    hgb_policy.fit(X_tr_pol, y_tr)
    hgb_policy.calibrate(X_val_pol, y_val)
    probs_policy = hgb_policy.predict_proba(X_val_pol, calibration='isotonic')
    metrics_policy = calculate_classification_metrics(y_val, probs_policy, nominal_val)
    _record_experiment(
        results, "HistGBM_With_PolicyFlag", "HistGBM With Policy Flag",
        ("pr_auc", "total_cost"), metrics_policy
    )
    
    # ---------------- EXPERIMENT 5: Random Under-Sampling (Class Imbalance Strategy) ----------------
    logger.info("Training HistGradientBoosting with Random Under-Sampling (majority:minority = 10:1)...")
    t_start = time.time()
    pos_mask = y_tr == 1
    df_pos = df_tr_split[pos_mask]
    df_neg = df_tr_split[~pos_mask]
    n_neg_target = int(min(len(df_neg), 10 * len(df_pos)))
    df_neg_sampled = sk_resample(df_neg, n_samples=n_neg_target, replace=False, random_state=42)
    df_under = pd.concat([df_pos, df_neg_sampled], ignore_index=True)
    df_under = df_under.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    pipe_under = ClassificationFeaturePipeline(include_policy_flag=False)
    pipe_under.fit(df_under)
    X_tr_under, feat_names_under = pipe_under.transform(df_under)
    X_val_under, _ = pipe_under.transform(df_val_split)
    cat_idx_under = pipe_under.categorical_indices
    
    hgb_under = HistGradientBoostingFraudModel(
        class_weight=None,
        categorical_features=cat_idx_under,
        random_state=42
    )
    hgb_under.fit(X_tr_under, df_under['is_fraud'].values)
    probs_under = hgb_under.predict_proba(X_val_under, calibration='none')
    metrics_under = calculate_classification_metrics(y_val, probs_under, nominal_val)
    dur_under = time.time() - t_start
    _record_experiment(
        results, "HistGBM_RandomUnderSampler", "HistGBM (Random Under-Sampling 10:1)",
        ("pr_auc", "brier_score", "total_cost"), metrics_under,
        experiment_id="CLF_EXP_05_RESAMPLE",
        model_name="HistGradientBoosting_RandomUnderSampler",
        features_used=feat_names_under,
        hyperparameters={"resampling": "random_undersample_10_to_1", "class_weight": "none"},
        runtime_seconds=dur_under,
        notes="Third imbalance strategy: random under-sampling of majority class"
    )
    
    # ---------------- 4. COST OPTIMIZATION ANALYSIS ----------------
    logger.info("Conducting empirical financial cost optimization and threshold sweep...")
    multipliers = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    sweep_records = []
    
    for m in multipliers:
        flag_dec = hgb_balanced.optimize_decision_threshold(probs_isotonic, nominal_val, cost_per_review=25000.0, multiplier=m)
        m_eval = calculate_classification_metrics(y_val, probs_isotonic, nominal_val, flag_decision=flag_dec)
        sweep_records.append({
            "multiplier": m,
            "threshold_desc": f"{m:.1f}x (Rp {m * 25000:,.0f})",
            "flagged_count": m_eval['flagged_count'],
            "flagged_rate": m_eval['flagged_rate'],
            "fraud_recall": m_eval['fraud_recall'],
            "review_cost": m_eval['review_cost'],
            "missed_fraud_cost": m_eval['missed_fraud_cost'],
            "total_cost": m_eval['total_cost']
        })
        
    df_sweep = pd.DataFrame(sweep_records)
    best_row = df_sweep.loc[df_sweep['total_cost'].idxmin()]
    best_multiplier = float(best_row['multiplier'])
    logger.info(f"Optimal cost multiplier: {best_multiplier:.2f} with Total Cost=Rp {best_row['total_cost']:,.0f} (vs Flag Nothing=Rp {metrics_isotonic['cost_flag_nothing']:,.0f})")
    results["cost_sweep"] = sweep_records
    results["best_multiplier"] = best_multiplier
    
    # ---------------- 5. SUB-GROUP ERROR BREAKDOWN ----------------
    logger.info("Computing detailed error breakdown across kanal, kategori_merchant, and tipe_perangkat...")
    df_eval = df_val_split.copy()
    df_eval['prob'] = probs_isotonic
    df_eval['tandai'] = hgb_balanced.optimize_decision_threshold(probs_isotonic, nominal_val, multiplier=best_multiplier)
    
    # Breakdown by Kanal
    breakdown_kanal = df_eval.groupby('kanal').agg(
        n_trx=('is_fraud', 'count'),
        n_fraud=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean'),
        flagged_count=('tandai', 'sum'),
        avg_nominal=('nominal', 'mean')
    ).reset_index()
    breakdown_kanal['fraud_caught'] = [
        int(np.sum((df_eval['kanal'] == k) & (df_eval['tandai'] == 1) & (df_eval['is_fraud'] == 1)))
        for k in breakdown_kanal['kanal']
    ]
    breakdown_kanal['recall'] = breakdown_kanal['fraud_caught'] / np.maximum(1, breakdown_kanal['n_fraud'])
    
    # Breakdown by Device
    breakdown_device = df_eval.groupby('tipe_perangkat').agg(
        n_trx=('is_fraud', 'count'),
        n_fraud=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean'),
        flagged_count=('tandai', 'sum')
    ).reset_index()
    
    results["breakdown_kanal"] = breakdown_kanal.to_dict(orient="records")
    results["breakdown_device"] = breakdown_device.to_dict(orient="records")
    
    # Breakdown by Merchant Category (kategori_merchant)
    breakdown_merchant = df_eval.groupby('kategori_merchant').agg(
        n_trx=('is_fraud', 'count'),
        n_fraud=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean'),
        flagged_count=('tandai', 'sum'),
        avg_nominal=('nominal', 'mean')
    ).reset_index()
    breakdown_merchant['fraud_caught'] = [
        int(np.sum((df_eval['kategori_merchant'] == k) & (df_eval['tandai'] == 1) & (df_eval['is_fraud'] == 1)))
        for k in breakdown_merchant['kategori_merchant']
    ]
    breakdown_merchant['recall'] = breakdown_merchant['fraud_caught'] / np.maximum(1, breakdown_merchant['n_fraud'])
    breakdown_merchant = breakdown_merchant.sort_values('n_trx', ascending=False).reset_index(drop=True)
    
    results["breakdown_merchant"] = breakdown_merchant.to_dict(orient="records")
    
    # ---------------- 5b. PATTERN SHIFT ANALYSIS (fraud over time) ----------------
    logger.info("Analyzing fraud pattern shift across months and validation-window drift...")
    df_time = df_train.copy()
    df_time['month'] = pd.to_datetime(df_time['waktu']).dt.strftime('%Y-%m')
    monthly_stats = df_time.groupby('month').agg(
        n_trx=('is_fraud', 'count'),
        n_fraud=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean')
    ).reset_index()
    channel_share = pd.crosstab(df_time['month'], df_time['kanal'], normalize='index').round(4)
    if 'kartu' in channel_share.columns:
        monthly_stats['kartu_share'] = channel_share['kartu'].values
    device_share = pd.crosstab(df_time['month'], df_time['tipe_perangkat'], normalize='index').round(4)
    if 'android' in device_share.columns:
        monthly_stats['android_share'] = device_share['android'].values
    
    # Model performance drift inside the validation window (July vs August 2025)
    val_month = pd.to_datetime(df_val_split['waktu']).dt.month.values
    monthly_pr_auc = {}
    for mth in sorted(set(val_month)):
        m_mask = val_month == mth
        monthly_pr_auc[int(mth)] = {
            "n_trx": int(m_mask.sum()),
            "n_fraud": int(y_val[m_mask].sum()),
            "pr_auc": float(average_precision_score(y_val[m_mask], probs_isotonic[m_mask])) if y_val[m_mask].sum() > 0 else None
        }
    logger.info(f"Monthly PR-AUC drift: {monthly_pr_auc}")
    
    results["pattern_shift"] = {
        "monthly_fraud_stats": monthly_stats.to_dict(orient="records"),
        "validation_window_monthly_pr_auc": monthly_pr_auc,
        "adaptation": (
            "Model dilatih dengan bobot kelas seimbang dan divalidasi secara temporal (Jan-Jun -> Jul-Agu); "
            "probabilitas dikalibrasi ulang secara isotonic pada jendela validasi out-of-time sehingga "
            "pergeseran pola fraud antar-bulan diserap oleh kalibrasi, bukan oleh overwrite fitur."
        )
    }
    
    # ---------------- 6. GENERATE DIAGNOSTIC FIGURES ----------------
    logger.info("Generating publication-grade diagnostic figures for classification...")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # Figure 1: PR Curve & ROC Curve
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))
    prec_iso, rec_iso, _ = precision_recall_curve(y_val, probs_isotonic)
    prec_lr, rec_lr, _ = precision_recall_curve(y_val, probs_lr)
    ax1a.plot(rec_iso, prec_iso, 'b-', linewidth=2, label=f'HistGBM Isotonic (PR-AUC={metrics_isotonic["pr_auc"]:.3f})')
    ax1a.plot(rec_lr, prec_lr, 'r--', linewidth=1.5, label=f'Logistic Regression (PR-AUC={metrics_lr["pr_auc"]:.3f})')
    ax1a.axhline(y_val.mean(), color='grey', linestyle=':', label=f'No Skill Baseline ({y_val.mean():.4f})')
    ax1a.set_xlabel("Recall")
    ax1a.set_ylabel("Precision")
    ax1a.set_title("Precision-Recall Curves (Fraud Detection)", fontweight="bold")
    ax1a.legend(loc="upper right")
    
    fpr_iso, tpr_iso, _ = roc_curve(y_val, probs_isotonic)
    fpr_lr, tpr_lr, _ = roc_curve(y_val, probs_lr)
    ax1b.plot(fpr_iso, tpr_iso, 'b-', linewidth=2, label=f'HistGBM Isotonic (ROC-AUC={metrics_isotonic["roc_auc"]:.3f})')
    ax1b.plot(fpr_lr, tpr_lr, 'r--', linewidth=1.5, label=f'Logistic Regression (ROC-AUC={metrics_lr["roc_auc"]:.3f})')
    ax1b.plot([0, 1], [0, 1], 'k:')
    ax1b.set_xlabel("False Positive Rate")
    ax1b.set_ylabel("True Positive Rate")
    ax1b.set_title("ROC Curves (Fraud Detection)", fontweight="bold")
    ax1b.legend(loc="lower right")
    save_figure(fig1, "fig_classification_pr_roc_curves.png")
    
    # Figure 2: Reliability Diagram (Probability Calibration)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    prob_true_raw, prob_pred_raw = calibration_curve(y_val, probs_raw, n_bins=10)
    prob_true_sig, prob_pred_sig = calibration_curve(y_val, probs_sigmoid, n_bins=10)
    prob_true_iso, prob_pred_iso = calibration_curve(y_val, probs_isotonic, n_bins=10)
    ax2.plot([0, 1], [0, 1], 'k:', label='Perfectly Calibrated')
    ax2.plot(prob_pred_raw, prob_true_raw, 's--', color='gray', label=f'Uncalibrated (Brier={metrics_raw["brier_score"]:.4f})')
    ax2.plot(prob_pred_sig, prob_true_sig, 'o-', color='purple', label=f'Sigmoid/Platt (Brier={metrics_sigmoid["brier_score"]:.4f})')
    ax2.plot(prob_pred_iso, prob_true_iso, '^-', color='blue', linewidth=2, label=f'Isotonic (Brier={metrics_isotonic["brier_score"]:.4f})')
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Fraction of Positives (Empirical Fraud Rate)")
    ax2.set_title("Reliability Diagram: Calibration Curves Comparison", fontweight="bold")
    ax2.legend()
    save_figure(fig2, "fig_classification_calibration_curves.png")
    
    # Figure 3: Cost Optimization Curve
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    costs_million = [r['total_cost'] / 1e6 for r in sweep_records]
    x_mults = [r['multiplier'] for r in sweep_records]
    ax3.plot(x_mults, costs_million, 'b-o', linewidth=2, label='Decision Rule Total Cost')
    ax3.axhline(metrics_isotonic['cost_flag_nothing'] / 1e6, color='red', linestyle='--', label=f'Flag Nothing (Rp {metrics_isotonic["cost_flag_nothing"]/1e6:.1f}M)')
    ax3.plot(best_multiplier, best_row['total_cost'] / 1e6, 'g*', markersize=14, label=f'Optimal Multiplier ({best_multiplier:.1f}x: Rp {best_row["total_cost"]/1e6:.1f}M)')
    ax3.set_xlabel("Decision Rule Multiplier (Threshold = Multiplier * Rp 25,000)")
    ax3.set_ylabel("Total Business Cost (Million IDR)")
    ax3.set_title("Financial Cost Optimization Curve on Holdout Period (Jul-Aug 2025)", fontweight="bold")
    ax3.legend()
    save_figure(fig3, "fig_classification_cost_curve.png")
    
    # ---------------- 7. SERIALIZE METRICS ----------------
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = METRICS_DIR / "classification_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved classification metrics to {metrics_path}")
    
    # ---------------- 8. FIT FINAL MODEL ON ALL 403K ROWS & GENERATE SUBMISSION ----------------
    logger.info("Fitting Final Production Fraud Classifier on full 403,077 transactions...")
    pipe_full = ClassificationFeaturePipeline(include_policy_flag=False)
    pipe_full.fit(df_train)
    X_full, feat_names_full = pipe_full.transform(df_train)
    y_full = df_train['is_fraud'].values
    cat_indices_full = pipe_full.categorical_indices
    
    final_hgb = HistGradientBoostingFraudModel(
        max_iter=180,
        learning_rate=0.08,
        class_weight='balanced',
        categorical_features=cat_indices_full,
        random_state=42
    )
    final_hgb.fit(X_full, y_full)
    # Calibrate using the fitted calibrator from the holdout validation
    final_hgb.calibrator_isotonic = hgb_balanced.calibrator_isotonic
    
    # Generate test predictions
    logger.info("Generating predictions for hidden test dataset (106,674 transactions)...")
    X_test, _ = pipe_full.transform(df_test)
    test_probs = final_hgb.predict_proba(X_test, calibration='isotonic')
    test_flags = final_hgb.optimize_decision_threshold(test_probs, df_test['nominal'].values, multiplier=best_multiplier)
    
    df_sub = pd.DataFrame({
        'id_transaksi': df_test['id_transaksi'],
        'prob': np.round(test_probs, 6),
        'tandai': test_flags.astype(int)
    })
    
    # Quality Assurance Checks
    assert len(df_sub) == 106674, f"Expected 106,674 predictions, got {len(df_sub)}"
    assert ((df_sub['prob'] >= 0.0) & (df_sub['prob'] <= 1.0)).all(), "Probabilities out of bounds [0, 1]!"
    assert set(df_sub['tandai'].unique()).issubset({0, 1}), "Tandai contains non-binary values!"
    assert df_sub['prob'].isnull().sum() == 0, "Missing values in prob!"
    assert df_sub['tandai'].isnull().sum() == 0, "Missing values in tandai!"
    
    sub_path = BASE_DIR / "prediksi_klasifikasi.csv"
    sub_path_out = PREDICTIONS_DIR / "prediksi_klasifikasi.csv"
    os.makedirs(os.path.dirname(sub_path_out), exist_ok=True)
    df_sub.to_csv(sub_path, index=False)
    df_sub.to_csv(sub_path_out, index=False)
    logger.info(f"Successfully generated and validated {sub_path} and {sub_path_out} ({len(df_sub):,} rows)")
    logger.info(f"Test Fraud Predictions: Flagged={df_sub['tandai'].sum():,} ({df_sub['tandai'].mean()*100:.2f}%), "
                f"Mean Prob={df_sub['prob'].mean():.4f}, Max Prob={df_sub['prob'].max():.4f}")
    
    return results

if __name__ == "__main__":
    run_classification_pipeline()
