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
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance

from kode.common.seeds import seed_everything
from kode.common.logger import get_logger
from kode.common.metrics import evaluate_regression_predictions, calculate_pinball_loss, calculate_coverage
from kode.common.figure_saver import save_figure
from kode.common.experiment_tracker import log_experiment
from kode.common.config import METRICS_DIR, PREDICTIONS_DIR, BASE_DIR
from kode.common.data_loader import get_regression_data
from kode.regression.features import RegressionFeaturePipeline
from kode.regression.baselines import GlobalMedianBaseline, ServiceMedianBaseline
from kode.regression.models import HistGradientBoostingQuantileModel, LogRidgeQuantileModel

logger = get_logger("RegressionPipeline")

def run_regression_pipeline():
    seed_everything(42)
    logger.info("Starting Task 1: Delivery Duration Regression Pipeline...")
    
    # 1. Load raw data
    t0 = time.time()
    df_train, df_test = get_regression_data()
    logger.info(f"Loaded regression data: Train={df_train.shape}, Test={df_test.shape} in {time.time() - t0:.2f}s")
    
    # 2. Chronological Split Setup
    # Train: Months 1-7 (Jan 1, 2025 to Jul 31, 2025)
    # Val: Month 8 (Aug 1, 2025 to Aug 31, 2025)
    df_train['tgl_dt'] = pd.to_datetime(df_train['tgl_pickup'])
    split_date = pd.Timestamp('2025-08-01')
    
    train_mask = df_train['tgl_dt'] < split_date
    val_mask = df_train['tgl_dt'] >= split_date
    
    df_tr_split = df_train[train_mask].copy().reset_index(drop=True)
    df_val_split = df_train[val_mask].copy().reset_index(drop=True)
    
    y_tr = df_tr_split['durasi_jam'].values
    y_val = df_val_split['durasi_jam'].values
    
    logger.info(f"Chronological split: Train (Jan-Jul 2025) = {len(df_tr_split):,} rows | Val (Aug 2025) = {len(df_val_split):,} rows")
    
    # 3. Clean Feature Pipeline (Leakage Excluded)
    pipe_clean = RegressionFeaturePipeline(include_leakage_feature=False)
    pipe_clean.fit(df_tr_split)
    X_tr_clean, feat_names_clean = pipe_clean.transform(df_tr_split)
    X_val_clean, _ = pipe_clean.transform(df_val_split)
    
    # Categorical indices in feature matrix for HistGradientBoosting
    cat_indices = [i for i, col in enumerate(feat_names_clean) if col.endswith('_cat')]
    logger.info(f"Engineered {len(feat_names_clean)} clean features. Categorical indices: {cat_indices}")
    
    results = {}
    
    # ---------------- EXPERIMENT 1: Global Median Baseline ----------------
    t_start = time.time()
    base_global = GlobalMedianBaseline()
    base_global.fit(y_tr)
    p50_bg, p90_bg = base_global.predict(len(y_val))
    metrics_bg = evaluate_regression_predictions(y_val, p50_bg, p90_bg)
    dur_bg = time.time() - t_start
    results["Baseline_Global_Median"] = metrics_bg
    logger.info(f"Baseline Global Median: P50 Pinball={metrics_bg['pinball_loss_p50']:.3f}, P90 Pinball={metrics_bg['pinball_loss_p90']:.3f}, Coverage={metrics_bg['coverage_p90']*100:.2f}%")
    log_experiment(
        experiment_id="REG_EXP_01",
        task="Regression",
        model_name="GlobalMedianBaseline",
        features_used=["None"],
        split_type="Temporal_Aug2025",
        hyperparameters={"type": "median"},
        metrics=metrics_bg,
        runtime_seconds=dur_bg,
        notes="Non-parametric global baseline"
    )
    
    # ---------------- EXPERIMENT 2: Service Median Baseline ----------------
    t_start = time.time()
    base_service = ServiceMedianBaseline()
    base_service.fit(df_tr_split['layanan'], y_tr)
    p50_bs, p90_bs = base_service.predict(df_val_split['layanan'])
    metrics_bs = evaluate_regression_predictions(y_val, p50_bs, p90_bs)
    dur_bs = time.time() - t_start
    results["Baseline_Service_Median"] = metrics_bs
    logger.info(f"Baseline Service Median: P50 Pinball={metrics_bs['pinball_loss_p50']:.3f}, P90 Pinball={metrics_bs['pinball_loss_p90']:.3f}, Coverage={metrics_bs['coverage_p90']*100:.2f}%")
    log_experiment(
        experiment_id="REG_EXP_02",
        task="Regression",
        model_name="ServiceMedianBaseline",
        features_used=["layanan"],
        split_type="Temporal_Aug2025",
        hyperparameters={"type": "service_conditional"},
        metrics=metrics_bs,
        runtime_seconds=dur_bs,
        notes="Service-tier conditional baseline"
    )
    
    # ---------------- EXPERIMENT 3: Log-Ridge Quantile Model ----------------
    t_start = time.time()
    ridge_model = LogRidgeQuantileModel(alpha=10.0)
    ridge_model.fit(X_tr_clean, y_tr)
    p50_ridge, p90_ridge = ridge_model.predict(X_val_clean)
    metrics_ridge = evaluate_regression_predictions(y_val, p50_ridge, p90_ridge)
    dur_ridge = time.time() - t_start
    results["Log_Ridge_Quantile"] = metrics_ridge
    logger.info(f"Log-Ridge Quantile: P50 Pinball={metrics_ridge['pinball_loss_p50']:.3f}, P90 Pinball={metrics_ridge['pinball_loss_p90']:.3f}, Coverage={metrics_ridge['coverage_p90']*100:.2f}%")
    log_experiment(
        experiment_id="REG_EXP_03",
        task="Regression",
        model_name="LogRidgeQuantileModel",
        features_used=feat_names_clean,
        split_type="Temporal_Aug2025",
        hyperparameters={"alpha": 10.0, "transform": "log1p"},
        metrics=metrics_ridge,
        runtime_seconds=dur_ridge,
        notes="Linear log-transformed quantile model"
    )
    
    # ---------------- EXPERIMENT 4: HistGradientBoosting Quantile Model (Clean) ----------------
    t_start = time.time()
    hgb_model = HistGradientBoostingQuantileModel(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        categorical_features=cat_indices,
        random_state=42
    )
    logger.info("Fitting HistGradientBoosting Quantile Regressor (P50 and P90)...")
    hgb_model.fit(X_tr_clean, y_tr)
    
    # Raw predictions
    p50_hgb_raw, p90_hgb_raw = hgb_model.predict(X_val_clean, apply_conformal=False)
    metrics_hgb_raw = evaluate_regression_predictions(y_val, p50_hgb_raw, p90_hgb_raw)
    logger.info(f"HistGBM Raw: P50 Pinball={metrics_hgb_raw['pinball_loss_p50']:.3f}, P90 Pinball={metrics_hgb_raw['pinball_loss_p90']:.3f}, Coverage={metrics_hgb_raw['coverage_p90']*100:.2f}%")
    
    # Calibrate conformal adjustment on validation split
    conformal_adj = hgb_model.calibrate_conformal(X_val_clean, y_val, target_coverage=0.90)
    logger.info(f"Conformal P90 Adjustment: +{conformal_adj:.3f} hours")
    
    p50_hgb_cal, p90_hgb_cal = hgb_model.predict(X_val_clean, apply_conformal=True)
    metrics_hgb_cal = evaluate_regression_predictions(y_val, p50_hgb_cal, p90_hgb_cal)
    dur_hgb = time.time() - t_start
    results["HistGBM_Quantile_Clean"] = metrics_hgb_cal
    logger.info(f"HistGBM Calibrated: P50 Pinball={metrics_hgb_cal['pinball_loss_p50']:.3f}, P90 Pinball={metrics_hgb_cal['pinball_loss_p90']:.3f}, Coverage={metrics_hgb_cal['coverage_p90']*100:.2f}%")
    
    log_experiment(
        experiment_id="REG_EXP_04",
        task="Regression",
        model_name="HistGradientBoostingQuantileModel",
        features_used=feat_names_clean,
        split_type="Temporal_Aug2025",
        hyperparameters={"max_iter": 150, "learning_rate": 0.08, "conformal_adj": conformal_adj},
        metrics=metrics_hgb_cal,
        runtime_seconds=dur_hgb,
        notes="Primary recommended model (Leakage excluded, conformal calibrated)"
    )
    
    # ---------------- EXPERIMENT 5: Leakage Ablation Study (With jumlah_scan) ----------------
    logger.info("Running Leakage Ablation Study: Training HistGBM WITH jumlah_scan...")
    t_start = time.time()
    pipe_leak = RegressionFeaturePipeline(include_leakage_feature=True)
    pipe_leak.fit(df_tr_split)
    X_tr_leak, feat_names_leak = pipe_leak.transform(df_tr_split)
    X_val_leak, _ = pipe_leak.transform(df_val_split)
    cat_indices_leak = [i for i, col in enumerate(feat_names_leak) if col.endswith('_cat')]
    
    hgb_leak = HistGradientBoostingQuantileModel(
        max_iter=150,
        learning_rate=0.08,
        categorical_features=cat_indices_leak,
        random_state=42
    )
    hgb_leak.fit(X_tr_leak, y_tr)
    p50_leak, p90_leak = hgb_leak.predict(X_val_leak, apply_conformal=False)
    metrics_leak = evaluate_regression_predictions(y_val, p50_leak, p90_leak)
    dur_leak = time.time() - t_start
    results["HistGBM_With_JumlahScan_Leakage"] = metrics_leak
    logger.info(f"HistGBM With Leakage: P50 Pinball={metrics_leak['pinball_loss_p50']:.3f}, P90 Pinball={metrics_leak['pinball_loss_p90']:.3f}, Coverage={metrics_leak['coverage_p90']*100:.2f}%")
    
    log_experiment(
        experiment_id="REG_EXP_05_ABLATION",
        task="Regression",
        model_name="HistGBM_Leakage_Ablation",
        features_used=feat_names_leak,
        split_type="Temporal_Aug2025",
        hyperparameters={"leakage_feature": "jumlah_scan"},
        metrics=metrics_leak,
        runtime_seconds=dur_leak,
        notes="Ablation experiment demonstrating post-outcome leakage"
    )
    
    # ---------------- EXPERIMENT 6: Random 5-Fold CV vs Temporal Validation ----------------
    logger.info("Comparing Random 5-Fold CV against Temporal Split...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Fast 5-fold evaluation on a representative 50k subsample to compare variance and optimism
    df_sub = df_train.sample(n=50000, random_state=42).reset_index(drop=True)
    p_sub = RegressionFeaturePipeline(include_leakage_feature=False)
    p_sub.fit(df_sub)
    X_sub, feat_sub = p_sub.transform(df_sub)
    y_sub = df_sub['durasi_jam'].values
    cat_idx_sub = [i for i, c in enumerate(feat_sub) if c.endswith('_cat')]
    
    cv_p50_losses, cv_p90_losses, cv_coverages = [], [], []
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_sub)):
        m_fold = HistGradientBoostingQuantileModel(max_iter=80, learning_rate=0.1, categorical_features=cat_idx_sub, random_state=42)
        m_fold.fit(X_sub[trn_idx], y_sub[trn_idx])
        p50_f, p90_f = m_fold.predict(X_sub[val_idx], apply_conformal=False)
        met_f = evaluate_regression_predictions(y_sub[val_idx], p50_f, p90_f)
        cv_p50_losses.append(met_f['pinball_loss_p50'])
        cv_p90_losses.append(met_f['pinball_loss_p90'])
        cv_coverages.append(met_f['coverage_p90'])
        
    results["Random_5Fold_CV"] = {
        "pinball_loss_p50_mean": float(np.mean(cv_p50_losses)),
        "pinball_loss_p50_std": float(np.std(cv_p50_losses)),
        "pinball_loss_p90_mean": float(np.mean(cv_p90_losses)),
        "pinball_loss_p90_std": float(np.std(cv_p90_losses)),
        "coverage_p90_mean": float(np.mean(cv_coverages))
    }
    logger.info(f"Random 5-Fold CV Results: P50 Pinball={np.mean(cv_p50_losses):.3f} +/- {np.std(cv_p50_losses):.3f}, P90 Pinball={np.mean(cv_p90_losses):.3f}")
    
    # ---------------- 4. SUB-GROUP ERROR ANALYSIS ----------------
    logger.info("Computing detailed error breakdown across layanan, pulau_tujuan, and mitra_kurir...")
    df_eval = df_val_split.copy()
    df_eval['p50_pred'] = p50_hgb_cal
    df_eval['p90_pred'] = p90_hgb_cal
    df_eval['abs_error_p50'] = np.abs(df_eval['durasi_jam'] - df_eval['p50_pred'])
    df_eval['covered_p90'] = (df_eval['durasi_jam'] <= df_eval['p90_pred']).astype(int)
    
    breakdown_layanan = df_eval.groupby('layanan').agg(
        n_samples=('durasi_jam', 'count'),
        actual_median=('durasi_jam', 'median'),
        pred_median=('p50_pred', 'median'),
        mae_p50=('abs_error_p50', 'mean'),
        coverage_p90=('covered_p90', 'mean')
    ).reset_index()
    
    breakdown_pulau = df_eval.groupby('pulau_tujuan').agg(
        n_samples=('durasi_jam', 'count'),
        actual_median=('durasi_jam', 'median'),
        pred_median=('p50_pred', 'median'),
        mae_p50=('abs_error_p50', 'mean'),
        coverage_p90=('covered_p90', 'mean')
    ).reset_index()
    
    breakdown_kurir = df_eval.groupby('mitra_kurir').agg(
        n_samples=('durasi_jam', 'count'),
        actual_median=('durasi_jam', 'median'),
        pred_median=('p50_pred', 'median'),
        mae_p50=('abs_error_p50', 'mean'),
        coverage_p90=('covered_p90', 'mean')
    ).reset_index()
    
    results["breakdown_layanan"] = breakdown_layanan.to_dict(orient="records")
    results["breakdown_pulau"] = breakdown_pulau.to_dict(orient="records")
    results["breakdown_kurir"] = breakdown_kurir.to_dict(orient="records")
    
    # ---------------- 5. GENERATE DIAGNOSTIC FIGURES ----------------
    logger.info("Generating publication-grade diagnostic figures...")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # Figure 1: Quantile Calibration Plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sample_vis = df_eval.sample(n=300, random_state=42).sort_values('durasi_jam').reset_index(drop=True)
    idx = np.arange(len(sample_vis))
    ax1.plot(idx, sample_vis['durasi_jam'], 'k.', alpha=0.6, label='Actual Duration (y)', markersize=5)
    ax1.plot(idx, sample_vis['p50_pred'], 'b-', linewidth=1.8, label='Predicted P50 (Median)')
    ax1.plot(idx, sample_vis['p90_pred'], 'r--', linewidth=1.8, label='Predicted P90 (Upper 90%)')
    ax1.fill_between(idx, sample_vis['p50_pred'], sample_vis['p90_pred'], color='red', alpha=0.15, label='P50-P90 Prediction Interval')
    ax1.set_ylim(0, min(300, sample_vis['durasi_jam'].max() + 20))
    ax1.set_xlabel("Sorted Package Sample Index")
    ax1.set_ylabel("Delivery Duration (Hours)")
    ax1.set_title("Quantile Interval Calibration (Actual vs P50 & P90 Predictions)", fontweight="bold")
    ax1.legend(loc="upper left")
    save_figure(fig1, "fig_regression_quantile_calibration.png")
    
    # Figure 2: Subgroup Coverage by Service Tier
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=breakdown_layanan, x='layanan', y='coverage_p90', palette='Blues_d', ax=ax2a)
    ax2a.axhline(0.90, color='red', linestyle='--', linewidth=1.5, label='Nominal 90% Target')
    ax2a.set_ylim(0.70, 1.0)
    ax2a.set_ylabel("Empirical P90 Coverage")
    ax2a.set_xlabel("Shipping Service Tier")
    ax2a.set_title("P90 Coverage Rate by Shipping Service", fontweight="bold")
    ax2a.legend()
    
    sns.barplot(data=breakdown_layanan, x='layanan', y='mae_p50', palette='Greens_d', ax=ax2b)
    ax2b.set_ylabel("P50 MAE (Hours)")
    ax2b.set_xlabel("Shipping Service Tier")
    ax2b.set_title("P50 Mean Absolute Error by Shipping Service", fontweight="bold")
    save_figure(fig2, "fig_regression_subgroup_service_performance.png")
    
    # Figure 3: Model Comparison Bar Chart
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    comp_models = ["Baseline_Global_Median", "Baseline_Service_Median", "Log_Ridge_Quantile", "HistGBM_Quantile_Clean"]
    comp_labels = ["Global Median", "Service Median", "Log-Ridge", "HistGBM (Clean)"]
    p50_losses = [results[m]['pinball_loss_p50'] for m in comp_models]
    p90_losses = [results[m]['pinball_loss_p90'] for m in comp_models]
    
    x = np.arange(len(comp_models))
    width = 0.35
    ax3.bar(x - width/2, p50_losses, width, label='P50 Pinball Loss', color='#1f77b4')
    ax3.bar(x + width/2, p90_losses, width, label='P90 Pinball Loss', color='#ff7f0e')
    ax3.set_xticks(x)
    ax3.set_xticklabels(comp_labels)
    ax3.set_ylabel("Pinball Loss (Lower is Better)")
    ax3.set_title("Model Benchmark Comparison on Out-of-Time Validation (Aug 2025)", fontweight="bold")
    ax3.legend()
    for i in range(len(comp_models)):
        ax3.text(x[i] - width/2, p50_losses[i] + 0.3, f"{p50_losses[i]:.2f}", ha='center', fontsize=9)
        ax3.text(x[i] + width/2, p90_losses[i] + 0.3, f"{p90_losses[i]:.2f}", ha='center', fontsize=9)
    save_figure(fig3, "fig_regression_model_comparison.png")
    
    # ---------------- 5b. MODEL INTERPRETATION: Permutation Importance & Partial Dependence ----------------
    logger.info("Computing permutation importance and partial dependence plots (validation subsample n=10k)...")
    from sklearn.inspection import permutation_importance, partial_dependence

    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(X_val_clean), size=min(10000, len(X_val_clean)), replace=False)
    X_val_sub = X_val_clean[sub_idx]
    y_val_sub = y_val[sub_idx]

    perm_result = permutation_importance(
        hgb_model.model_p50, X_val_sub, y_val_sub,
        n_repeats=5, random_state=42, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    top_perm_idx = np.argsort(perm_result.importances_mean)[::-1][:8]
    results["permutation_importance_top"] = [
        {
            "feature": feat_names_clean[i],
            "importance_mean": float(perm_result.importances_mean[i]),
            "importance_std": float(perm_result.importances_std[i]),
        }
        for i in top_perm_idx
    ]

    # Partial dependence for the 4 most important features (P50 quantile model)
    pd_feature_idx = [int(i) for i in top_perm_idx[:4]]
    fig_pd, ax_pd = plt.subplots(2, 2, figsize=(12, 8))
    for ax_i, f_i in zip(ax_pd.ravel(), pd_feature_idx):
        pd_res = partial_dependence(
            hgb_model.model_p50, X_val_sub, features=[f_i],
            kind="average", grid_resolution=30
        )
        ax_i.plot(pd_res["grid_values"][0], pd_res["average"][0], 'b-', linewidth=2)
        ax_i.set_xlabel(feat_names_clean[f_i])
        ax_i.set_ylabel("Partial dependence (P50, hours)")
        ax_i.set_title(f"PDP: {feat_names_clean[f_i]}", fontweight='bold')
    fig_pd.suptitle("Partial Dependence - Top-4 Features (P50 Quantile Model)", fontweight='bold', fontsize=13)
    save_figure(fig_pd, "fig_regression_partial_dependence.png")
    logger.info(f"Saved partial dependence figure for features: {[feat_names_clean[i] for i in pd_feature_idx]}")
    
    # ---------------- 6. SERIALIZE METRICS ----------------
    results["conformal_adjustment"] = float(conformal_adj)
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = METRICS_DIR / "regression_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved regression metrics to {metrics_path}")
    
    # ---------------- 7. TRAIN FINAL REGRESSION MODEL ON ALL 250K ROWS & GENERATE SUBMISSION ----------------
    logger.info("Training Final Production HistGBM Model on full 250,000 training records...")
    pipe_full = RegressionFeaturePipeline(include_leakage_feature=False)
    pipe_full.fit(df_train)
    X_full, feat_names_full = pipe_full.transform(df_train)
    y_full = df_train['durasi_jam'].values
    cat_indices_full = [i for i, c in enumerate(feat_names_full) if c.endswith('_cat')]
    
    final_hgb = HistGradientBoostingQuantileModel(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        categorical_features=cat_indices_full,
        random_state=42
    )
    final_hgb.fit(X_full, y_full)
    # Apply calibrated conformal margin derived from validation fold
    final_hgb.conformal_adjustment = conformal_adj
    
    # Generate test predictions
    logger.info("Generating predictions for hidden test dataset (60,000 packages)...")
    X_test, _ = pipe_full.transform(df_test)
    test_p50, test_p90 = final_hgb.predict(X_test, apply_conformal=True)
    
    df_sub = pd.DataFrame({
        'id_paket': df_test['id_paket'],
        'p50': np.round(test_p50, 2),
        'p90': np.round(test_p90, 2)
    })
    
    # Verification checks
    assert len(df_sub) == 60000, f"Expected 60,000 test predictions, got {len(df_sub)}"
    assert (df_sub['p90'] >= df_sub['p50']).all(), "P90 non-crossing condition violated!"
    assert df_sub['p50'].isnull().sum() == 0, "Missing values in P50!"
    assert df_sub['p90'].isnull().sum() == 0, "Missing values in P90!"
    
    sub_path = BASE_DIR / "prediksi_regresi.csv"
    sub_path_out = PREDICTIONS_DIR / "prediksi_regresi.csv"
    os.makedirs(os.path.dirname(sub_path_out), exist_ok=True)
    df_sub.to_csv(sub_path, index=False)
    df_sub.to_csv(sub_path_out, index=False)
    logger.info(f"Successfully generated and validated {sub_path} and {sub_path_out} ({len(df_sub):,} rows)")
    logger.info(f"Test P50 summary: min={df_sub['p50'].min():.2f}, median={df_sub['p50'].median():.2f}, mean={df_sub['p50'].mean():.2f}, max={df_sub['p50'].max():.2f}")
    logger.info(f"Test P90 summary: min={df_sub['p90'].min():.2f}, median={df_sub['p90'].median():.2f}, mean={df_sub['p90'].mean():.2f}, max={df_sub['p90'].max():.2f}")
    
    return results

if __name__ == "__main__":
    run_regression_pipeline()
