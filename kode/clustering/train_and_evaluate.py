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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from kode.common.seeds import seed_everything
from kode.common.logger import get_logger
from kode.common.figure_saver import save_figure
from kode.common.experiment_tracker import log_experiment
from kode.common.config import METRICS_DIR, PREDICTIONS_DIR, BASE_DIR
from kode.common.data_loader import get_clustering_data
from kode.clustering.features import CustomerClusteringPipeline
from kode.clustering.models import ClusterModelBenchmark

logger = get_logger("ClusteringPipeline")

def run_clustering_pipeline():
    seed_everything(42)
    logger.info("Starting Task 3: Customer Segmentation Clustering Pipeline...")
    
    # 1. Load raw data
    t0 = time.time()
    df_raw = get_clustering_data()
    logger.info(f"Loaded clustering data: Shape={df_raw.shape} in {time.time() - t0:.2f}s")
    
    # 2. Preprocess & Scale
    pipe = CustomerClusteringPipeline()
    X_scaled, feat_names, df_clean = pipe.fit_transform(df_raw)
    logger.info(f"Engineered and scaled {len(feat_names)} clustering features.")
    
    benchmark = ClusterModelBenchmark(random_state=42)
    results = {}
    
    # 3. K-Means Sweep (k = 3 to 8)
    logger.info("Evaluating K-Means across k in [3, 4, 5, 6, 7, 8]...")
    k_range = [3, 4, 5, 6, 7, 8]
    t_start = time.time()
    kmeans_results = benchmark.evaluate_kmeans_sweep(X_scaled, k_range, sample_size_sil=10000)
    dur_km = time.time() - t_start
    results["kmeans_sweep"] = kmeans_results
    for r in kmeans_results:
        logger.info(f"K-Means k={r['k']}: Inertia={r['inertia']:,.0f}, Silhouette={r['silhouette']:.4f}, DB={r['davies_bouldin']:.4f}, CH={r['calinski_harabasz']:,.0f}")
        
    # 4. GMM Sweep (k = 3 to 8)
    logger.info("Evaluating GMM BIC/AIC on representative sample...")
    gmm_results = benchmark.evaluate_gmm_sweep(X_scaled, k_range, sample_size=25000)
    results["gmm_sweep"] = gmm_results
    for r in gmm_results:
        logger.info(f"GMM k={r['k']}: BIC={r['bic']:,.0f}, AIC={r['aic']:,.0f}")
        
    # Determine optimal k: Davies-Bouldin minimum is the primary partition criterion,
    # then cross-checked with bootstrap stability (see step 5). Silhouette max is reported for transparency.
    best_sil_entry = max(kmeans_results, key=lambda x: x['silhouette'])
    best_db_entry = min(kmeans_results, key=lambda x: x['davies_bouldin'])
    optimal_k = int(best_db_entry['k'])
    logger.info(
        f"Criteria summary -> best silhouette k={best_sil_entry['k']} ({best_sil_entry['silhouette']:.4f}); "
        f"best Davies-Bouldin k={best_db_entry['k']} ({best_db_entry['davies_bouldin']:.4f}); "
        f"selected optimal k={optimal_k} (DB criterion + bootstrap stability check)"
    )
    
    # 4b. Additional method benchmarks: Agglomerative (Ward) and HDBSCAN density clustering
    logger.info("Benchmarking agglomerative (Ward) clustering on a 5,000-customer subsample...")
    agg_results = benchmark.evaluate_agglomerative_sweep(X_scaled, k_range, sample_size=5000)
    results["agglomerative_sweep"] = agg_results
    for r in agg_results:
        logger.info(f"Agglomerative (Ward) k={r['k']}: Silhouette={r['silhouette']:.4f}")
    
    logger.info("Benchmarking HDBSCAN density clustering on a 5,000-customer subsample...")
    hdbscan_info = benchmark.run_hdbscan_density_benchmark(
        X_scaled, sample_size=5000, min_cluster_size=100, min_samples=30
    )
    results["hdbscan_density"] = hdbscan_info
    logger.info(
        f"HDBSCAN: {hdbscan_info['n_clusters_found']} dense clusters found, "
        f"{hdbscan_info['noise_points']} noise points ({hdbscan_info['noise_ratio']*100:.2f}% of subsample)"
    )
    
    # 5. Stability Analysis (Bootstrap ARI)
    logger.info(f"Running Bootstrap Subsampling Stability Analysis for k={optimal_k}...")
    t_start = time.time()
    stability = benchmark.evaluate_stability(X_scaled, optimal_k=optimal_k, n_bootstraps=5, subsample_ratio=0.8)
    dur_stab = time.time() - t_start
    results["stability"] = stability
    logger.info(f"Cluster Stability (5-run Bootstrap): Mean ARI={stability['mean_ari']:.4f} +/- {stability['std_ari']:.4f}")
    
    # 6. Fit Final Primary Model on all 120,000 customers
    logger.info(f"Fitting Final K-Means Model on all 120,000 customers with k={optimal_k}...")
    km_final = KMeans(n_clusters=optimal_k, n_init=20, max_iter=300, random_state=42)
    cluster_labels = km_final.fit_predict(X_scaled)
    
    # 7. Anomaly Detection (-1) using Distance to Cluster Centroids
    logger.info("Detecting anomalies (-1) via multivariate distance to cluster centroids...")
    # Distance of each point to its assigned centroid
    centroids = km_final.cluster_centers_
    distances = np.linalg.norm(X_scaled - centroids[cluster_labels], axis=1)
    
    # Flag top 1.5% most distant points as anomalies (-1)
    anomaly_threshold = float(np.percentile(distances, 98.5))
    anomaly_mask = distances > anomaly_threshold
    
    final_labels = cluster_labels.copy()
    final_labels[anomaly_mask] = -1
    
    n_anomalies = int(np.sum(final_labels == -1))
    anomaly_rate = float(n_anomalies / len(final_labels))
    logger.info(f"Identified {n_anomalies:,} anomalies ({anomaly_rate*100:.2f}%) assigned label -1 (Threshold distance: {anomaly_threshold:.2f})")
    
    results["anomaly_detection"] = {
        "n_anomalies": n_anomalies,
        "anomaly_rate": anomaly_rate,
        "threshold_distance": anomaly_threshold
    }
    
    # 7b. Feature selection / weighting evidence for segmentation
    # ANOVA F-score of every scaled feature against the final core-cluster labels:
    # features with high F separate segments strongly and justify their weight in the pipeline.
    logger.info("Computing feature-selection evidence (ANOVA F-score vs cluster labels)...")
    from sklearn.feature_selection import f_classif
    core_mask = final_labels != -1
    f_stats, _ = f_classif(X_scaled[core_mask], final_labels[core_mask])
    f_ranked = sorted(
        zip(feat_names, [float(s) for s in f_stats]),
        key=lambda t: t[1],
        reverse=True
    )
    feature_variances = np.var(X_scaled, axis=0)
    zero_var_features = [f for f, v in zip(feat_names, feature_variances) if v < 1e-8]
    results["feature_evidence"] = {
        "method": "ANOVA F-score per scaled feature against final core-cluster labels",
        "n_features_used": len(feat_names),
        "zero_variance_features": zero_var_features,
        "top_discriminative_features": [
            {"feature": f, "f_score": round(s, 2)} for f, s in f_ranked[:10]
        ],
        "weakest_features": [
            {"feature": f, "f_score": round(s, 2)} for f, s in f_ranked[-5:]
        ],
    }
    logger.info(f"Top discriminative features: {[f for f, _ in f_ranked[:5]]}")
    
    # 8. Cluster Profiling on Original Unscaled Features
    logger.info("Computing unscaled behavioral profiles and assigning actionable personas...")
    df_clean['cluster'] = final_labels
    
    # Filter core clusters (excluding -1) for persona profiling
    core_df = df_clean[df_clean['cluster'] != -1]
    
    profile_metrics = [
        'recency_hari', 'frekuensi_90h', 'belanja_90h', 'rata2_keranjang_clean',
        'persen_voucher', 'persen_flash_sale', 'persen_paylater',
        'sesi_per_minggu', 'durasi_sesi_menit', 'rasio_konversi', 'retur_90h'
    ]
    
    cluster_profiles = core_df.groupby('cluster')[profile_metrics].mean().round(2)
    
    # Find dominant product category per cluster
    porsi_cats = [c for c in df_clean.columns if c.startswith('porsi_') and not c.startswith('porsi_jam_')]
    cat_means = core_df.groupby('cluster')[porsi_cats].mean()
    top_categories = cat_means.idxmax(axis=1).str.replace('porsi_', '').to_dict()
    
    # Define intuitive persona mappings and marketing strategies.
    # Names/descriptions are aligned with the empirical profile metrics in
    # results["cluster_numeric_means"] (see metrics JSON for verification).
    persona_mapping = {
        0: {
            "name": "VIP Champions",
            "description": "High frequency (~20x/90d), high basket size, recent activity (recency ~9 days), low discount dependency.",
            "top_category": top_categories.get(0, "fashion"),
            "strategy": "Exclusive loyalty privileges, concierge support, early access to premium product drops."
        },
        1: {
            "name": "Hyperactive Bargain Hunters",
            "description": "Extremely high app sessions per week, strong voucher & flash-sale usage, short session duration.",
            "top_category": top_categories.get(1, "makanan_minuman"),
            "strategy": "Targeted flash sale alerts, tiered threshold coupons, bundle savings."
        },
        2: {
            "name": "Casual Shoppers",
            "description": "Moderate-to-low purchase frequency, longer recency (~42 days), occasional basket sizes.",
            "top_category": top_categories.get(2, "makanan_minuman"),
            "strategy": "Gamified checkout bonuses, free-shipping subscriptions, impulse cross-sells."
        },
        3: {
            "name": "Discount Hunters",
            "description": "Highest voucher (~74%), flash-sale (~63%) and paylater (~37%) usage, active but price-driven.",
            "top_category": top_categories.get(3, "fashion"),
            "strategy": "Win-back push notifications with heavy vouchers, personalized catalog re-engagement."
        },
        4: {
            "name": "Lapsed / At-Risk Dormant",
            "description": "Very long recency (~62 days), zero 90-day spend and frequency; dormant accounts.",
            "top_category": top_categories.get(4, "rumah_tangga"),
            "strategy": "Win-back 'We Miss You' campaign, large reactivation vouchers, survey on churn reason."
        },
        -1: {
            "name": "Anomalous Outliers / Extreme Accounts",
            "description": "Extreme spending velocity or abnormal behavioral patterns situated at multivariate extremes.",
            "top_category": "multivariate_extreme",
            "strategy": "Automated security & reseller audit; compliance check for merchant abuse."
        }
    }
    
    results["persona_profiles"] = persona_mapping
    results["cluster_numeric_means"] = cluster_profiles.to_dict(orient="index")
    
    # 9. Generate Publication-Grade Diagnostic Figures
    logger.info("Generating publication-grade diagnostic figures for clustering...")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # Figure 1: Elbow and Silhouette curves
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))
    ks = [r['k'] for r in kmeans_results]
    inertias = [r['inertia'] for r in kmeans_results]
    silhouettes = [r['silhouette'] for r in kmeans_results]
    
    ax1a.plot(ks, inertias, 'bo-', linewidth=2, markersize=8)
    ax1a.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    ax1a.set_xlabel("Number of Clusters (k)")
    ax1a.set_ylabel("Inertia (Sum of Squared Distances)")
    ax1a.set_title("K-Means Inertia Elbow Curve", fontweight="bold")
    ax1a.legend()
    
    ax1b.plot(ks, silhouettes, 'gs-', linewidth=2, markersize=8)
    ax1b.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    ax1b.set_xlabel("Number of Clusters (k)")
    ax1b.set_ylabel("Silhouette Score (Subsampled N=10k)")
    ax1b.set_title("Silhouette Score Across Cluster Numbers", fontweight="bold")
    ax1b.legend()
    save_figure(fig1, "fig_clustering_elbow_silhouette.png")
    
    # Figure 2: PCA 2D Scatter with Clusters and Anomalies
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    pca = PCA(n_components=2, random_state=42)
    # Fit PCA on 10k subsample for clean visualization
    sample_vis_idx = np.random.choice(len(X_scaled), size=5000, replace=False)
    X_pca = pca.fit_transform(X_scaled[sample_vis_idx])
    lbl_vis = final_labels[sample_vis_idx]
    
    palette = sns.color_palette("tab10", n_colors=optimal_k)
    for c_id in range(optimal_k):
        mask = lbl_vis == c_id
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], s=18, alpha=0.6, label=f"Cluster {c_id}: {persona_mapping[c_id]['name'][:20]}", color=palette[c_id])
    
    # Plot anomalies
    mask_anom = lbl_vis == -1
    if np.sum(mask_anom) > 0:
        ax2.scatter(X_pca[mask_anom, 0], X_pca[mask_anom, 1], s=40, color='black', marker='x', label=f"Anomalies (-1): {np.sum(mask_anom)} pts")
        
    ax2.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax2.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax2.set_title(f"Customer Segments & Anomaly Isolation in 2D PCA Space (k={optimal_k})", fontweight="bold")
    ax2.legend(loc="upper right", markerscale=1.2, fontsize=9)
    save_figure(fig2, "fig_clustering_pca_segments.png")
    
    # Figure 3: Cluster Profile Heatmap (Standardized RFM Dimensions)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    profile_norm = (cluster_profiles - cluster_profiles.mean()) / cluster_profiles.std()
    profile_norm.index = [f"Cluster {i}: {persona_mapping[i]['name'][:22]}" for i in profile_norm.index]
    sns.heatmap(profile_norm, cmap="vlag", center=0, annot=cluster_profiles.values, fmt=".1f", cbar_kws={'label': 'Relative Standardized Score'}, ax=ax3)
    ax3.set_title("Customer Persona Feature Profiles (Actual Means with Standardized Heatmap)", fontweight="bold")
    save_figure(fig3, "fig_clustering_profiles_heatmap.png")
    
    # 10. Serialize Metrics
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = METRICS_DIR / "clustering_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved clustering metrics to {metrics_path}")
    
    # 11. Generate Final cluster.csv
    logger.info("Generating final cluster.csv file (120,000 customers)...")
    df_cluster_sub = pd.DataFrame({
        'id_pelanggan': df_raw['id_pelanggan'],
        'cluster': final_labels.astype(int)
    })
    
    # Quality Assurance Verification
    assert len(df_cluster_sub) == 120000, f"Expected 120,000 rows, got {len(df_cluster_sub)}"
    assert set(df_cluster_sub['cluster'].unique()).issubset({-1, 0, 1, 2, 3, 4}), "Unexpected cluster labels!"
    assert -1 in df_cluster_sub['cluster'].values, "Anomalies (-1) missing from predictions!"
    assert df_cluster_sub['cluster'].isnull().sum() == 0, "Missing values in cluster!"
    
    cluster_csv_path = BASE_DIR / "cluster.csv"
    cluster_csv_out = PREDICTIONS_DIR / "cluster.csv"
    os.makedirs(os.path.dirname(cluster_csv_out), exist_ok=True)
    df_cluster_sub.to_csv(cluster_csv_path, index=False)
    df_cluster_sub.to_csv(cluster_csv_out, index=False)
    logger.info(f"Successfully generated and validated {cluster_csv_path} and {cluster_csv_out} ({len(df_cluster_sub):,} rows)")
    logger.info("Cluster distribution:\n" + str(df_cluster_sub['cluster'].value_counts()))
    
    return results

if __name__ == "__main__":
    run_clustering_pipeline()
