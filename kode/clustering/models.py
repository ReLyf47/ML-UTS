import numpy as np
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

class ClusterModelBenchmark:
    """Benchmark suite comparing K-Means, Gaussian Mixture Models, and HDBSCAN."""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def evaluate_kmeans_sweep(self, X: np.ndarray, k_range: List[int], sample_size_sil: int = 10000) -> List[Dict]:
        """Evaluate K-Means across k range measuring Inertia, Silhouette, DB, and CH indices."""
        results = []
        np.random.seed(self.random_state)
        sil_indices = np.random.choice(len(X), size=min(sample_size_sil, len(X)), replace=False)
        X_sil = X[sil_indices]
        
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=self.random_state)
            labels = km.fit_predict(X)
            
            sil = float(silhouette_score(X_sil, labels[sil_indices]))
            db = float(davies_bouldin_score(X, labels))
            ch = float(calinski_harabasz_score(X, labels))
            inertia = float(km.inertia_)
            
            results.append({
                "k": k,
                "inertia": inertia,
                "silhouette": sil,
                "davies_bouldin": db,
                "calinski_harabasz": ch
            })
        return results

    def evaluate_gmm_sweep(self, X: np.ndarray, k_range: List[int], sample_size: int = 25000) -> List[Dict]:
        """Evaluate GMM across k range measuring BIC and AIC on representative subsample."""
        results = []
        np.random.seed(self.random_state)
        sample_idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
        X_sub = X[sample_idx]
        
        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type='diag', max_iter=100, random_state=self.random_state)
            gmm.fit(X_sub)
            
            bic = float(gmm.bic(X_sub))
            aic = float(gmm.aic(X_sub))
            
            results.append({
                "k": k,
                "bic": bic,
                "aic": aic
            })
        return results

    def run_hdbscan_anomalies(self, X: np.ndarray, min_cluster_size: int = 150, min_samples: int = 50) -> np.ndarray:
        """Run HDBSCAN density clustering to identify noise points / anomalies (-1)."""
        hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=False)
        labels = hdb.fit_predict(X)
        return labels

    def evaluate_agglomerative_sweep(
        self,
        X: np.ndarray,
        k_range: List[int],
        sample_size: int = 5000
    ) -> List[Dict]:
        """Compare Ward hierarchical (agglomerative) clustering on a representative subsample.

        Uses the same silhouette metric as K-Means so methods are comparable on equal footing.
        """
        results = []
        np.random.seed(self.random_state)
        idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
        X_sub = X[idx]

        for k in k_range:
            agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = agg.fit_predict(X_sub)
            sil = float(silhouette_score(X_sub, labels))
            results.append({"k": k, "silhouette": sil})

        return results

    def run_hdbscan_density_benchmark(
        self,
        X: np.ndarray,
        sample_size: int = 5000,
        min_cluster_size: int = 100,
        min_samples: int = 30
    ) -> Dict:
        """Run HDBSCAN density-based clustering on a deterministic subsample.

        HDBSCAN does not require choosing k: it discovers dense regions and marks
        the remainder as noise (-1), which is exactly what an anomaly detector needs.
        """
        np.random.seed(self.random_state)
        idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
        X_sub = X[idx]

        hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=False)
        labels = hdb.fit_predict(X_sub)

        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        noise_points = int((labels == -1).sum())
        return {
            "sample_size": int(len(X_sub)),
            "n_clusters_found": n_clusters,
            "noise_points": noise_points,
            "noise_ratio": float(noise_points / len(labels)),
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
        }

    def evaluate_stability(self, X: np.ndarray, optimal_k: int, n_bootstraps: int = 5, subsample_ratio: float = 0.8) -> Dict:
        """Subsampling bootstrap stability analysis computing pairwise Adjusted Rand Index (ARI)."""
        n_samples = len(X)
        sub_size = int(n_samples * subsample_ratio)
        
        runs_indices = []
        runs_labels = []
        
        for i in range(n_bootstraps):
            np.random.seed(self.random_state + i)
            idx = np.random.choice(n_samples, size=sub_size, replace=False)
            km = KMeans(n_clusters=optimal_k, n_init=5, max_iter=200, random_state=self.random_state + i)
            lbl = km.fit_predict(X[idx])
            
            runs_indices.append(idx)
            runs_labels.append(lbl)
            
        # Pairwise ARI on overlapping instances
        pairwise_ari = []
        for i in range(n_bootstraps):
            for j in range(i + 1, n_bootstraps):
                common_ids, positions_i, positions_j = np.intersect1d(
                    runs_indices[i], runs_indices[j], assume_unique=True, return_indices=True
                )
                if len(common_ids) > 100:
                    lbls_i = runs_labels[i][positions_i]
                    lbls_j = runs_labels[j][positions_j]
                    ari = float(adjusted_rand_score(lbls_i, lbls_j))
                    pairwise_ari.append(ari)
                    
        return {
            "optimal_k": optimal_k,
            "mean_ari": float(np.mean(pairwise_ari)),
            "std_ari": float(np.std(pairwise_ari)),
            "pairwise_aris": pairwise_ari
        }
