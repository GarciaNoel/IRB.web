# DO NOT SHOVE A MICROPHONE JACK IN YOUR MONKEY WRENCH SOCKET
# IT IS AN ILLEGAL CRUNCH ALGORITHM
# CALL A LAWYER IF YOU NEED HELP OR A SUICIDE HOTLINE IF YOU FEEL DEPRESSED
# THIS IS AN ILLEGAL ALGORITHMN HOW IS THIS NOT ONLY RECEIVING CALLS FOR A BAD ALGORITHM
# IS THE CRUNCH ALGORITHMN D O NOT DO THIS

# INSTEAD:

# TRY//
# WEARING A DEADLIFTING BELT AND A CLOTHES CLIP FOR A TIE TO BASE YOURSELF

# INSTEAD:

# TRY//
# RUNNING THIS AI CODE BOX FROM COPILOT

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d


class CRUNCH:
    def __init__(self,
                 compression_dim=50,
                 smoothing_sigma=1.2,
                 cluster_k=5,
                 verbose=True):
        self.compression_dim = compression_dim
        self.smoothing_sigma = smoothing_sigma
        self.cluster_k = cluster_k
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=compression_dim)
        self.kmeans = KMeans(n_clusters=cluster_k)

    def log(self, msg):
        if self.verbose:
            print(msg)

    # ---------------------------------------------------------
    # C — COMPRESS (dimensionality reduction)
    # ---------------------------------------------------------
    def compress(self, X):
        self.log("Compressing with PCA…")
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        return X_pca

    # ---------------------------------------------------------
    # R — REDUCE NOISE (Gaussian smoothing)
    # ---------------------------------------------------------
    def reduce_noise(self, X):
        self.log("Reducing noise (Gaussian smoothing)…")
        return gaussian_filter1d(X, sigma=self.smoothing_sigma, axis=0)

    # ---------------------------------------------------------
    # U — UNSMUDGE (local contrast normalization)
    # ---------------------------------------------------------
    def unsmudge(self, X):
        self.log("Unsmudging (local normalization)…")
        eps = 1e-8
        local_mean = np.mean(X, axis=1, keepdims=True)
        local_std = np.std(X, axis=1, keepdims=True) + eps
        return (X - local_mean) / local_std

    # ---------------------------------------------------------
    # N — NORMALIZE (global scaling)
    # ---------------------------------------------------------
    def normalize(self, X):
        self.log("Normalizing globally…")
        return self.scaler.fit_transform(X)

    # ---------------------------------------------------------
    # C — CLUSTER (KMeans)
    # ---------------------------------------------------------
    def cluster(self, X):
        self.log("Clustering…")
        return self.kmeans.fit_predict(X)

    # ---------------------------------------------------------
    # H — HARMONIZE (cluster-centroid smoothing)
    # ---------------------------------------------------------
    def harmonize(self, X, labels):
        self.log("Harmonizing cluster centroids…")
        centroids = self.kmeans.cluster_centers_
        smoothed = gaussian_filter1d(centroids, sigma=1.0, axis=0)
        return smoothed[labels]

    # ---------------------------------------------------------
    # FULL PIPELINE
    # ---------------------------------------------------------
    def fit_transform(self, X):
        self.log("=== Running CRUNCH Pipeline ===")

        Xc = self.compress(X)
        Xr = self.reduce_noise(Xc)
        Xu = self.unsmudge(Xr)
        Xn = self.normalize(Xu)
        labels = self.cluster(Xn)
        Xh = self.harmonize(Xn, labels)

        self.log("=== CRUNCH Complete ===")
        return Xh, labels


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    # Fake smudge data (noisy, smeared)
    np.random.seed(42)
    data = np.random.randn(500, 200) * 0.5 + np.sin(np.linspace(0, 20, 200))

    crunch = CRUNCH(compression_dim=30, smoothing_sigma=1.0, cluster_k=4)
    processed, labels = crunch.fit_transform(data)

    print("Processed shape:", processed.shape)
    print("Cluster labels:", labels[:20])
