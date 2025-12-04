import numpy as np


class MixtureSampler:
    """Dirichlet sampler for cluster mixture weights."""

    def __init__(self, cluster_sizes: np.ndarray, seed: int = 42):
        cluster_sizes = np.asarray(cluster_sizes, dtype=float)
        if cluster_sizes.ndim != 1:
            raise ValueError("cluster_sizes must be 1D.")
        if (cluster_sizes <= 0).any():
            raise ValueError("cluster_sizes must be positive.")

        self.n_clusters = cluster_sizes.shape[0]
        self.alpha = cluster_sizes / cluster_sizes.sum() * self.n_clusters
        self.rng = np.random.default_rng(seed)

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample mixture weights from the base Dirichlet distribution."""
        return self.rng.dirichlet(self.alpha, size=n_samples)

    def sample_from_top(self, predicted_perf: np.ndarray, top_n: int, n_samples: int) -> np.ndarray:
        """
        Sample indices from the top-N predicted configurations to balance explore/exploit.
        """
        if predicted_perf.shape[0] < top_n:
            top_n = predicted_perf.shape[0]
        top_indices = np.argsort(predicted_perf)[-top_n:]
        chosen = self.rng.choice(top_indices, size=n_samples, replace=False)
        return chosen
