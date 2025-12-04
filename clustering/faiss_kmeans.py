from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - optional dependency
    faiss = None


@dataclass
class ClusterResult:
    assignments: np.ndarray
    centroids: np.ndarray


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 1000,
    n_iter: int = 20,
    use_gpu: bool = True,
    seed: int = 42,
) -> ClusterResult:
    """
    Run FAISS K-means on embeddings following the CLIMB recipe.

    Args:
        embeddings: float32 array of shape (n_items, dim)
        n_clusters: initial K (K_init)
        n_iter: number of Lloyd iterations
        use_gpu: enable FAISS GPU path when available
    """
    if faiss is None:
        raise ImportError("faiss is required for clustering. Install faiss-gpu or faiss-cpu.")

    data = np.asarray(embeddings, dtype="float32")
    if data.ndim != 2:
        raise ValueError("Embeddings must be 2D (n_items, dim).")

    faiss.normalize_L2(data)
    kmeans = faiss.Kmeans(
        d=data.shape[1],
        k=n_clusters,
        niter=n_iter,
        verbose=True,
        gpu=use_gpu,
        seed=seed,
    )
    kmeans.train(data)

    _, idx = kmeans.index.search(data, 1)
    return ClusterResult(assignments=idx.ravel(), centroids=kmeans.centroids)
