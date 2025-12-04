from typing import Dict, Iterable, List, Tuple

import numpy as np


def prune_clusters(
    assignments: np.ndarray,
    quality_scores: np.ndarray,
    threshold: float = 3.0,
) -> List[int]:
    """
    Prune clusters with mean quality below threshold.

    Args:
        assignments: cluster ids per sample (shape: n_samples,)
        quality_scores: scalar quality per sample (shape: n_samples,)
        threshold: minimum cluster mean to keep
    """
    if assignments.shape[0] != quality_scores.shape[0]:
        raise ValueError("assignments and quality_scores must share the same length.")

    cluster_ids = np.unique(assignments)
    kept: List[int] = []
    for cid in cluster_ids:
        mask = assignments == cid
        if mask.any() and quality_scores[mask].mean() >= threshold:
            kept.append(int(cid))
    return kept


def merge_clusters(
    centroids: np.ndarray,
    assignments: np.ndarray,
    distance_threshold: float = 1.5,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Merge clusters whose centroids are within a distance threshold.

    Returns updated assignments and a mapping from old -> new ids.
    """
    if centroids.ndim != 2:
        raise ValueError("centroids must be 2D (n_clusters, dim).")

    n_clusters = centroids.shape[0]
    dist = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :],
        axis=2,
    )

    visited = set()
    groups: List[List[int]] = []
    for i in range(n_clusters):
        if i in visited:
            continue
        group = [i]
        for j in range(i + 1, n_clusters):
            if j in visited:
                continue
            if dist[i, j] < distance_threshold:
                group.append(j)
                visited.add(j)
        visited.add(i)
        groups.append(group)

    mapping: Dict[int, int] = {}
    for new_id, group in enumerate(groups):
        for old_id in group:
            mapping[old_id] = new_id

    merged_assignments = np.array([mapping[int(cid)] for cid in assignments], dtype=int)
    return merged_assignments, mapping
