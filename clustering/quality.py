import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Protocol

import numpy as np


class QualityScorer(Protocol):
    """Protocol for scoring a single video."""

    def score(self, video_path: str) -> Mapping[str, float]:
        ...


@dataclass
class ClusterQuality:
    cluster_id: int
    scores: Dict[str, float]
    count: int


def score_clusters(
    assignments: np.ndarray,
    video_paths: Iterable[str],
    scorer: QualityScorer,
    sample_size: int = 1000,
) -> List[ClusterQuality]:
    """
    Compute mean quality metrics per cluster using a sampling strategy.
    """
    video_paths = list(video_paths)
    if len(video_paths) != len(assignments):
        raise ValueError("video_paths and assignments must share the same length.")

    by_cluster: Dict[int, List[int]] = defaultdict(list)
    for idx, cid in enumerate(assignments):
        by_cluster[int(cid)].append(idx)

    results: List[ClusterQuality] = []
    for cid, indices in by_cluster.items():
        sampled = indices
        if len(indices) > sample_size:
            sampled = random.sample(indices, sample_size)

        metrics: Dict[str, List[float]] = defaultdict(list)
        for idx in sampled:
            scores = scorer.score(video_paths[idx])
            for key, value in scores.items():
                metrics[key].append(float(value))

        aggregated = {k: float(np.mean(v)) for k, v in metrics.items()}
        results.append(ClusterQuality(cluster_id=cid, scores=aggregated, count=len(sampled)))

    return results
