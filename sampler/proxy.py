from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class VideoMixture:
    weights: np.ndarray
    performance: float | None = None


class ProxyTrainer:
    """
    Minimal proxy training harness.

    Users supply model and evaluation callables; this class handles sampling
    according to mixture weights and abstracts the training lifecycle.
    """

    def __init__(
        self,
        videos_by_cluster: Dict[int, Sequence[str]],
        tokens_budget: int = 1_000_000_000,
        seed: int = 42,
    ):
        self.videos_by_cluster = videos_by_cluster
        self.tokens_budget = tokens_budget
        self.rng = np.random.default_rng(seed)

    def sample_videos(self, weights: np.ndarray) -> List[str]:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

        chosen: List[str] = []
        for cid, w in enumerate(weights):
            cluster_videos = list(self.videos_by_cluster.get(cid, []))
            if not cluster_videos:
                continue
            take = max(1, int(self.tokens_budget * w / max(len(cluster_videos), 1)))
            picks = self.rng.choice(cluster_videos, size=take, replace=True)
            chosen.extend(picks.tolist())
        return chosen

    def build_dataloader(self, videos: Iterable[str]):
        """
        Placeholder hook to construct a dataloader.
        Replace with task-specific dataset logic.
        """
        return list(videos)

    def run(
        self,
        mixture: VideoMixture,
        build_model: Callable[[], object],
        train_fn: Callable[[object, Iterable[str]], None],
        eval_fn: Callable[[object], float],
    ) -> float:
        """
        Train and evaluate a proxy model for the given mixture.

        Args:
            mixture: mixture weights to sample videos
            build_model: returns a fresh proxy model
            train_fn: training closure consuming (model, dataloader)
            eval_fn: returns scalar performance on calibration set
        """
        videos = self.sample_videos(mixture.weights)
        dataloader = self.build_dataloader(videos)
        model = build_model()
        train_fn(model, dataloader)
        performance = eval_fn(model)
        mixture.performance = performance
        return performance
