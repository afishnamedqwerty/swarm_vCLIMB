from typing import Callable, Dict, Iterable, Sequence, Tuple

import numpy as np

from .dirichlet_sampler import MixtureSampler
from .predictor import PerformancePredictor
from .proxy import ProxyTrainer, VideoMixture


def _contiguous_clusters(videos_by_cluster: Dict[int, Sequence[str]]) -> Tuple[Dict[int, Sequence[str]], Dict[int, int]]:
    """
    Re-index arbitrary cluster ids into a contiguous 0..N-1 mapping.
    Returns the remapped dict and reverse lookup.
    """
    ordered = sorted(videos_by_cluster.keys())
    id_map = {cid: idx for idx, cid in enumerate(ordered)}
    remapped = {id_map[cid]: vids for cid, vids in videos_by_cluster.items()}
    reverse = {v: k for k, v in id_map.items()}
    return remapped, reverse


def climb_bootstrap(
    videos_by_cluster: Dict[int, Sequence[str]],
    build_model: Callable[[], object],
    train_fn: Callable[[object, Iterable[str]], None],
    eval_fn: Callable[[object], float],
    iterations: int = 3,
    samples_per_iteration: Sequence[int] = (64, 32, 16),
    top_n: int = 100,
    candidate_pool: int = 1000,
    tokens_budget: int = 1_000_000_000,
    trainer: ProxyTrainer | None = None,
) -> Tuple[VideoMixture, Dict[int, int]]:
    """
    Iterative CLIMB bootstrapping loop.

    Returns:
        best_mixture: predicted best mixture after final iteration
        reverse_cluster_map: mapping from contiguous cluster ids back to original ids
    """
    remapped, reverse_map = _contiguous_clusters(videos_by_cluster)
    n_clusters = len(remapped)
    if iterations > len(samples_per_iteration):
        raise ValueError("samples_per_iteration must have >= iterations entries.")
    cluster_sizes = np.array([len(remapped[cid]) for cid in range(n_clusters)])

    sampler = MixtureSampler(cluster_sizes)
    predictor = PerformancePredictor()
    trainer = trainer or ProxyTrainer(remapped, tokens_budget=tokens_budget)

    history_weights = []
    history_perf = []

    for it in range(iterations):
        n_samples = samples_per_iteration[it]
        if it == 0 or predictor.model is None:
            weights = sampler.sample(n_samples)
        else:
            candidates = sampler.sample(candidate_pool)
            predicted = predictor.predict(candidates)
            chosen_idx = sampler.sample_from_top(predicted, top_n=top_n, n_samples=n_samples)
            weights = candidates[chosen_idx]

        performances = []
        for w in weights:
            mixture = VideoMixture(weights=w)
            perf = trainer.run(mixture, build_model, train_fn, eval_fn)
            performances.append(perf)
            history_weights.append(w)
            history_perf.append(perf)

        predictor.update(np.stack(weights), np.asarray(performances))

    final_candidates = sampler.sample(candidate_pool * 10)
    final_predictions = predictor.predict(final_candidates)
    best_idx = int(np.argmax(final_predictions))
    best_weights = final_candidates[best_idx]
    best_mixture = VideoMixture(weights=best_weights, performance=final_predictions[best_idx])

    return best_mixture, reverse_map
