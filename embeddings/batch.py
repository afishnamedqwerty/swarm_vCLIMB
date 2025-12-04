import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, List, Optional

from .vast_embedder import VASTEmbedder


def _embed_batch(
    paths: List[str],
    metadata: Optional[List[str]],
    embedder_factory: Callable[[], VASTEmbedder],
):
    embedder = embedder_factory()
    results = []
    for idx, path in enumerate(paths):
        metadata_text = metadata[idx] if metadata else None
        results.append(embedder.embed_video(path, metadata_text=metadata_text).fused)
    return results


def embed_video_batch(
    video_paths: Iterable[str],
    embedder_factory: Callable[[], VASTEmbedder] = VASTEmbedder,
    metadata_text: Optional[Iterable[str]] = None,
    batch_size: int = 16,
    max_workers: int = 4,
) -> np.ndarray:
    """
    Embed many videos in parallel batches.

    `embedder_factory` should construct a VASTEmbedder per worker to avoid
    cross-process GPU contention.

    Returns a float32 numpy array shaped (n_videos, embedding_dim).
    """
    video_paths = list(video_paths)
    metadata_list = list(metadata_text) if metadata_text is not None else None
    embeddings: List[np.ndarray] = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i : i + batch_size]
            batch_meta = metadata_list[i : i + batch_size] if metadata_list else None
            futures.append(pool.submit(_embed_batch, batch_paths, batch_meta, embedder_factory))

        for fut in futures:
            embeddings.extend(fut.result())

    return np.asarray(embeddings, dtype="float32")
