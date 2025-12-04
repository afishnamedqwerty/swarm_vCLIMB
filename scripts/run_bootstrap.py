#!/usr/bin/env python
import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np

from sampler import (
    EmbeddingProxyTrainer,
    ProxyConfig,
    climb_bootstrap,
    make_proxy_functions,
)


def read_lines(path: str) -> List[str]:
    return [line.strip() for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def load_labels(path: str) -> Dict[str, int]:
    with open(path, "r") as f:
        raw = json.load(f)
    return {k: int(v) for k, v in raw.items()}


def build_videos_by_cluster(videos: List[str], assignments: np.ndarray) -> Dict[int, List[str]]:
    if len(videos) != len(assignments):
        raise ValueError("videos and assignments must have same length")
    buckets: Dict[int, List[str]] = {}
    for path, cid in zip(videos, assignments):
        buckets.setdefault(int(cid), []).append(path)
    return buckets


def main():
    parser = argparse.ArgumentParser(description="Run CLIMB bootstrapping with cached embeddings.")
    parser.add_argument("--video-list", required=True, help="Text file with one video path per line.")
    parser.add_argument("--embeddings", required=True, help="Path to .npy embeddings aligned with video-list.")
    parser.add_argument("--assignments", required=True, help="Path to .npy cluster assignments aligned with video-list.")
    parser.add_argument("--labels", required=True, help="JSON mapping video_path -> int label for calibration.")
    parser.add_argument("--output", required=True, help="Where to write the best mixture JSON.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--samples", type=str, default="64,32,16", help="Comma-separated samples per iteration.")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--candidate-pool", type=int, default=1000)
    parser.add_argument("--tokens-budget", type=int, default=1_000_000_000)
    parser.add_argument("--device", default="cuda", help="Device for proxy model (cuda/cpu).")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    videos = read_lines(args.video_list)
    embeddings = np.load(args.embeddings)
    assignments = np.load(args.assignments)
    labels = load_labels(args.labels)

    feature_store = {path: emb for path, emb in zip(videos, embeddings)}
    videos_by_cluster = build_videos_by_cluster(videos, assignments)

    cfg = ProxyConfig(
        input_dim=embeddings.shape[1],
        num_classes=len(set(labels.values())),
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    trainer = EmbeddingProxyTrainer(
        videos_by_cluster=videos_by_cluster,
        features=feature_store,
        labels=labels,
        cfg=cfg,
        tokens_budget=args.tokens_budget,
    )
    build_model, train_fn, eval_fn = make_proxy_functions(trainer)

    samples_per_iteration = tuple(int(x) for x in args.samples.split(",") if x)
    best_mixture, reverse_map = climb_bootstrap(
        videos_by_cluster,
        build_model=build_model,
        train_fn=train_fn,
        eval_fn=eval_fn,
        iterations=args.iterations,
        samples_per_iteration=samples_per_iteration,
        top_n=args.top_n,
        candidate_pool=args.candidate_pool,
        tokens_budget=args.tokens_budget,
        trainer=trainer,
    )

    out = {
        "weights": best_mixture.weights.tolist(),
        "predicted_performance": best_mixture.performance,
        "cluster_id_map": reverse_map,
    }
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote best mixture to {out_path}")


if __name__ == "__main__":
    main()
