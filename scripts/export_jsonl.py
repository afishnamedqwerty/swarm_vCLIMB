#!/usr/bin/env python
import argparse
import json
import pathlib
from typing import Dict, List, Optional

import numpy as np


def read_lines(path: str) -> List[str]:
    return [line.strip() for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def load_quality(path: Optional[str]) -> Optional[Dict[str, dict]]:
    if not path:
        return None
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Export embeddings + cluster assignments to JSONL.")
    parser.add_argument("--video-list", required=True, help="Text file with one video path per line.")
    parser.add_argument("--embeddings", required=True, help="Path to .npy embeddings aligned with video-list.")
    parser.add_argument("--assignments", required=True, help="Path to .npy cluster assignments aligned with video-list.")
    parser.add_argument("--quality-json", help="Optional JSON mapping video_path -> quality_scores dict.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    args = parser.parse_args()

    videos = read_lines(args.video_list)
    embeddings = np.load(args.embeddings)
    assignments = np.load(args.assignments)

    if embeddings.shape[0] != len(videos):
        raise ValueError("embeddings rows must match number of videos")
    if assignments.shape[0] != len(videos):
        raise ValueError("assignments length must match number of videos")

    quality_map = load_quality(args.quality_json)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for idx, (video, emb, cid) in enumerate(zip(videos, embeddings, assignments)):
            entry = {
                "id": idx,
                "path": video,
                "embedding": emb.tolist(),
                "cluster_id": int(cid),
            }
            if quality_map and video in quality_map:
                entry["quality_scores"] = quality_map[video]
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
