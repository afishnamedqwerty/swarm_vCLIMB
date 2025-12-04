#!/usr/bin/env bash
set -euo pipefail

echo "[0/5] Building datamap-rs locally..."
pushd datamap-rs > /dev/null
cargo build --release
popd > /dev/null
DATAMAP_BIN="${DATAMAP_BIN:-$(pwd)/datamap-rs/target/release/datamap}"

# User-configurable inputs
: "${VIDEO_LIST:=data/videos.txt}"               # one path per line
: "${LABELS_JSON:=data/calibration_labels.json}" # video_path -> int label
: "${OUT_DIR:=outputs}"
: "${N_CLUSTERS:=1000}"

mkdir -p "${OUT_DIR}"

EMBEDDINGS="${OUT_DIR}/embeddings.npy"
ASSIGNMENTS="${OUT_DIR}/assignments.npy"
CENTROIDS="${OUT_DIR}/centroids.npy"
QUALITY_JSON="${OUT_DIR}/quality_scores.json"
METADATA_JSONL="${OUT_DIR}/metadata/metadata.jsonl"
MIXTURE_JSON="${OUT_DIR}/mixture.json"
RESHARDED_DIR="${OUT_DIR}/resharded"
FILTERED_DIR="${OUT_DIR}/filtered"
PARTITIONED_DIR="${OUT_DIR}/partitioned"

echo "[1/5] Embedding videos with ImageBind + VAST fusion..."
python - <<'PY'
import os
import numpy as np
from embeddings.batch import embed_video_batch
from embeddings.vast_embedder import VASTEmbedder

video_list = os.environ["VIDEO_LIST"]
out_embeddings = os.environ["EMBEDDINGS"]

videos = [v.strip() for v in open(video_list, "r") if v.strip()]
embeddings = embed_video_batch(
    videos,
    embedder_factory=VASTEmbedder,
    batch_size=8,
    max_workers=4,
)
np.save(out_embeddings, embeddings)
print(f"Saved embeddings to {out_embeddings}, shape={embeddings.shape}")
PY

echo "[2/5] Clustering with FAISS K-means..."
python - <<'PY'
import os
import numpy as np
from clustering.faiss_kmeans import cluster_embeddings

embeddings = np.load(os.environ["EMBEDDINGS"])
n_clusters = int(os.environ.get("N_CLUSTERS", "1000"))
result = cluster_embeddings(embeddings, n_clusters=n_clusters, n_iter=20, use_gpu=True)
np.save(os.environ["ASSIGNMENTS"], result.assignments)
np.save(os.environ["CENTROIDS"], result.centroids)
print(f"Saved assignments to {os.environ['ASSIGNMENTS']}")
PY

echo "[3/5] Quality scoring videos (ffprobe heuristics)..."
python - <<'PY'
import os, json
from clustering.quality_impl import FFprobeQualityScorer

video_list = os.environ["VIDEO_LIST"]
videos = [v.strip() for v in open(video_list, "r") if v.strip()]
scorer = FFprobeQualityScorer()
scores = {v: scorer.score(v) for v in videos}
out_path = os.environ["QUALITY_JSON"]
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(scores, f, indent=2)
print(f"Wrote quality scores to {out_path}")
PY

echo "[4/5] Exporting JSONL metadata and running datamap-rs..."
mkdir -p "$(dirname "${METADATA_JSONL}")"
python scripts/export_jsonl.py \
  --video-list "${VIDEO_LIST}" \
  --embeddings "${EMBEDDINGS}" \
  --assignments "${ASSIGNMENTS}" \
  --quality-json "${QUALITY_JSON}" \
  --output "${METADATA_JSONL}"

# datamap-rs steps using locally built binary (override with DATAMAP_BIN)
"${DATAMAP_BIN}" reshard --input "$(dirname "${METADATA_JSONL}")" --output "${RESHARDED_DIR}" --target-size 256MB
"${DATAMAP_BIN}" map --input "${RESHARDED_DIR}" --output "${FILTERED_DIR}" --config datamap-rs/configs/filter_config.yaml
"${DATAMAP_BIN}" discrete-partition --input "${FILTERED_DIR}" --output "${PARTITIONED_DIR}" --key cluster_id

echo "[5/5] Running CLIMB bootstrapping loop..."
python scripts/run_bootstrap.py \
  --video-list "${VIDEO_LIST}" \
  --embeddings "${EMBEDDINGS}" \
  --assignments "${ASSIGNMENTS}" \
  --labels "${LABELS_JSON}" \
  --output "${MIXTURE_JSON}"

echo "Complete. Mixture weights written to ${MIXTURE_JSON}"
