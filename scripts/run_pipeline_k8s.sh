#!/usr/bin/env bash
# Pipeline script for Kubernetes distributed training
# This version prepares data locally, then submits training to Kubernetes cluster
set -euo pipefail

echo "=== swarm_vCLIMB Kubernetes Pipeline ==="
echo "This script prepares data locally, then orchestrates distributed training on Kubernetes"
echo ""

# User-configurable inputs
: "${VIDEO_LIST:=data/videos.txt}"               # one path per line
: "${LABELS_JSON:=data/calibration_labels.json}" # video_path -> int label
: "${OUT_DIR:=outputs}"
: "${N_CLUSTERS:=1000}"
: "${K8S_NAMESPACE:=swarm-vclimb}"
: "${ORCHESTRATOR_SVC:=orchestrator}"
: "${LOCAL_PORT:=8000}"

# Training configuration
: "${ITERATIONS:=3}"
: "${SAMPLES_PER_ITERATION:=64,32,16}"
: "${TOP_N:=100}"
: "${CANDIDATE_POOL:=1000}"
: "${HIDDEN_DIM:=256}"
: "${EPOCHS:=2}"
: "${BATCH_SIZE:=64}"
: "${LEARNING_RATE:=0.001}"
: "${WEIGHT_DECAY:=0.0001}"

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

# Check if datamap-rs is needed
BUILD_DATAMAP="${BUILD_DATAMAP:-true}"
if [[ "${BUILD_DATAMAP}" == "true" ]]; then
    echo "[0/6] Building datamap-rs locally..."
    pushd datamap-rs > /dev/null
    cargo build --release
    popd > /dev/null
    DATAMAP_BIN="${DATAMAP_BIN:-$(pwd)/datamap-rs/target/release/datamap}"
fi

echo "[1/6] Embedding videos with ImageBind + VAST fusion..."
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

echo "[2/6] Clustering with FAISS K-means..."
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
print(f"Clustered into {n_clusters} clusters")
PY

echo "[3/6] Quality scoring videos (ffprobe heuristics)..."
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

if [[ "${BUILD_DATAMAP}" == "true" ]]; then
    echo "[4/6] Exporting JSONL metadata and running datamap-rs..."
    mkdir -p "$(dirname "${METADATA_JSONL}")"
    python scripts/export_jsonl.py \
      --video-list "${VIDEO_LIST}" \
      --embeddings "${EMBEDDINGS}" \
      --assignments "${ASSIGNMENTS}" \
      --quality-json "${QUALITY_JSON}" \
      --output "${METADATA_JSONL}"

    # datamap-rs steps using locally built binary
    "${DATAMAP_BIN}" reshard --input "$(dirname "${METADATA_JSONL}")" --output "${RESHARDED_DIR}" --target-size 256MB
    "${DATAMAP_BIN}" map --input "${RESHARDED_DIR}" --output "${FILTERED_DIR}" --config datamap-rs/configs/filter_config.yaml
    "${DATAMAP_BIN}" discrete-partition --input "${FILTERED_DIR}" --output "${PARTITIONED_DIR}" --key cluster_id
else
    echo "[4/6] Skipping datamap-rs processing (BUILD_DATAMAP=false)"
fi

echo "[5/6] Uploading data to Kubernetes PVC..."
echo "Checking if Kubernetes cluster is accessible..."
if ! kubectl cluster-info &> /dev/null; then
    echo "ERROR: Cannot access Kubernetes cluster. Please configure kubectl."
    exit 1
fi

echo "Checking if namespace ${K8S_NAMESPACE} exists..."
if ! kubectl get namespace "${K8S_NAMESPACE}" &> /dev/null; then
    echo "ERROR: Namespace ${K8S_NAMESPACE} does not exist. Please deploy Kubernetes manifests first:"
    echo "  kubectl apply -f k8s/namespace.yaml"
    echo "  kubectl apply -f k8s/pvc.yaml"
    exit 1
fi

echo "Creating temporary data loader pod..."
kubectl run -n "${K8S_NAMESPACE}" data-loader --image=busybox --restart=Never --command -- sleep 3600 2>/dev/null || true
kubectl wait --for=condition=ready pod/data-loader -n "${K8S_NAMESPACE}" --timeout=120s

echo "Mounting PVC to data loader pod..."
kubectl set volume pod/data-loader -n "${K8S_NAMESPACE}" --add \
  --name=data --type=pvc --claim-name=embeddings-pvc --mount-path=/data 2>/dev/null || true

# Wait a bit for volume to mount
sleep 5

echo "Copying data files to PVC..."
kubectl cp "${EMBEDDINGS}" "${K8S_NAMESPACE}/data-loader:/data/embeddings.npy"
kubectl cp "${ASSIGNMENTS}" "${K8S_NAMESPACE}/data-loader:/data/assignments.npy"
kubectl cp "${LABELS_JSON}" "${K8S_NAMESPACE}/data-loader:/data/labels.json"
kubectl cp "${VIDEO_LIST}" "${K8S_NAMESPACE}/data-loader:/data/video_list.txt"

echo "Verifying uploaded files..."
kubectl exec -n "${K8S_NAMESPACE}" data-loader -- ls -lh /data

echo "Cleaning up data loader pod..."
kubectl delete pod data-loader -n "${K8S_NAMESPACE}" --wait=false

echo "[6/6] Submitting distributed training job to Kubernetes..."
echo "Port forwarding to orchestrator service..."
kubectl port-forward -n "${K8S_NAMESPACE}" "svc/${ORCHESTRATOR_SVC}" "${LOCAL_PORT}:8000" &
PF_PID=$!
sleep 5  # Wait for port forward to establish

# Cleanup function
cleanup() {
    echo "Cleaning up port forward..."
    kill $PF_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Checking orchestrator health..."
if ! curl -s "http://localhost:${LOCAL_PORT}/health" &> /dev/null; then
    echo "ERROR: Cannot reach orchestrator service. Check deployment:"
    echo "  kubectl get pods -n ${K8S_NAMESPACE}"
    echo "  kubectl logs -n ${K8S_NAMESPACE} -l app=orchestrator"
    exit 1
fi

echo "Submitting bootstrap training job..."
cat > /tmp/bootstrap_request.json <<EOF
{
  "embeddings_path": "embeddings.npy",
  "assignments_path": "assignments.npy",
  "labels_path": "labels.json",
  "video_list_path": "video_list.txt",
  "iterations": ${ITERATIONS},
  "samples_per_iteration": [${SAMPLES_PER_ITERATION}],
  "top_n": ${TOP_N},
  "candidate_pool": ${CANDIDATE_POOL},
  "config": {
    "hidden_dim": ${HIDDEN_DIM},
    "epochs": ${EPOCHS},
    "batch_size": ${BATCH_SIZE},
    "lr": ${LEARNING_RATE},
    "weight_decay": ${WEIGHT_DECAY},
    "tokens_budget": 1000000000
  }
}
EOF

curl -X POST "http://localhost:${LOCAL_PORT}/bootstrap/start" \
  -H "Content-Type: application/json" \
  -d @/tmp/bootstrap_request.json

echo ""
echo "Training job submitted! Monitor progress with:"
echo "  curl http://localhost:${LOCAL_PORT}/bootstrap/status"
echo "  kubectl logs -n ${K8S_NAMESPACE} -l app=orchestrator -f"
echo "  kubectl logs -n ${K8S_NAMESPACE} -l app=proxy-worker -f"
echo ""
echo "Waiting for training to complete..."
echo "(This may take several hours depending on model size and iterations)"

# Poll for completion
while true; do
    STATUS=$(curl -s "http://localhost:${LOCAL_PORT}/bootstrap/status" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('is_training', 'unknown'))")

    if [[ "${STATUS}" == "False" ]] || [[ "${STATUS}" == "false" ]]; then
        echo "Training completed!"
        break
    fi

    echo -n "."
    sleep 30
done

echo ""
echo "Retrieving results..."
curl -s "http://localhost:${LOCAL_PORT}/bootstrap/result" > "${MIXTURE_JSON}"

echo ""
echo "=== Pipeline Complete ==="
echo "Mixture weights written to ${MIXTURE_JSON}"
echo ""
echo "To view results:"
echo "  cat ${MIXTURE_JSON}"
echo ""
echo "To clean up:"
echo "  kubectl delete pod data-loader -n ${K8S_NAMESPACE} 2>/dev/null || true"
