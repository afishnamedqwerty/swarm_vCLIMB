# swarm_vCLIMB Multimodal Toolkit

Nemotron-videoCLIMB iterative mixture refinement loop for video, audio, and text utilizing datamap-rs toolkit (credit to AllenAI team). It includes ImageBind-based VAST embedding generation, FAISS clustering with pruning/merging, quality scoring hooks, Dirichlet sampling, proxy training scaffolding, a LightGBM predictor, and the iterative bootstrapping loop.

## Repository Layout
- `embeddings/`: ImageBind preprocessing, VAST embedder, and batch embedding helpers
- `clustering/`: FAISS K-means, pruning/merging, and quality scoring interfaces
- `sampler/`: Dirichlet sampler, proxy training harness, LightGBM predictor, and CLIMB loop
- `scripts/`: End-to-end runners for JSONL export and bootstrapping
- `datamap-rs/`: External toolkit for large-scale metadata filtering/resharding (https://github.com/allenai/datamap-rs)

## Dependencies
- Python packages: `torch`, `imagebind`, `faiss-gpu` (or `faiss-cpu`), `lightgbm`
- System tools: `ffmpeg` for audio/frame extraction
- Optional: Whisper ASR + text encoder for speech/metadata text, `decord`/`opencv-python` if you want to customize preprocessing
- Install Rust (if not already installed):
```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
- Add python packages:
```bash
   python3 -m venv ./.venv
   source .venv/bin/activate
   pip install -r requirements.txt
```

## Embedding Pipeline
```python
from embeddings.vast_embedder import VASTEmbedder

embedder = VASTEmbedder()  # optionally pass whisper + text_encoder callables
vast = embedder.embed_video("/data/video.mp4", metadata_text="title or description")
print(vast.fused.shape)  # unified VAST vector
```

Batch embedding in parallel:
```python
from embeddings.batch import embed_video_batch
embeddings = embed_video_batch(video_paths, embedder_factory=VASTEmbedder, batch_size=16)
```

## Clustering and Quality
```python
from clustering.faiss_kmeans import cluster_embeddings
from clustering.merge_prune import prune_clusters, merge_clusters

result = cluster_embeddings(embeddings, n_clusters=1000, n_iter=20, use_gpu=True)
kept = prune_clusters(result.assignments, quality_scores, threshold=3.0)
merged_assignments, mapping = merge_clusters(result.centroids, result.assignments)
```

Quality scoring hooks live in `clustering/quality.py`; plug in your own multimodal scorer and use `score_clusters` to aggregate cluster-level metrics.

## Iterative Bootstrapping Loop
```python
from sampler import climb_bootstrap, EmbeddingProxyTrainer, ProxyConfig, make_proxy_functions
import numpy as np

# videos_by_cluster: dict[int, list[str]] from clustering
# feature_store: dict[path -> np.ndarray] aligned with your video list
# labels: dict[path -> int] for calibration set
cfg = ProxyConfig(input_dim=embedding_dim, num_classes=num_labels)
trainer = EmbeddingProxyTrainer(videos_by_cluster, feature_store, labels, cfg)
build_model, train_fn, eval_fn = make_proxy_functions(trainer)

best_mixture, cluster_map = climb_bootstrap(
    videos_by_cluster,
    build_model=build_model,
    train_fn=train_fn,
    eval_fn=eval_fn,
    iterations=3,
    samples_per_iteration=(64, 32, 16),
    trainer=trainer,
)
print(best_mixture.weights, best_mixture.performance)
```

The loop follows the CLIMB schedule: Dirichlet sampling on iteration 1, predictor-guided Top-N sampling thereafter, proxy training per sample, and LightGBM fitting at each iteration.

## Quality Scoring
`clustering/quality_impl.py` provides an ffprobe-based heuristic scorer producing visual/audio quality (0-5 scale) and neutral priors for other dimensions. Plug in your own multimodal classifiers by implementing `QualityScorer.score`.

## datamap-rs Integration
Use the embeddings and cluster assignments to emit JSONL metadata, then process with `datamap-rs`:
- `reshard` to 256MB chunks
- `map` to filter low-quality items (quality score thresholds)
- `discrete-partition` by `cluster_id`
- `group` for deduplication by `content_hash`

The `datamap-rs` directory contains the upstream toolkit; consult its README for CLI usage.

Pipeline notes:
- No source changes to `datamap-rs` are needed; `scripts/run_pipeline.sh` builds the local binary (`DATAMAP_BIN` override supported) and uses the CLI directly.
- JSONL schema must include `cluster_id` (and `quality_scores.*` if you want to filter on them); add `content_hash` if you plan to use the `group` dedup stage.
- Filtering config lives in `datamap-rs/configs/filter_config.yaml`; adjust thresholds or add annotators as needed and pass via `--config`.
- The canonical op sequence is `reshard -> map -> discrete-partition -> group` (optional) over the JSONL emitted by `scripts/export_jsonl.py`.

## Local End-to-End Runner
`scripts/run_pipeline.sh` wires everything together:
- VAST embedding generation
- FAISS clustering
- ffprobe-based quality scoring
- JSONL export + datamap filtering/resharding
- CLIMB bootstrapping loop via cached embeddings

Configure `VIDEO_LIST`, `LABELS_JSON`, `OUT_DIR`, `N_CLUSTERS`, then run:
```bash
bash scripts/run_pipeline.sh
```

## Distributed End-to-End Runner
`scripts/run_pipeline_k8s.sh` extends the pipeline for Kubernetes-based distributed training:
- VAST embedding generation (local)
- FAISS clustering (local)
- ffprobe-based quality scoring (local)
- JSONL export + datamap filtering/resharding (local)
- **Upload data to Kubernetes PVC**
- **Distributed CLIMB bootstrapping via worker pool**

This script performs steps 1-4 locally to prepare data, then uploads embeddings, assignments, labels, and video list to Kubernetes shared storage. Training jobs are distributed across GPU worker pods coordinated by the orchestrator via Redis queue.

Configure environment variables, ensure Kubernetes cluster is accessible, then run:
```bash
export VIDEO_LIST=data/videos.txt
export LABELS_JSON=data/calibration_labels.json
export N_CLUSTERS=1000
export ITERATIONS=3
export SAMPLES_PER_ITERATION=64,32,16

bash scripts/run_pipeline_k8s.sh
```

The script will:
1. Prepare data locally (same as local runner)
2. Upload to Kubernetes PVC automatically
3. Submit training job to orchestrator API
4. Poll for completion and retrieve results

**Prerequisites**: Kubernetes cluster with GPU nodes, deployed manifests (`kubectl apply -f k8s/ istio/`)

## Distributed Kubernetes Deployment

For production deployments with 30M+ parameter proxy models, the system can be deployed to Kubernetes with distributed training across multiple GPU worker pods.

### Architecture

- **Orchestrator Pod**: Manages CLIMB loop, Dirichlet sampling, LightGBM predictor
- **Worker Pods**: Train individual proxy models in parallel (horizontally scalable)
- **Redis Pod**: Job queue for distributing training tasks
- **Shared Storage**: PersistentVolumes for embeddings, assignments, labels
- **Istio Service Mesh**: mTLS, traffic management, load balancing

### Quick Start

1. **Build and push Docker images**:
```bash
docker build -f Dockerfile.orchestrator -t your-registry/swarm-orchestrator:latest .
docker build -f Dockerfile.worker -t your-registry/swarm-worker:latest .
docker push your-registry/swarm-orchestrator:latest
docker push your-registry/swarm-worker:latest
```

2. **Update image references** in `k8s/orchestrator.yaml` and `k8s/worker.yaml`

3. **Prepare data**: Upload embeddings, assignments, labels, and video list to shared PVC:
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml

# Create temporary data loader pod
kubectl run -n swarm-vclimb data-loader --image=busybox --restart=Never -- sleep 3600
kubectl set volume pod/data-loader -n swarm-vclimb --add \
  --name=data --type=pvc --claim-name=embeddings-pvc --mount-path=/data

# Copy data files
kubectl cp outputs/embeddings.npy swarm-vclimb/data-loader:/data/
kubectl cp outputs/assignments.npy swarm-vclimb/data-loader:/data/
kubectl cp data/calibration_labels.json swarm-vclimb/data-loader:/data/labels.json
kubectl cp data/videos.txt swarm-vclimb/data-loader:/data/video_list.txt
```

4. **Deploy to Kubernetes**:
```bash
# Deploy core components
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/orchestrator.yaml
kubectl apply -f k8s/worker.yaml

# Deploy Istio configuration
kubectl apply -f istio/peer-authentication.yaml
kubectl apply -f istio/virtual-service.yaml
kubectl apply -f istio/destination-rule.yaml
kubectl apply -f istio/authorization-policy.yaml

# Verify deployment
kubectl get pods -n swarm-vclimb
```

5. **Start distributed training**:
```bash
# Port forward to orchestrator
kubectl port-forward -n swarm-vclimb svc/orchestrator 8000:8000

# Submit training job
curl -X POST http://localhost:8000/bootstrap/start \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings_path": "embeddings.npy",
    "assignments_path": "assignments.npy",
    "labels_path": "labels.json",
    "video_list_path": "video_list.txt",
    "iterations": 3,
    "samples_per_iteration": [64, 32, 16],
    "config": {
      "hidden_dim": 256,
      "epochs": 2,
      "batch_size": 64
    }
  }'

# Monitor status
curl http://localhost:8000/bootstrap/status

# Retrieve results
curl http://localhost:8000/bootstrap/result > mixture.json
```

### Scaling Workers

Scale horizontally based on available GPU resources:

```bash
# Scale up to 10 workers
kubectl scale deployment proxy-worker -n swarm-vclimb --replicas=10

# Scale down to 3 workers
kubectl scale deployment proxy-worker -n swarm-vclimb --replicas=3
```

### Configuration

Edit `k8s/configmap.yaml` to adjust training parameters:

```yaml
data:
  HIDDEN_DIM: "256"
  EPOCHS: "2"
  BATCH_SIZE: "64"
  LEARNING_RATE: "0.001"
```

Edit `k8s/worker.yaml` to adjust resource limits for 30M+ parameter models:

```yaml
resources:
  requests:
    memory: "16Gi"
    cpu: "8"
    nvidia.com/gpu: "1"
```

### Resource Requirements

- Kubernetes cluster (v1.24+) with GPU nodes
- Istio service mesh (v1.18+) installed
- Storage class supporting ReadWriteMany (NFS, CephFS, etc.)
- NVIDIA device plugin for GPU scheduling
- Worker pods: 16-32Gi memory, 1 GPU (V100 or better)
- Orchestrator: 8-16Gi memory, no GPU needed
- Storage: 100Gi+ for embeddings/data (ReadWriteMany PVC)
- Recommended: 5-10 worker replicas for good throughput

## Roadmap 
### Swarm-Based Mixing
1. Swarm construction
- Train 5 × K_enhanced proxy models (30M parameters each)
- Each trained for 5× Chinchilla tokens on a Dirichlet-sampled mixture
- Dirichlet centered on natural distribution
2. Per-Task Regression
- Evaluate each proxy on T evaluation tasks
- Fit T separate GLMs: f_t(α) predicting BPB for task t
- Optionally fit auxiliary LightGBM ensemble for nonlinearity detection
3. Constrained Optimization
- Objective: minimize Σ_t w_t · f_t(α)
- Constraints: simplex, non-negativity, per-domain upsampling limits (r_max = 7)
- Solver: SLSQP or interior-point method
- Initialize from natural distribution; run from multiple initializations
4. Quality-aware Upsampling
- For each cluster i, compute required integral I_i = (α*_i · N) / n_i
- Fit piecewise linear upsampling curve respecting I_i and r_max
- Apply curve to sample training instances with quality-proportional repetition

### Conditional Mixing for Incremental Updates
1. Freeze optimized mixture as virtual domain V
2. When new domains D_new arrive:
- Define reduced optimization over {V} ∪ D_new
- Train smaller swarm (5 × |{V} ∪ D_new|)
- Re-optimize with V's internal weights frozen

### Staged Hierarchical Optimization
- Stage 1: Optimize visual domain clusters → freeze as V_visual
- Stage 2: Optimize {V_visual, audio clusters} → freeze as V_av
- Stage 3: Optimize {V_av, speech/instructional} → freeze as V_full
- Stage 4: Optimize {V_full, synthetic data} → final production mixture



