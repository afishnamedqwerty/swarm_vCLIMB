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

## End-to-End Runner
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



