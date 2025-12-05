"""
Worker training service for distributed proxy model training.
Receives training jobs from Redis queue, trains models, reports results back.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import RedisClient, TrainingJob, TrainingResult, JobStatus
from sampler.proxy_impl import (
    EmbeddingDataset,
    EmbeddingProxyTrainer,
    ProxyConfig,
    make_proxy_functions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrainingService:
    """Distributed training service for proxy models"""

    def __init__(
        self,
        data_path: str = "/data",
        redis_host: str = "redis",
        redis_port: int = 6379,
        orchestrator_url: str = "http://orchestrator:8000",
    ):
        self.data_path = Path(data_path)
        self.redis_client = RedisClient(host=redis_host, port=redis_port)
        self.orchestrator_url = orchestrator_url

        # GPU configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load shared data from PVC
        self.embeddings: Dict[str, np.ndarray] = {}
        self.labels: Dict[str, int] = {}
        self.videos_by_cluster: Dict[int, List[str]] = {}

        logger.info("Training service initialized")

    def load_shared_data(self, embeddings_path: str, labels_path: str, assignments_path: str):
        """Load embeddings, labels, and cluster assignments from shared PVC"""
        import json

        logger.info("Loading shared data...")

        # Load embeddings
        emb_file = self.data_path / embeddings_path
        if emb_file.exists():
            embeddings_array = np.load(str(emb_file))
            logger.info(f"Loaded embeddings: {embeddings_array.shape}")
        else:
            raise FileNotFoundError(f"Embeddings file not found: {emb_file}")

        # Load labels
        labels_file = self.data_path / labels_path
        if labels_file.exists():
            with open(labels_file, "r") as f:
                self.labels = json.load(f)
            logger.info(f"Loaded {len(self.labels)} labels")
        else:
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        # Load assignments
        assignments_file = self.data_path / assignments_path
        if assignments_file.exists():
            assignments = np.load(str(assignments_file))
            logger.info(f"Loaded assignments: {assignments.shape}")
        else:
            raise FileNotFoundError(f"Assignments file not found: {assignments_file}")

        # Build videos_by_cluster and embeddings dict
        video_paths = list(self.labels.keys())
        for path, emb, cid in zip(video_paths, embeddings_array, assignments):
            self.embeddings[path] = emb
            cluster_id = int(cid)
            if cluster_id not in self.videos_by_cluster:
                self.videos_by_cluster[cluster_id] = []
            self.videos_by_cluster[cluster_id].append(path)

        logger.info(f"Organized {len(self.videos_by_cluster)} clusters")

    def train_proxy_model(self, job: TrainingJob) -> TrainingResult:
        """Train a single proxy model for the given job"""
        logger.info(f"Training job {job.job_id} - iteration {job.iteration}, sample {job.sample_idx}")

        try:
            # Extract config
            cfg = ProxyConfig(
                input_dim=job.config["input_dim"],
                num_classes=job.config["num_classes"],
                hidden_dim=job.config.get("hidden_dim", 256),
                epochs=job.config.get("epochs", 2),
                batch_size=job.config.get("batch_size", 64),
                lr=job.config.get("lr", 1e-3),
                weight_decay=job.config.get("weight_decay", 1e-4),
                device=str(self.device),
            )

            # Create trainer
            trainer = EmbeddingProxyTrainer(
                videos_by_cluster=self.videos_by_cluster,
                features=self.embeddings,
                labels=self.labels,
                cfg=cfg,
                tokens_budget=job.config.get("tokens_budget", 1_000_000_000),
            )

            # Get training functions
            build_model, train_fn, eval_fn = make_proxy_functions(trainer)

            # Sample videos based on mixture weights
            videos = trainer.sample_videos(job.mixture_weights)
            logger.info(f"Sampled {len(videos)} videos for training")

            # Build dataloader
            dataloader = trainer.build_dataloader(videos)

            # Train model
            model = build_model()
            train_fn(model, dataloader)

            # Evaluate
            performance = eval_fn(model)
            logger.info(f"Job {job.job_id} completed with performance: {performance:.4f}")

            # Create result
            result = TrainingResult(
                job_id=job.job_id,
                performance=performance,
                mixture_weights=job.mixture_weights,
                iteration=job.iteration,
                sample_idx=job.sample_idx,
                status=JobStatus.COMPLETED,
            )

            # Clean up
            del model
            torch.cuda.empty_cache()

            return result

        except Exception as e:
            logger.error(f"Training failed for job {job.job_id}: {e}", exc_info=True)
            return TrainingResult(
                job_id=job.job_id,
                performance=0.0,
                mixture_weights=job.mixture_weights,
                iteration=job.iteration,
                sample_idx=job.sample_idx,
                status=JobStatus.FAILED,
                error_message=str(e),
            )

    def report_result(self, result: TrainingResult):
        """Report training result back to orchestrator via Redis"""
        result_data = {
            "job_id": result.job_id,
            "performance": result.performance,
            "mixture_weights": result.mixture_weights.tolist(),
            "iteration": result.iteration,
            "sample_idx": result.sample_idx,
            "status": result.status.value,
            "error_message": result.error_message,
        }

        # Store result in Redis
        self.redis_client.set_result(result.job_id, result_data)

        # Update job status
        self.redis_client.set_job_status(result.job_id, result.status.value)

        logger.info(f"Reported result for job {result.job_id}")

    def run_worker_loop(self, queue_name: str = "training_queue"):
        """Main worker loop - poll for jobs and execute training"""
        logger.info(f"Starting worker loop, listening on queue: {queue_name}")

        while True:
            try:
                # Block until job available (timeout=30s for health checks)
                job_data = self.redis_client.pop_job(queue_name, timeout=30)

                if job_data is None:
                    logger.debug("No jobs available, waiting...")
                    continue

                # Parse job
                job = TrainingJob(
                    job_id=job_data["job_id"],
                    mixture_weights=np.array(job_data["mixture_weights"]),
                    iteration=job_data["iteration"],
                    sample_idx=job_data["sample_idx"],
                    config=job_data["config"],
                )

                # Update status to running
                self.redis_client.set_job_status(job.job_id, JobStatus.RUNNING.value)

                # Train model
                result = self.train_proxy_model(job)

                # Report result
                self.report_result(result)

            except KeyboardInterrupt:
                logger.info("Worker shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                continue

        logger.info("Worker stopped")
