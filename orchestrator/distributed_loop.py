"""
Distributed CLIMB bootstrapping loop using Kubernetes workers.
"""
import asyncio
import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

from common import RedisClient, TrainingJob, JobStatus
from orchestrator.job_manager import JobManager
from sampler.dirichlet_sampler import MixtureSampler
from sampler.predictor import PerformancePredictor
from sampler.proxy import VideoMixture

logger = logging.getLogger(__name__)


class DistributedCLIMBOrchestrator:
    """
    Orchestrates distributed CLIMB bootstrapping loop across Kubernetes worker pods.

    Replaces the sequential training in sampler/loop.py with parallel distributed training.
    """

    def __init__(
        self,
        videos_by_cluster: Dict[int, Sequence[str]],
        redis_client: RedisClient,
        config: Dict,
        queue_name: str = "training_queue",
    ):
        """
        Args:
            videos_by_cluster: Mapping of cluster ID to video paths
            redis_client: Redis client for job queue
            config: Configuration dict with training parameters
            queue_name: Redis queue name for jobs
        """
        self.videos_by_cluster = videos_by_cluster
        self.config = config
        self.job_manager = JobManager(redis_client, queue_name)

        # Remap clusters to contiguous IDs
        self.remapped, self.reverse_map = self._contiguous_clusters(videos_by_cluster)
        self.n_clusters = len(self.remapped)
        self.cluster_sizes = np.array([len(self.remapped[cid]) for cid in range(self.n_clusters)])

        # Initialize sampler and predictor
        self.sampler = MixtureSampler(self.cluster_sizes)
        self.predictor = PerformancePredictor()

        # History tracking
        self.history_weights = []
        self.history_perf = []

        logger.info(
            f"Initialized orchestrator with {self.n_clusters} clusters, "
            f"{len(videos_by_cluster)} total videos"
        )

    @staticmethod
    def _contiguous_clusters(
        videos_by_cluster: Dict[int, Sequence[str]]
    ) -> Tuple[Dict[int, Sequence[str]], Dict[int, int]]:
        """Re-index arbitrary cluster ids into a contiguous 0..N-1 mapping"""
        ordered = sorted(videos_by_cluster.keys())
        id_map = {cid: idx for idx, cid in enumerate(ordered)}
        remapped = {id_map[cid]: vids for cid, vids in videos_by_cluster.items()}
        reverse = {v: k for k, v in id_map.items()}
        return remapped, reverse

    async def run_bootstrap(
        self,
        iterations: int = 3,
        samples_per_iteration: Sequence[int] = (64, 32, 16),
        top_n: int = 100,
        candidate_pool: int = 1000,
        timeout_per_job: float = 7200.0,
    ) -> Tuple[VideoMixture, Dict[int, int]]:
        """
        Run distributed CLIMB bootstrapping loop.

        Args:
            iterations: Number of CLIMB iterations
            samples_per_iteration: Number of mixtures to sample per iteration
            top_n: Number of top predictions to sample from (after iteration 0)
            candidate_pool: Size of candidate pool for predictor sampling
            timeout_per_job: Timeout in seconds for each training job

        Returns:
            best_mixture: Predicted best mixture after final iteration
            reverse_cluster_map: Mapping from contiguous cluster IDs to original IDs
        """
        logger.info(
            f"Starting distributed CLIMB bootstrap: {iterations} iterations, "
            f"samples={samples_per_iteration}"
        )

        if iterations > len(samples_per_iteration):
            raise ValueError("samples_per_iteration must have >= iterations entries")

        for it in range(iterations):
            logger.info(f"=== Iteration {it + 1}/{iterations} ===")

            n_samples = samples_per_iteration[it]

            # Sample mixture weights
            if it == 0 or self.predictor.model is None:
                # First iteration: pure Dirichlet sampling
                weights = self.sampler.sample(n_samples)
                logger.info(f"Sampled {n_samples} mixtures via Dirichlet")
            else:
                # Subsequent iterations: predictor-guided sampling
                candidates = self.sampler.sample(candidate_pool)
                predicted = self.predictor.predict(candidates)
                chosen_idx = self.sampler.sample_from_top(predicted, top_n=top_n, n_samples=n_samples)
                weights = candidates[chosen_idx]
                logger.info(
                    f"Sampled {n_samples} mixtures from top-{top_n} of {candidate_pool} candidates"
                )

            # Submit training jobs to workers
            jobs = []
            for sample_idx, w in enumerate(weights):
                job = TrainingJob(
                    job_id=f"iter{it}_sample{sample_idx}",
                    mixture_weights=w,
                    iteration=it,
                    sample_idx=sample_idx,
                    config=self.config,
                    status=JobStatus.PENDING,
                )
                jobs.append(job)

            # Submit all jobs
            submitted_count = self.job_manager.submit_jobs_batch(jobs)
            logger.info(f"Submitted {submitted_count} training jobs to worker pool")

            # Wait for all jobs to complete
            job_ids = [job.job_id for job in jobs]
            results = await self.job_manager.wait_for_results(
                job_ids, poll_interval=5.0, timeout=timeout_per_job
            )

            # Extract performances
            performances = []
            for result in results:
                if result.status == JobStatus.COMPLETED:
                    performances.append(result.performance)
                    self.history_weights.append(result.mixture_weights)
                    self.history_perf.append(result.performance)
                else:
                    logger.warning(
                        f"Job {result.job_id} failed: {result.error_message}"
                    )
                    performances.append(0.0)  # Assign worst performance

            # Update predictor
            if performances:
                self.predictor.update(np.stack(weights), np.asarray(performances))
                avg_perf = np.mean(performances)
                best_perf = np.max(performances)
                logger.info(
                    f"Iteration {it + 1} complete: avg_perf={avg_perf:.4f}, best_perf={best_perf:.4f}"
                )
            else:
                logger.error(f"No valid results for iteration {it + 1}")

        # Final prediction
        logger.info("Computing final best mixture prediction...")
        final_candidates = self.sampler.sample(candidate_pool * 10)
        final_predictions = self.predictor.predict(final_candidates)
        best_idx = int(np.argmax(final_predictions))
        best_weights = final_candidates[best_idx]
        best_mixture = VideoMixture(
            weights=best_weights, performance=final_predictions[best_idx]
        )

        logger.info(
            f"Bootstrap complete! Best predicted performance: {best_mixture.performance:.4f}"
        )

        return best_mixture, self.reverse_map

    def get_status(self) -> Dict:
        """Get current orchestrator status"""
        return {
            "n_clusters": self.n_clusters,
            "total_history": len(self.history_perf),
            "queue_status": self.job_manager.get_queue_status(),
            "predictor_fitted": self.predictor.model is not None,
        }
