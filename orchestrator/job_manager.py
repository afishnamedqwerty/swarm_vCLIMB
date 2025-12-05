"""
Job management for distributed training orchestration.
"""
import asyncio
import logging
from typing import Dict, List, Optional

from common import RedisClient, TrainingJob, TrainingResult, JobStatus, JobTracker

logger = logging.getLogger(__name__)


class JobManager:
    """Manages job submission, tracking, and result collection"""

    def __init__(self, redis_client: RedisClient, queue_name: str = "training_queue"):
        self.redis = redis_client
        self.queue_name = queue_name
        self.tracker = JobTracker()

    def submit_job(self, job: TrainingJob) -> bool:
        """Submit job to Redis queue"""
        job_data = {
            "job_id": job.job_id,
            "mixture_weights": job.mixture_weights.tolist(),
            "iteration": job.iteration,
            "sample_idx": job.sample_idx,
            "config": job.config,
        }

        success = self.redis.push_job(self.queue_name, job_data)
        if success:
            self.tracker.add_job(job)
            self.redis.set_job_status(job.job_id, JobStatus.PENDING.value)
            logger.info(f"Submitted job {job.job_id}")
        return success

    def submit_jobs_batch(self, jobs: List[TrainingJob]) -> int:
        """Submit multiple jobs"""
        count = 0
        for job in jobs:
            if self.submit_job(job):
                count += 1
        logger.info(f"Submitted {count}/{len(jobs)} jobs")
        return count

    async def wait_for_results(
        self, job_ids: List[str], poll_interval: float = 2.0, timeout: float = 7200.0
    ) -> List[TrainingResult]:
        """Wait for all jobs to complete and collect results"""
        logger.info(f"Waiting for {len(job_ids)} jobs to complete...")

        start_time = asyncio.get_event_loop().time()
        completed_results = []
        remaining_job_ids = set(job_ids)

        while remaining_job_ids:
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                logger.error(f"Timeout waiting for jobs: {remaining_job_ids}")
                break

            # Check each remaining job
            for job_id in list(remaining_job_ids):
                result_data = self.redis.get_result(job_id)
                if result_data:
                    result = TrainingResult(
                        job_id=result_data["job_id"],
                        performance=result_data["performance"],
                        mixture_weights=result_data["mixture_weights"],
                        iteration=result_data["iteration"],
                        sample_idx=result_data["sample_idx"],
                        status=JobStatus(result_data["status"]),
                        error_message=result_data.get("error_message"),
                    )
                    completed_results.append(result)
                    self.tracker.update_result(result)
                    remaining_job_ids.remove(job_id)
                    logger.info(
                        f"Job {job_id} completed ({len(completed_results)}/{len(job_ids)})"
                    )

            # Poll interval
            if remaining_job_ids:
                await asyncio.sleep(poll_interval)

        logger.info(f"Collected {len(completed_results)} results")
        return completed_results

    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        return {
            "queue_length": self.redis.get_queue_length(self.queue_name),
            "total_jobs": len(self.tracker.jobs),
            "pending_jobs": len(self.tracker.get_pending_jobs()),
            "completed_jobs": len(self.tracker.get_completed_jobs()),
        }

    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get status of specific job"""
        return self.redis.get_job_status(job_id)

    def get_iteration_results(self, iteration: int) -> List[TrainingResult]:
        """Get all results for a specific iteration"""
        return self.tracker.get_iteration_results(iteration)

    def is_iteration_complete(self, iteration: int) -> bool:
        """Check if all jobs for iteration are complete"""
        return self.tracker.is_iteration_complete(iteration)

    def clear_queue(self):
        """Clear all pending jobs"""
        self.redis.clear_queue(self.queue_name)
        logger.info("Cleared job queue")
