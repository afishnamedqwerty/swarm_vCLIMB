"""
Redis client for job queue management.
"""
import json
import logging
from typing import Optional

import redis


logger = logging.getLogger(__name__)


class RedisClient:
    """Redis client wrapper for job queue operations"""

    def __init__(self, host: str = "redis", port: int = 6379, db: int = 0):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )
        self.ping()

    def ping(self):
        """Check Redis connection"""
        try:
            self.client.ping()
            logger.info("Redis connection established")
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def push_job(self, queue_name: str, job_data: dict) -> bool:
        """Push job to queue (FIFO)"""
        try:
            serialized = json.dumps(job_data)
            self.client.lpush(queue_name, serialized)
            logger.info(f"Pushed job {job_data.get('job_id')} to {queue_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to push job: {e}")
            return False

    def pop_job(self, queue_name: str, timeout: int = 0) -> Optional[dict]:
        """Pop job from queue (blocking)"""
        try:
            result = self.client.brpop(queue_name, timeout=timeout)
            if result:
                _, serialized = result
                job_data = json.loads(serialized)
                logger.info(f"Popped job {job_data.get('job_id')} from {queue_name}")
                return job_data
            return None
        except Exception as e:
            logger.error(f"Failed to pop job: {e}")
            return None

    def get_queue_length(self, queue_name: str) -> int:
        """Get number of jobs in queue"""
        return self.client.llen(queue_name)

    def set_result(self, job_id: str, result_data: dict, ttl: int = 3600):
        """Store job result with TTL"""
        try:
            key = f"result:{job_id}"
            serialized = json.dumps(result_data)
            self.client.setex(key, ttl, serialized)
            logger.info(f"Stored result for job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store result: {e}")
            return False

    def get_result(self, job_id: str) -> Optional[dict]:
        """Retrieve job result"""
        try:
            key = f"result:{job_id}"
            serialized = self.client.get(key)
            if serialized:
                return json.loads(serialized)
            return None
        except Exception as e:
            logger.error(f"Failed to get result: {e}")
            return None

    def set_job_status(self, job_id: str, status: str, ttl: int = 7200):
        """Update job status"""
        try:
            key = f"status:{job_id}"
            self.client.setex(key, ttl, status)
            return True
        except Exception as e:
            logger.error(f"Failed to set status: {e}")
            return False

    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get job status"""
        try:
            key = f"status:{job_id}"
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return None

    def clear_queue(self, queue_name: str):
        """Clear all jobs from queue"""
        self.client.delete(queue_name)
        logger.info(f"Cleared queue {queue_name}")

    def get_all_job_ids(self, pattern: str = "status:*") -> list:
        """Get all job IDs matching pattern"""
        keys = self.client.keys(pattern)
        return [k.replace("status:", "") for k in keys]
