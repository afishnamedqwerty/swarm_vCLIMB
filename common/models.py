"""
Shared data models for orchestrator-worker communication.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingJobModel(BaseModel):
    """Pydantic model for API serialization"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    job_id: str
    mixture_weights: List[float]
    iteration: int
    sample_idx: int
    config: Dict
    video_paths: Optional[List[str]] = None


class TrainingResultModel(BaseModel):
    """Pydantic model for API serialization"""
    job_id: str
    performance: float
    mixture_weights: List[float]
    iteration: int
    sample_idx: int
    status: JobStatus
    error_message: Optional[str] = None


@dataclass
class TrainingJob:
    """Internal dataclass for training job"""
    job_id: str
    mixture_weights: np.ndarray
    iteration: int
    sample_idx: int
    config: Dict
    video_paths: Optional[List[str]] = None
    status: JobStatus = JobStatus.PENDING

    def to_model(self) -> TrainingJobModel:
        """Convert to Pydantic model for API"""
        return TrainingJobModel(
            job_id=self.job_id,
            mixture_weights=self.mixture_weights.tolist(),
            iteration=self.iteration,
            sample_idx=self.sample_idx,
            config=self.config,
            video_paths=self.video_paths,
        )

    @classmethod
    def from_model(cls, model: TrainingJobModel) -> "TrainingJob":
        """Create from Pydantic model"""
        return cls(
            job_id=model.job_id,
            mixture_weights=np.array(model.mixture_weights),
            iteration=model.iteration,
            sample_idx=model.sample_idx,
            config=model.config,
            video_paths=model.video_paths,
        )


@dataclass
class TrainingResult:
    """Internal dataclass for training result"""
    job_id: str
    performance: float
    mixture_weights: np.ndarray
    iteration: int
    sample_idx: int
    status: JobStatus = JobStatus.COMPLETED
    error_message: Optional[str] = None

    def to_model(self) -> TrainingResultModel:
        """Convert to Pydantic model for API"""
        return TrainingResultModel(
            job_id=self.job_id,
            performance=self.performance,
            mixture_weights=self.mixture_weights.tolist(),
            iteration=self.iteration,
            sample_idx=self.sample_idx,
            status=self.status,
            error_message=self.error_message,
        )

    @classmethod
    def from_model(cls, model: TrainingResultModel) -> "TrainingResult":
        """Create from Pydantic model"""
        return cls(
            job_id=model.job_id,
            performance=model.performance,
            mixture_weights=np.array(model.mixture_weights),
            iteration=model.iteration,
            sample_idx=model.sample_idx,
            status=model.status,
            error_message=model.error_message,
        )


@dataclass
class JobTracker:
    """Track job status across iterations"""
    jobs: Dict[str, TrainingJob] = field(default_factory=dict)
    results: Dict[str, TrainingResult] = field(default_factory=dict)

    def add_job(self, job: TrainingJob):
        self.jobs[job.job_id] = job

    def update_result(self, result: TrainingResult):
        self.results[result.job_id] = result
        if result.job_id in self.jobs:
            self.jobs[result.job_id].status = result.status

    def get_pending_jobs(self) -> List[TrainingJob]:
        return [j for j in self.jobs.values() if j.status == JobStatus.PENDING]

    def get_completed_jobs(self) -> List[TrainingJob]:
        return [j for j in self.jobs.values() if j.status == JobStatus.COMPLETED]

    def is_iteration_complete(self, iteration: int) -> bool:
        iter_jobs = [j for j in self.jobs.values() if j.iteration == iteration]
        if not iter_jobs:
            return False
        return all(j.status == JobStatus.COMPLETED for j in iter_jobs)

    def get_iteration_results(self, iteration: int) -> List[TrainingResult]:
        return [r for r in self.results.values() if r.iteration == iteration]
