"""
Orchestrator API server for distributed CLIMB training.
Provides endpoints to start training, check status, and retrieve results.
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import RedisClient
from orchestrator.distributed_loop import DistributedCLIMBOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CLIMB Orchestrator", version="1.0.0")

# Global state
orchestrator: Optional[DistributedCLIMBOrchestrator] = None
redis_client: Optional[RedisClient] = None
bootstrap_task: Optional[asyncio.Task] = None
bootstrap_result = None
is_training = False


class BootstrapRequest(BaseModel):
    """Request model for starting bootstrap training"""
    embeddings_path: str = "embeddings.npy"
    assignments_path: str = "assignments.npy"
    labels_path: str = "labels.json"
    video_list_path: str = "video_list.txt"
    iterations: int = 3
    samples_per_iteration: list[int] = [64, 32, 16]
    top_n: int = 100
    candidate_pool: int = 1000
    config: Dict = {}


class StatusResponse(BaseModel):
    """Response model for status endpoint"""
    is_training: bool
    orchestrator_status: Optional[Dict] = None
    bootstrap_complete: bool = False
    result: Optional[Dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize Redis client on startup"""
    global redis_client

    logger.info("Starting orchestrator API server...")

    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    redis_client = RedisClient(host=redis_host, port=redis_port)

    logger.info("Orchestrator API server started successfully")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "CLIMB Orchestrator", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes"""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not initialized")

    try:
        redis_client.ping()
        return JSONResponse(content={"status": "healthy", "redis": "connected"})
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis connection failed: {e}")


@app.post("/bootstrap/start")
async def start_bootstrap(request: BootstrapRequest, background_tasks: BackgroundTasks):
    """Start distributed CLIMB bootstrap training"""
    global orchestrator, is_training, bootstrap_result

    if is_training:
        raise HTTPException(status_code=400, detail="Training already in progress")

    logger.info("Received bootstrap start request")

    # Load data
    data_path = Path(os.getenv("DATA_PATH", "/data"))

    try:
        # Load embeddings and assignments
        embeddings = np.load(str(data_path / request.embeddings_path))
        assignments = np.load(str(data_path / request.assignments_path))

        # Load labels
        with open(data_path / request.labels_path, "r") as f:
            labels = json.load(f)

        # Load video list
        with open(data_path / request.video_list_path, "r") as f:
            video_paths = [line.strip() for line in f if line.strip()]

        logger.info(
            f"Loaded data: {len(embeddings)} embeddings, "
            f"{len(labels)} labels, {len(video_paths)} videos"
        )

        # Build videos_by_cluster
        videos_by_cluster = {}
        for path, cid in zip(video_paths, assignments):
            cluster_id = int(cid)
            if cluster_id not in videos_by_cluster:
                videos_by_cluster[cluster_id] = []
            videos_by_cluster[cluster_id].append(path)

        # Prepare config
        config = {
            "input_dim": embeddings.shape[1],
            "num_classes": len(set(labels.values())),
            "hidden_dim": request.config.get("hidden_dim", 256),
            "epochs": request.config.get("epochs", 2),
            "batch_size": request.config.get("batch_size", 64),
            "lr": request.config.get("lr", 1e-3),
            "weight_decay": request.config.get("weight_decay", 1e-4),
            "tokens_budget": request.config.get("tokens_budget", 1_000_000_000),
        }

        # Create orchestrator
        orchestrator = DistributedCLIMBOrchestrator(
            videos_by_cluster=videos_by_cluster,
            redis_client=redis_client,
            config=config,
        )

        # Start bootstrap in background
        is_training = True
        bootstrap_result = None

        async def run_bootstrap():
            global is_training, bootstrap_result
            try:
                best_mixture, reverse_map = await orchestrator.run_bootstrap(
                    iterations=request.iterations,
                    samples_per_iteration=request.samples_per_iteration,
                    top_n=request.top_n,
                    candidate_pool=request.candidate_pool,
                )
                bootstrap_result = {
                    "weights": best_mixture.weights.tolist(),
                    "predicted_performance": best_mixture.performance,
                    "cluster_id_map": reverse_map,
                    "status": "completed",
                }
                logger.info("Bootstrap training completed successfully")
            except Exception as e:
                logger.error(f"Bootstrap training failed: {e}", exc_info=True)
                bootstrap_result = {"status": "failed", "error": str(e)}
            finally:
                is_training = False

        background_tasks.add_task(run_bootstrap)

        return JSONResponse(
            content={
                "message": "Bootstrap training started",
                "n_clusters": len(videos_by_cluster),
                "iterations": request.iterations,
            }
        )

    except Exception as e:
        logger.error(f"Failed to start bootstrap: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bootstrap/status", response_model=StatusResponse)
async def get_status():
    """Get current bootstrap training status"""
    global orchestrator, is_training, bootstrap_result

    status = StatusResponse(
        is_training=is_training,
        orchestrator_status=orchestrator.get_status() if orchestrator else None,
        bootstrap_complete=bootstrap_result is not None,
        result=bootstrap_result,
    )

    return status


@app.get("/bootstrap/result")
async def get_result():
    """Get final bootstrap result"""
    global bootstrap_result

    if bootstrap_result is None:
        raise HTTPException(status_code=404, detail="No result available yet")

    return JSONResponse(content=bootstrap_result)


@app.post("/bootstrap/stop")
async def stop_bootstrap():
    """Stop ongoing bootstrap training"""
    global is_training, bootstrap_task

    if not is_training:
        raise HTTPException(status_code=400, detail="No training in progress")

    # Clear job queue
    if orchestrator:
        orchestrator.job_manager.clear_queue()

    is_training = False

    return JSONResponse(content={"message": "Bootstrap training stopped"})


@app.get("/queue/status")
async def queue_status():
    """Get job queue status"""
    if orchestrator is None:
        raise HTTPException(status_code=400, detail="Orchestrator not initialized")

    return JSONResponse(content=orchestrator.job_manager.get_queue_status())


@app.post("/queue/clear")
async def clear_queue():
    """Clear job queue"""
    if orchestrator is None:
        raise HTTPException(status_code=400, detail="Orchestrator not initialized")

    orchestrator.job_manager.clear_queue()
    return JSONResponse(content={"message": "Queue cleared"})


if __name__ == "__main__":
    port = int(os.getenv("ORCHESTRATOR_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
