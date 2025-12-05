"""
Worker main entry point with FastAPI health check endpoint.
"""
import asyncio
import logging
import os
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))

from worker.training_service import TrainingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Proxy Training Worker", version="1.0.0")

# Global training service instance
training_service = None
worker_thread = None


@app.on_event("startup")
async def startup_event():
    """Initialize training service on startup"""
    global training_service, worker_thread

    logger.info("Starting worker service...")

    # Get configuration from environment
    data_path = os.getenv("DATA_PATH", "/data")
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")

    # Initialize service
    training_service = TrainingService(
        data_path=data_path,
        redis_host=redis_host,
        redis_port=redis_port,
        orchestrator_url=orchestrator_url,
    )

    # Load shared data
    embeddings_path = os.getenv("EMBEDDINGS_FILE", "embeddings.npy")
    labels_path = os.getenv("LABELS_FILE", "labels.json")
    assignments_path = os.getenv("ASSIGNMENTS_FILE", "assignments.npy")

    try:
        training_service.load_shared_data(embeddings_path, labels_path, assignments_path)
    except Exception as e:
        logger.error(f"Failed to load shared data: {e}")
        # Don't fail startup - worker can still respond to health checks

    # Start worker loop in background thread
    worker_thread = threading.Thread(
        target=training_service.run_worker_loop,
        kwargs={"queue_name": os.getenv("QUEUE_NAME", "training_queue")},
        daemon=True,
    )
    worker_thread.start()

    logger.info("Worker service started successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes"""
    if training_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return JSONResponse(
        content={
            "status": "healthy",
            "device": str(training_service.device),
            "data_loaded": len(training_service.embeddings) > 0,
        }
    )


@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint"""
    if training_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    queue_length = training_service.redis_client.get_queue_length("training_queue")

    return JSONResponse(
        content={
            "embeddings_loaded": len(training_service.embeddings),
            "labels_loaded": len(training_service.labels),
            "clusters_count": len(training_service.videos_by_cluster),
            "queue_length": queue_length,
        }
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "Proxy Training Worker", "version": "1.0.0"}


if __name__ == "__main__":
    port = int(os.getenv("WORKER_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
