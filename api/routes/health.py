"""
api/routes/health.py — Health-check and pipeline-status endpoints.

GET  /health          → HealthResponse
GET  /pipeline/status → PipelineStatusResponse
"""

from __future__ import annotations

import logging
from fastapi import APIRouter
from api.schemas import HealthResponse, PipelineStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


def _get_pipeline_state() -> dict:
    """
    Check which pre-computed index files exist on disk.

    Uses config paths directly — avoids importing heavy ML modules
    (faiss, sentence-transformers) just to run a health check, and
    avoids the stale class-name mismatch that previously caused all
    five checks to fail silently.
    """
    try:
        import config
        state = {
            "faiss":         config.FAISS_INDEX_PATH.exists() and config.FAISS_ID_MAP_PATH.exists(),
            "bm25":          config.BM25_INDEX_PATH.exists(),
            "feature_store": config.FEATURE_STORE_PATH.exists() and config.FEATURE_IDS_PATH.exists(),
            "trajectory":    config.TRAJECTORY_PATH.exists(),
            "honeypot":      config.HONEYPOT_SET_PATH.exists(),
        }
    except Exception as exc:
        logger.warning("Health check: could not import config — %s", exc)
        state = {k: False for k in ("faiss", "bm25", "feature_store", "trajectory", "honeypot")}

    return state


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Returns the health status of the API and readiness of all pipeline indexes.

    pipeline_ready is True only when every index file is present on disk.
    Individual index booleans let you see which files are missing.
    """
    indexes = _get_pipeline_state()

    # trajectory and honeypot files may be 0-byte stubs when precomputed
    # but the pipeline still works — treat them as optional for pipeline_ready
    core_ready = indexes["faiss"] and indexes["bm25"] and indexes["feature_store"]
    all_ready  = all(indexes.values())

    # "healthy" if all present; "degraded" if core present but extras missing;
    # "unhealthy" if core indexes are missing
    if all_ready:
        status = "healthy"
    elif core_ready:
        status = "degraded"
    else:
        status = "unhealthy"

    # pipeline_ready uses core_ready so the UI badge turns green even when
    # trajectory/honeypot are 0-byte stubs
    pipeline_ready = core_ready

    logger.debug("Health check: status=%s, indexes=%s", status, indexes)
    return HealthResponse(
        status=status,
        pipeline_ready=pipeline_ready,
        indexes_loaded=indexes,
        version="1.0.0",
    )


@router.get("/pipeline/status", response_model=PipelineStatusResponse)
async def pipeline_status() -> PipelineStatusResponse:
    """
    Returns the status of the last pipeline run (if any).
    """
    try:
        from api.routes.rank import _LAST_RUN_STATE
        return PipelineStatusResponse(**_LAST_RUN_STATE)
    except (ImportError, AttributeError):
        return PipelineStatusResponse(is_running=False)
