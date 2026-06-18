"""
api/main.py — FastAPI application entry point for RAGnarok ATS.

Start with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

UI:         http://localhost:8000        (ats_platform.html served here)
Swagger UI: http://localhost:8000/docs
ReDoc:      http://localhost:8000/redoc
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse

# ── Logging config ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Lifespan: startup / shutdown tasks ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup: log readiness. Future: warm up FAISS / BM25 index caches.
    On shutdown: clean up resources.
    """
    logger.info("RAGnarok ATS API starting up…")
    logger.info("UI:         http://localhost:8000")
    logger.info("Swagger UI: http://localhost:8000/docs")
    logger.info("ReDoc:      http://localhost:8000/redoc")

    # Optionally warm-up indexes here (uncomment when indexes exist):
    # from indexing.faiss_builder import FAISSBuilder
    # FAISSBuilder.load()

    yield

    logger.info("RAGnarok ATS API shutting down.")


# ── Create FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(
    title="RAGnarok ATS — Intelligent Candidate Ranking API",
    description=(
        "Enterprise-grade candidate ranking pipeline for the Redrob Hackathon. "
        "5-path retrieval (semantic + keyword + ontology + trajectory + signal) → "
        "RRF fusion → cross-encoder rerank → composite scoring → adversarial trust layer.\n\n"
        "**Constraints**: CPU-only · ≤5 min ranking window · No network during ranking · "
        "100K candidate pool."
    ),
    version="1.0.0",
    contact={
        "name": "Krishna Zalavadiya (DEV A)",
        "url": "https://github.com/krishna-zalavadiya/RAGnarok",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# ── Register middleware ─────────────────────────────────────────────────────────
from api.middleware import register_middleware
register_middleware(app)

# ── Register routers ───────────────────────────────────────────────────────────
from api.routes import health, rank as rank_router

app.include_router(health.router)
app.include_router(rank_router.router)


# ── Serve UI (ats_platform.html) at GET / ─────────────────────────────────────
_UI_HTML = Path(__file__).parent.parent / "ui" / "ats_platform.html"

@app.get("/", include_in_schema=False)
async def root():
    """Serve the ATS platform UI."""
    if _UI_HTML.exists():
        return FileResponse(str(_UI_HTML), media_type="text/html")
    # Fallback info if HTML not found
    return JSONResponse({
        "service": "RAGnarok ATS",
        "version": "1.0.0",
        "ui": "ui/ats_platform.html not found",
        "docs": "/docs",
        "health": "/health",
        "rank": "POST /rank",
        "results": "GET /results",
        "export_csv": "POST /export/csv",
    })
