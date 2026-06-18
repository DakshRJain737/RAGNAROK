"""
api/middleware.py — FastAPI middleware for the RAGnarok ranking API.

Responsibilities:
    1. Request size limit: reject uploads > MAX_REQUEST_BYTES (50 MB).
    2. Simple rate limiting: max MAX_REQUESTS_PER_MINUTE per client IP.
    3. Request timing: adds X-Process-Time header to every response.
    4. Error normalization: wraps unhandled exceptions in JSON responses.
    5. CORS: allow localhost + HuggingFace Spaces origins for sandbox demo.
"""

from __future__ import annotations

import time
import logging
from collections import defaultdict, deque
from typing import Callable

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MAX_REQUEST_BYTES     = 50 * 1024 * 1024   # 50 MB
MAX_REQUESTS_PER_MIN  = 30                  # per unique IP
RATE_WINDOW_SECONDS   = 60


# ─── CORS ORIGINS ─────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:8000",        # FastAPI local — serves ats_platform.html
    "http://127.0.0.1:8000",
    "null",                         # file:// origin (local HTML open)
    "https://*.hf.space",           # HuggingFace Spaces
]


# ─── SIZE LIMIT MIDDLEWARE ─────────────────────────────────────────────────────
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Reject requests with Content-Length > MAX_REQUEST_BYTES before the body
    is read. Falls back to streaming check when Content-Length is absent.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BYTES:
            logger.warning(
                "Request rejected: Content-Length %s > %d bytes",
                content_length, MAX_REQUEST_BYTES,
            )
            return JSONResponse(
                status_code=413,
                content={
                    "error": "request_too_large",
                    "message": (
                        f"Request body exceeds the {MAX_REQUEST_BYTES // (1024*1024)} MB limit. "
                        f"Compress your candidates JSONL or split it into smaller batches."
                    ),
                },
            )
        return await call_next(request)


# ─── RATE LIMIT MIDDLEWARE ────────────────────────────────────────────────────
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory sliding-window rate limiter per client IP.
    Applies only to POST /rank (the expensive endpoint).
    """

    def __init__(self, app, max_requests: int = MAX_REQUESTS_PER_MIN,
                 window_seconds: int = RATE_WINDOW_SECONDS) -> None:
        super().__init__(app)
        self._max     = max_requests
        self._window  = window_seconds
        self._buckets: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.method == "POST" and "/rank" in request.url.path:
            ip    = request.client.host if request.client else "unknown"
            now   = time.monotonic()
            queue = self._buckets[ip]

            # Evict timestamps outside the window
            while queue and now - queue[0] > self._window:
                queue.popleft()

            if len(queue) >= self._max:
                retry_after = int(self._window - (now - queue[0])) + 1
                logger.warning("Rate limit exceeded for IP %s", ip)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": (
                            f"Too many requests. Max {self._max} POST /rank requests "
                            f"per {self._window}s. Retry after {retry_after}s."
                        ),
                    },
                    headers={"Retry-After": str(retry_after)},
                )
            queue.append(now)

        return await call_next(request)


# ─── TIMING MIDDLEWARE ─────────────────────────────────────────────────────────
class TimingMiddleware(BaseHTTPMiddleware):
    """Adds X-Process-Time header (ms) to every response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t0       = time.perf_counter()
        response = await call_next(request)
        elapsed  = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Process-Time"] = f"{elapsed:.1f}ms"
        return response


# ─── ERROR NORMALISATION MIDDLEWARE ───────────────────────────────────────────
class ErrorNormalizationMiddleware(BaseHTTPMiddleware):
    """
    Catch unhandled exceptions and return a consistent JSON error envelope.
    FastAPI handles HTTPExceptions itself; this catches everything else.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            logger.exception(
                "Unhandled exception on %s %s", request.method, request.url.path
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": str(exc),
                    "path": str(request.url.path),
                },
            )


# ─── HELPER: register all middleware on a FastAPI app ─────────────────────────
def register_middleware(app) -> None:
    """
    Register all middleware on the FastAPI application.

    Call this in api/main.py BEFORE the first request handler is registered.
    Middleware executes in REVERSE registration order (last registered = outermost).
    """
    # Outermost: CORS (must be first so preflight OPTIONS requests are handled)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_origin_regex=r"https://.*\.hf\.space",
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    # Error normalisation wraps everything else
    app.add_middleware(ErrorNormalizationMiddleware)
    # Timing on all requests
    app.add_middleware(TimingMiddleware)
    # Rate limiting on POST /rank
    app.add_middleware(RateLimitMiddleware)
    # Size limit (checked before body is read)
    app.add_middleware(RequestSizeLimitMiddleware)

    logger.info(
        "Middleware registered: CORS, ErrorNormalization, Timing, "
        "RateLimit(%d/min), SizeLimit(%dMB).",
        MAX_REQUESTS_PER_MIN,
        MAX_REQUEST_BYTES // (1024 * 1024),
    )
