"""
api/schemas.py — Pydantic models for the RAGnarok FastAPI endpoints.

These mirror pipeline/schemas.py dataclasses but are Pydantic models for
HTTP serialization/validation. They are intentionally separate so that the
core pipeline has no Pydantic dependency.

Endpoints:
    POST /rank      → RankRequest  →  RankResponse
    GET  /health    → HealthResponse
    GET  /results   → ResultsResponse
    GET  /pipeline/status → PipelineStatusResponse
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ─── REQUEST ──────────────────────────────────────────────────────────────────

class RankRequest(BaseModel):
    """
    Request body for POST /rank.

    Accepts a JSONL string of candidate profiles (same format as candidates.jsonl)
    and an optional job description override. If jd_text is omitted, the pre-parsed
    JD from parsed_job_description.json is used.
    """
    candidates_jsonl: str = Field(
        ...,
        description="JSONL string — one candidate JSON object per line.",
        max_length=50_000_000,   # 50MB hard cap (middleware enforces at byte level too)
    )
    jd_text: Optional[str] = Field(
        default=None,
        description="Optional job description markdown. If omitted, uses pre-parsed JD.",
        max_length=50_000,
    )
    top_k: int = Field(
        default=100,
        ge=1,
        le=100,
        description="Number of top candidates to return (default 100).",
    )

    @field_validator("candidates_jsonl")
    @classmethod
    def must_have_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("candidates_jsonl must not be empty.")
        return v


# ─── SKILL RECORD ─────────────────────────────────────────────────────────────

class SkillOut(BaseModel):
    """Serialized skill for API responses."""
    name: str
    proficiency: str
    endorsements: int
    duration_months: int
    assessment_score: float


# ─── SIGNALS OUT ──────────────────────────────────────────────────────────────

class SignalsOut(BaseModel):
    """Key behavioral signals exposed via API (subset of RedrobSignals)."""
    open_to_work: bool
    notice_period_days: int
    last_active_date: str           # ISO date string
    recruiter_response_rate: float  # 0.0–1.0
    github_activity_score: float    # 0–100 or -1.0
    profile_completeness_score: float
    willing_to_relocate: bool
    preferred_work_mode: str


# ─── COMPONENT SCORES OUT ─────────────────────────────────────────────────────

class ComponentScoresOut(BaseModel):
    """Score breakdown per candidate — used in RankedCandidateOut.components."""
    skill_match_score: float
    career_quality_score: float
    behavioral_score: float
    trajectory_velocity: float
    rrf_score: float
    cross_encoder_score: float
    weighted_sum: float
    location_bonus_applied: float
    uncertainty_penalty_applied: float
    paths_present: list[str]
    hard_disqualifier: bool
    honeypot_override: bool


# ─── ADVOCATE / SKEPTIC SIGNALS ───────────────────────────────────────────────

class AdvocateSignalOut(BaseModel):
    label: str
    confidence: str     # HIGH | MEDIUM | LOW
    value: str


class SkepticSignalOut(BaseModel):
    label: str
    severity: str       # HIGH | MODERATE | LOW
    value: str


class TrustVerdictOut(BaseModel):
    verdict: str                        # ROBUST | CONTESTED | FRAGILE
    flip_risk: str                      # LOW | MEDIUM | HIGH
    confidence_pct: float               # 0–100
    advocate_signals: list[AdvocateSignalOut]
    skeptic_signals: list[SkepticSignalOut]
    falsifiability: list[str]


# ─── RANKED CANDIDATE OUT ─────────────────────────────────────────────────────

class RankedCandidateOut(BaseModel):
    """
    One row in the ranked output — maps to one CSV row in submission.csv.
    Rich fields (components, trust, profile) are included for the UI.
    """
    candidate_id: str
    rank: int
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

    # Rich fields (present when pipeline runs in full mode)
    components: Optional[ComponentScoresOut] = None
    trust: Optional[TrustVerdictOut] = None

    # Profile summary (for UI display)
    name: Optional[str] = None
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    location: Optional[str] = None
    years_of_experience: Optional[float] = None
    skills: list[SkillOut] = []
    signals: Optional[SignalsOut] = None
    is_honeypot: bool = False


# ─── RESPONSE ─────────────────────────────────────────────────────────────────

class RankResponse(BaseModel):
    """Response body for POST /rank."""
    status: str                             # "success" | "error"
    total_candidates_input: int
    honeypots_removed: int
    ranked: list[RankedCandidateOut]
    pipeline_elapsed_ms: float
    stage_timings: dict[str, float]         # stage_name → elapsed_ms
    errors: list[str] = []                  # Non-fatal warnings from the run


class ResultsResponse(BaseModel):
    """Response body for GET /results — returns last pipeline run results."""
    status: str
    run_id: Optional[str] = None
    ranked: list[RankedCandidateOut] = []
    pipeline_elapsed_ms: Optional[float] = None
    message: str = ""


# ─── HEALTH ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str             # "healthy" | "degraded" | "unhealthy"
    pipeline_ready: bool
    indexes_loaded: dict[str, bool]     # faiss, bm25, feature_store, trajectory, honeypot
    version: str


# ─── PIPELINE STATUS ──────────────────────────────────────────────────────────

class PipelineStatusResponse(BaseModel):
    """Response body for GET /pipeline/status."""
    is_running: bool
    last_run_id: Optional[str] = None
    last_run_elapsed_ms: Optional[float] = None
    last_run_candidate_count: Optional[int] = None
    last_run_honeypots: Optional[int] = None
    stage_timings: dict[str, float] = {}


# ─── CSV EXPORT ───────────────────────────────────────────────────────────────

class CSVExportRequest(BaseModel):
    """Request to export results as submission.csv (POST /export/csv)."""
    run_id: Optional[str] = Field(
        default=None,
        description="If provided, exports that specific run. Otherwise exports the latest.",
    )
    validate_before_export: bool = Field(
        default=True,
        description="Run validate_submission.py checks before returning the CSV.",
    )
