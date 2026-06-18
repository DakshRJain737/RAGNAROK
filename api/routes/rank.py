"""
api/routes/rank.py — Core ranking endpoint for the RAGnarok ATS pipeline.

POST /rank   → accepts candidates JSONL + optional JD override
             → runs the full 5-path pipeline
             → returns RankResponse with ranked candidates + trust verdicts

GET  /results → returns the last run's ranked results (cached in memory)
POST /export/csv → export last results as submission.csv bytes
"""

from __future__ import annotations

import io
import csv
import json
import time
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import (
    RankRequest,
    RankResponse,
    ResultsResponse,
    RankedCandidateOut,
    ComponentScoresOut,
    TrustVerdictOut,
    AdvocateSignalOut,
    SkepticSignalOut,
    SkillOut,
    SignalsOut,
    CSVExportRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rank"])

# ── In-memory state for the last pipeline run ─────────────────────────────────
_LAST_RUN_STATE: dict = {
    "is_running": False,
    "last_run_id": None,
    "last_run_elapsed_ms": None,
    "last_run_candidate_count": None,
    "last_run_honeypots": None,
    "stage_timings": {},
}
_LAST_RESULTS: list[RankedCandidateOut] = []


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _load_pipeline_modules():
    """
    Lazy-load all heavy pipeline modules.
    Returns a dict of module references so caller can use them.
    Raises ImportError with a descriptive message if any module is missing.
    """
    try:
        from pipeline.candidate_parser import CandidateParser
        from pipeline.jd_parser import JDParser
        from pipeline.runner import PipelineRunner
        return {
            "CandidateParser": CandidateParser,
            "JDParser": JDParser,
            "PipelineRunner": PipelineRunner,
        }
    except ImportError as exc:
        raise ImportError(
            f"Pipeline modules not available: {exc}. "
            "Ensure precompute.py has been run and all dependencies are installed."
        ) from exc


def _cfv_to_signals_out(signals) -> Optional[SignalsOut]:
    """Convert RedrobSignals to SignalsOut."""
    try:
        return SignalsOut(
            open_to_work=signals.open_to_work_flag,
            notice_period_days=signals.notice_period_days,
            last_active_date=signals.last_active_date.isoformat(),
            recruiter_response_rate=signals.recruiter_response_rate,
            github_activity_score=signals.github_activity_score,
            profile_completeness_score=signals.profile_completeness_score,
            willing_to_relocate=signals.willing_to_relocate,
            preferred_work_mode=signals.preferred_work_mode,
        )
    except Exception:
        return None


def _build_ranked_out(rc) -> RankedCandidateOut:
    """
    Convert a pipeline RankedCandidate dataclass to the API RankedCandidateOut
    Pydantic model, pulling from the embedded feature_vector, components, and
    trust fields.
    """
    cfv        = rc.feature_vector
    components = rc.components
    trust      = rc.trust

    # Component scores
    comp_out: Optional[ComponentScoresOut] = None
    if components is not None:
        comp_out = ComponentScoresOut(
            skill_match_score=components.skill_match_score,
            career_quality_score=components.career_quality_score,
            behavioral_score=components.behavioral_score,
            trajectory_velocity=components.trajectory_velocity,
            rrf_score=components.rrf_score,
            cross_encoder_score=components.cross_encoder_score,
            weighted_sum=components.weighted_sum,
            location_bonus_applied=components.location_bonus_applied,
            uncertainty_penalty_applied=components.uncertainty_penalty_applied,
            paths_present=components.paths_present,
            hard_disqualifier=components.hard_disqualifier,
            honeypot_override=components.honeypot_override,
        )

    # Trust verdict
    trust_out: Optional[TrustVerdictOut] = None
    if trust is not None:
        trust_out = TrustVerdictOut(
            verdict=trust.verdict,
            flip_risk=trust.flip_risk,
            confidence_pct=trust.confidence_pct,
            advocate_signals=[
                AdvocateSignalOut(label=s.label, confidence=s.confidence, value=s.value)
                for s in trust.advocate_signals
            ],
            skeptic_signals=[
                SkepticSignalOut(label=s.label, severity=s.severity, value=s.value)
                for s in trust.skeptic_signals
            ],
            falsifiability=trust.falsifiability,
        )

    # Skills (from feature vector)
    skills_out = []
    if cfv is not None:
        skills_out = [
            SkillOut(
                name=s.name_raw,
                proficiency=s.proficiency,
                endorsements=s.endorsements,
                duration_months=s.duration_months,
                assessment_score=s.assessment_score,
            )
            for s in cfv.skills
        ]

    return RankedCandidateOut(
        candidate_id=rc.candidate_id,
        rank=rc.rank,
        score=rc.final_score,
        reasoning=rc.reasoning,
        components=comp_out,
        trust=trust_out,
        name=cfv.headline if cfv else None,
        current_title=cfv.current_title if cfv else None,
        current_company=cfv.current_company if cfv else None,
        location=cfv.location if cfv else None,
        years_of_experience=cfv.years_of_experience if cfv else None,
        skills=skills_out,
        signals=_cfv_to_signals_out(cfv.signals) if cfv else None,
        is_honeypot=cfv.is_honeypot if cfv else False,
    )


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@router.post("/rank", response_model=RankResponse)
async def rank_candidates(request: RankRequest) -> RankResponse:
    """
    Run the full RAGnarok ranking pipeline on the submitted candidates.

    - Parses candidate JSONL
    - Runs 5-path retrieval (semantic, keyword, ontology, trajectory, signal)
    - Applies RRF fusion → honeypot filter → cross-encoder rerank
    - Computes composite score (0.40×skill + 0.35×career + 0.25×behavioral)
    - Runs adversarial trust layer (advocate + skeptic + verdict)
    - Returns top-K ranked candidates with full score breakdown

    Returns 200 with RankResponse on success.
    Returns 422 on validation errors, 429 on rate limit, 413 on size limit.
    """
    global _LAST_RUN_STATE, _LAST_RESULTS

    if _LAST_RUN_STATE["is_running"]:
        raise HTTPException(
            status_code=409,
            detail="A pipeline run is already in progress. Please wait for it to complete.",
        )

    run_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()
    stage_timings: dict[str, float] = {}
    errors: list[str] = []

    _LAST_RUN_STATE["is_running"] = True
    logger.info("Pipeline run %s started.", run_id)

    try:
        # ── 1. Load pipeline modules ───────────────────────────────────────
        t0 = time.perf_counter()
        try:
            mods = _load_pipeline_modules()
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        stage_timings["module_load"] = (time.perf_counter() - t0) * 1000

        # ── 2. Parse candidate JSONL ──────────────────────────────────────
        t0 = time.perf_counter()
        parser = mods["CandidateParser"]()
        candidates = []
        parse_errors = 0
        for line_no, line in enumerate(request.candidates_jsonl.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                candidates.append(parser.parse_candidate(item))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                parse_errors += 1
                errors.append(f"Line {line_no}: {exc}")
                if parse_errors > 50:
                    errors.append("Too many parse errors — stopping early.")
                    break
        stage_timings["candidate_parse"] = (time.perf_counter() - t0) * 1000

        if not candidates:
            raise HTTPException(
                status_code=422,
                detail="No valid candidates could be parsed from candidates_jsonl.",
            )

        total_input = len(candidates)
        logger.info("Parsed %d candidates (%d errors).", total_input, parse_errors)

        # ── 3. Parse JD ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        jd_parser = mods["JDParser"]()
        if request.jd_text:
            jd_intent = jd_parser.parse(request.jd_text)
        else:
            jd_intent = jd_parser.load_parsed()   # loads parsed_job_description.json
        stage_timings["jd_parse"] = (time.perf_counter() - t0) * 1000

        # ── 4. Run full pipeline ──────────────────────────────────────────
        t0 = time.perf_counter()
        runner = mods["PipelineRunner"](jd=jd_intent, candidates=candidates)
        ranked_candidates, run_stage_timings = runner.run(top_k=request.top_k)
        stage_timings.update(run_stage_timings)
        stage_timings["pipeline_total"] = (time.perf_counter() - t0) * 1000

        # ── 5. Serialise results ──────────────────────────────────────────
        t0 = time.perf_counter()
        ranked_out = [_build_ranked_out(rc) for rc in ranked_candidates]
        stage_timings["serialization"] = (time.perf_counter() - t0) * 1000

        honeypots_removed = sum(1 for rc in ranked_candidates if rc.feature_vector and rc.feature_vector.is_honeypot)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # ── 6. Cache results ──────────────────────────────────────────────
        _LAST_RESULTS = ranked_out
        _LAST_RUN_STATE.update({
            "is_running": False,
            "last_run_id": run_id,
            "last_run_elapsed_ms": elapsed_ms,
            "last_run_candidate_count": total_input,
            "last_run_honeypots": honeypots_removed,
            "stage_timings": stage_timings,
        })

        logger.info(
            "Pipeline run %s complete: %d candidates → %d ranked, %d honeypots, %.0fms.",
            run_id, total_input, len(ranked_out), honeypots_removed, elapsed_ms,
        )

        return RankResponse(
            status="success",
            total_candidates_input=total_input,
            honeypots_removed=honeypots_removed,
            ranked=ranked_out,
            pipeline_elapsed_ms=round(elapsed_ms, 1),
            stage_timings={k: round(v, 1) for k, v in stage_timings.items()},
            errors=errors,
        )

    except HTTPException:
        _LAST_RUN_STATE["is_running"] = False
        raise
    except Exception as exc:
        _LAST_RUN_STATE["is_running"] = False
        logger.exception("Pipeline run %s failed.", run_id)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc


@router.get("/results", response_model=ResultsResponse)
async def get_results() -> ResultsResponse:
    """Return cached results from the most recent /rank call."""
    if not _LAST_RESULTS:
        return ResultsResponse(
            status="no_results",
            message="No pipeline run has been completed yet. POST /rank to run the pipeline.",
        )
    return ResultsResponse(
        status="success",
        run_id=_LAST_RUN_STATE.get("last_run_id"),
        ranked=_LAST_RESULTS,
        pipeline_elapsed_ms=_LAST_RUN_STATE.get("last_run_elapsed_ms"),
        message=f"{len(_LAST_RESULTS)} candidates ranked.",
    )


@router.post("/export/csv")
async def export_csv(body: CSVExportRequest) -> StreamingResponse:
    """
    Export the last pipeline run as submission.csv.

    Returns a CSV file with columns: candidate_id, rank, score, reasoning.
    Optionally runs submission validation checks before exporting.
    """
    if not _LAST_RESULTS:
        raise HTTPException(
            status_code=404,
            detail="No results available. Run POST /rank first.",
        )

    if body.validate_before_export:
        # Check monotonicity
        scores = [r.score for r in sorted(_LAST_RESULTS, key=lambda r: r.rank)]
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1]:
                raise HTTPException(
                    status_code=422,
                    detail=f"Non-monotonic scores at rank {i+1} ({scores[i]:.6f}) → rank {i+2} ({scores[i+1]:.6f}). Re-run pipeline.",
                )

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    for rc in sorted(_LAST_RESULTS, key=lambda r: r.rank):
        writer.writerow({
            "candidate_id": rc.candidate_id,
            "rank": rc.rank,
            "score": f"{rc.score:.6f}",
            "reasoning": rc.reasoning,
        })

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=submission.csv"},
    )
