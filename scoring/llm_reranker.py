"""
scoring/llm_reranker.py — Lightweight local LLM reranker using Qwen2.5-1.5B-Instruct (Q4 GGUF).

Scores the top-N candidates from RRF on a structured prompt.
No network calls during ranking. Model loads from local cache.

Usage:
    reranker = LLMReranker()
    pool = reranker.score_pool(rrf_pool, jd_intent, candidate_store, top_n=300)
    # Each result in pool now has .llm_score (float 0.0–1.0)

Tune via config.py:
    LLM_MODEL_PATH, LLM_TOP_N, LLM_BLEND_FACTOR, LLM_N_THREADS, LLM_N_CTX
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container — attach llm_score to whatever object your RRF returns
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMScoredResult:
    """
    Wraps an existing RRF result and adds an llm_score field.
    If your RRFResult already has extra fields, just add llm_score directly
    to that dataclass instead of using this wrapper.
    """
    candidate_id: str
    rrf_score: float
    llm_score: float = 0.5       # neutral default if LLM skipped
    paths_present: list = None

    def __post_init__(self):
        if self.paths_present is None:
            self.paths_present = []


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders — edit these to match your CandidateFeatureVector fields
# ─────────────────────────────────────────────────────────────────────────────

def build_jd_summary(jd) -> str:
    """
    Condense JDIntent → ~80 token string for the prompt.
    Edit field names to match your pipeline/schemas.py JDIntent class.
    """
    # ── Pull required skills ──────────────────────────────────────────────────
    try:
        req_skills = ", ".join(list(jd.required_skills)[:5])
    except Exception:
        req_skills = "embeddings, retrieval, ranking, Python"

    # ── Pull seniority / location hints ──────────────────────────────────────
    try:
        yoe = f"{jd.yoe_min:.0f}-{jd.yoe_max:.0f}y"
    except Exception:
        yoe = "5-9y"

    try:
        location = jd.preferred_locations[0] if jd.preferred_locations else "Pune/Noida"
    except Exception:
        location = "Pune/Noida"

    return (
        f"Role: Senior AI Engineer · Experience: {yoe} · Location: {location} · "
        f"Must have: {req_skills} · "
        f"No consulting-only backgrounds (TCS/Wipro/Infosys etc.) · "
        f"Product-company experience required"
    )


def build_candidate_summary(cfv) -> str:
    """
    Condense CandidateFeatureVector → ~100 token string for the prompt.
    Edit field names to match your pipeline/schemas.py CandidateFeatureVector class.
    """
    # ── Skills ────────────────────────────────────────────────────────────────
    try:
        # Adjust: cfv.skills may be a list of dicts or list of strings
        if cfv.skills and isinstance(cfv.skills[0], dict):
            skills = ", ".join(s.get("name", "") for s in cfv.skills[:6])
        else:
            skills = ", ".join(str(s) for s in cfv.skills[:6])
    except Exception:
        skills = "unknown"

    # ── Experience & career ───────────────────────────────────────────────────
    try:
        yoe = f"{cfv.years_of_experience:.0f}y"
    except Exception:
        yoe = "?"

    try:
        title = cfv.current_title or "Unknown"
    except Exception:
        title = "Unknown"

    try:
        company = cfv.current_company or "Unknown"
    except Exception:
        company = "Unknown"

    # ── Flags ─────────────────────────────────────────────────────────────────
    try:
        product_co = "yes" if cfv.has_product_co_experience else "no"
    except Exception:
        product_co = "unknown"

    try:
        consulting_only = "yes" if cfv.is_consulting_only else "no"
    except Exception:
        consulting_only = "unknown"

    try:
        active_days = cfv.days_since_active if hasattr(cfv, "days_since_active") else "?"
        activity = f"{active_days}d ago"
    except Exception:
        activity = "unknown"

    return (
        f"{yoe} exp · {title} at {company} · "
        f"Skills: {skills} · "
        f"Product-co: {product_co} · Consulting-only: {consulting_only} · "
        f"Last active: {activity}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template — single structured scoring prompt
# ─────────────────────────────────────────────────────────────────────────────

SCORING_PROMPT_TEMPLATE = """\
You are a senior technical recruiter scoring a candidate for a job.
Score from 0 to 10. Reply with ONLY a single integer or decimal number. Nothing else.

JOB: {jd_summary}

CANDIDATE: {candidate_summary}

SCORE (0-10):"""


# ─────────────────────────────────────────────────────────────────────────────
# Main reranker class
# ─────────────────────────────────────────────────────────────────────────────

class LLMReranker:
    """
    Local LLM scorer using llama-cpp-python with a Q4 GGUF model.

    Instantiate once per pipeline run (model load is expensive).
    Call score_pool() to batch-score a list of RRF results.

    Args:
        model_path: Path to .gguf file. Defaults to config.LLM_MODEL_PATH.
        n_threads:  CPU threads for inference. Defaults to config.LLM_N_THREADS.
        n_ctx:      Context window size. Keep at 512-1024 for speed.
        verbose:    Show llama.cpp logs. Set True to debug.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_threads: Optional[int] = None,
        n_ctx: Optional[int] = None,
        verbose: bool = False,
    ):
        import config

        self._model_path = model_path or config.LLM_MODEL_PATH
        self._n_threads  = n_threads  or config.LLM_N_THREADS
        self._n_ctx      = n_ctx      or config.LLM_N_CTX
        self._verbose    = verbose
        self._llm        = None        # lazy load

    # ── Lazy model loader ─────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Run: pip install llama-cpp-python"
            )

        logger.info("Loading LLM from %s …", self._model_path)
        t0 = time.perf_counter()
        self._llm = Llama(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            verbose=self._verbose,
        )
        logger.info("LLM loaded in %.1fs", time.perf_counter() - t0)

    # ── Single candidate scorer ───────────────────────────────────────────────

    def score_one(self, jd_summary: str, candidate_summary: str) -> float:
        """
        Score a single candidate. Returns float in [0.0, 1.0].
        Returns 0.5 (neutral) on any parse or inference failure.
        """
        self._load()

        prompt = SCORING_PROMPT_TEMPLATE.format(
            jd_summary=jd_summary,
            candidate_summary=candidate_summary,
        )

        try:
            out = self._llm(
                prompt,
                max_tokens=4,       # "10" or "7.5" — never more
                temperature=0.0,    # deterministic
                stop=["\n", " ", "."],
            )
            raw = out["choices"][0]["text"].strip()
            score = float(raw)
            # Clamp to [0, 10] then normalise to [0, 1]
            score = max(0.0, min(10.0, score))
            return score / 10.0
        except Exception as e:
            logger.debug("LLM score parse failed (%s) — using neutral 0.5", e)
            return 0.5

    # ── Batch scorer (main entry point) ──────────────────────────────────────

    def score_pool(
        self,
        rrf_pool: list,
        jd_intent,
        candidate_store: dict,
        top_n: Optional[int] = None,
    ) -> list:
        """
        Score up to top_n candidates in rrf_pool. Attaches .llm_score to each.

        Args:
            rrf_pool:        List of RRFResult objects (must have .candidate_id).
            jd_intent:       Parsed JDIntent object.
            candidate_store: Dict mapping candidate_id → CandidateFeatureVector.
            top_n:           How many to score. Defaults to config.LLM_TOP_N.

        Returns:
            The same rrf_pool list, with .llm_score set on each item.
            Items beyond top_n get llm_score = 0.5 (neutral).
        """
        import config
        top_n = top_n or config.LLM_TOP_N

        jd_summary = build_jd_summary(jd_intent)
        to_score   = rrf_pool[:top_n]
        skipped    = rrf_pool[top_n:]

        logger.info("LLM reranker: scoring %d candidates …", len(to_score))
        t0 = time.perf_counter()

        scored = 0
        failed = 0

        for result in to_score:
            cfv = candidate_store.get(result.candidate_id)
            if cfv is None:
                result.llm_score = 0.5
                failed += 1
                continue

            candidate_summary = build_candidate_summary(cfv)
            result.llm_score  = self.score_one(jd_summary, candidate_summary)
            scored += 1

        # Neutral score for candidates outside top_n
        for result in skipped:
            result.llm_score = 0.5

        elapsed = time.perf_counter() - t0
        logger.info(
            "LLM reranker done: %d scored, %d skipped/failed in %.1fs (%.0fms/candidate)",
            scored, failed, elapsed,
            (elapsed / scored * 1000) if scored else 0,
        )

        return rrf_pool

    # ── Convenience: download model if missing ────────────────────────────────

    @staticmethod
    def download_model(
        repo_id: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        local_dir: str = "models/",
    ) -> str:
        """
        Download GGUF model from HuggingFace Hub into local_dir.
        Call this from precompute.py, not from rank.py (requires network).

        Returns the local path to the downloaded file.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError("pip install huggingface-hub")

        logger.info("Downloading %s/%s → %s", repo_id, filename, local_dir)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
        )
        logger.info("Model saved to %s", path)
        return path