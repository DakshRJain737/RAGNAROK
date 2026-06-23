"""
scoring/llm_reranker.py — LLM justification layer for top-100 candidates.

ROLE
----
Last step in the pipeline — called AFTER composite scoring, cross-encoder,
and the trust (advocate/skeptic) layer have all completed.

Does NOT affect any score. Generates a 2-sentence recruiter brief that
justifies why a candidate holds their specific rank, grounded in the
advocate/skeptic signals that were already computed by the trust layer.

DESIGN
------
- Advocate signals  (confirmed strengths, HIGH→MEDIUM→LOW)
- Skeptic signals   (confirmed risks, HIGH→MODERATE→LOW)
...are serialised into a compact fact-block and injected into the LLM prompt.
The LLM's only job is to weave those facts into fluent, rank-specific language.
It cannot invent facts — every claim it makes must trace to a signal value.

Prompt budget: ~200 input tokens, 80 output tokens → ~1.5–2s on Qwen 1.5B Q4.
100 candidates × 2s ≈ 200s — within the 5-min pipeline wall clock.

Usage (called from pipeline/runner.py):
    reranker = LLMReranker(model_path=config.LLM_MODEL_PATH)
    justifications = reranker.justify_candidates(
        top100_cfvs, jd, ranks, trust_verdicts, fallbacks
    )
    # justifications: dict[candidate_id -> str]
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from pipeline.schemas import JDIntent, CandidateFeatureVector, TrustVerdict, ComponentScores

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Maximum character length for individual signal values in the LLM prompt.
# Longer values give the LLM more concrete facts; 120 chars fits comfortably
# within the 200-token input budget when combined with 3+2 signals.
_SIGNAL_VAL_MAX: int = 120

# Number of signals to include in the prompt block.
_ADVOCATE_SIGNALS_IN_PROMPT: int = 3   # top-3 advocate (HIGH first)
_SKEPTIC_SIGNALS_IN_PROMPT: int = 2    # top-2 skeptic (HIGH first)


def _build_signal_block(trust: TrustVerdict) -> str:
    """
    Convert advocate + skeptic signals into a rich, grouped, LLM-readable fact
    block with full-length values and the top falsifiability condition.

    Output format (example):
        STRENGTHS
          [HIGH] Required skill coverage: 82% of required skills matched: FAISS, sentence-transformers, BGE
          [HIGH] Product-company experience: 3 product company roles: Swiggy, Zepto, Razorpay
          [MED]  Career trajectory velocity: 78% velocity percentile (~1.2 promotions/yr)
        RISKS
          [HIGH] Platform inactivity: Last active ~4 months ago (120 days) — above 90-day threshold
          [MOD]  Partial required skill coverage: gaps: LoRA, XGBoost LTR
        KEY CONDITION: verify candidate is actively looking (120 days inactive)

    Caps at top-3 advocate + top-2 skeptic signals to stay within the 200-token
    input budget.  Signal values are allowed up to _SIGNAL_VAL_MAX chars so the
    LLM receives specific skill names, companies, and numbers rather than
    truncated fragments.
    """
    lines: list[str] = []

    # ── STRENGTHS block ───────────────────────────────────────────────────────
    adv_signals = trust.advocate_signals[:_ADVOCATE_SIGNALS_IN_PROMPT]
    if adv_signals:
        lines.append("STRENGTHS")
        for sig in adv_signals:
            # Keep the full value up to _SIGNAL_VAL_MAX chars — smart truncation
            # at the last comma to avoid mid-skill-name cuts.
            val = sig.value[:_SIGNAL_VAL_MAX].rstrip()
            if len(sig.value) > _SIGNAL_VAL_MAX:
                last_comma = val.rfind(", ")
                val = val[:last_comma] + "…" if last_comma > _SIGNAL_VAL_MAX // 2 else val + "…"
            # Pad tier label so columns align — easier for the LLM to parse.
            tier_tag = f"[{sig.confidence:<3}]".replace("MED", "MED").replace("LOW", "LOW")
            lines.append(f"  {tier_tag} {sig.label}: {val}")

    # ── RISKS block ───────────────────────────────────────────────────────────
    skep_signals = trust.skeptic_signals[:_SKEPTIC_SIGNALS_IN_PROMPT]
    if skep_signals:
        lines.append("RISKS")
        for sig in skep_signals:
            val = sig.value[:_SIGNAL_VAL_MAX].rstrip()
            if len(sig.value) > _SIGNAL_VAL_MAX:
                last_comma = val.rfind(", ")
                val = val[:last_comma] + "…" if last_comma > _SIGNAL_VAL_MAX // 2 else val + "…"
            # Map MODERATE → MOD for alignment.
            sev_tag = sig.severity.replace("MODERATE", "MOD ")
            lines.append(f"  [{sev_tag:<3}] {sig.label}: {val}")

    # ── KEY CONDITION (top falsifiability condition) ───────────────────────────
    # The falsifiability contract is the most interview-actionable fact in the
    # trust verdict.  Including it gives the LLM a concrete closing hook for
    # Sentence 2 instead of just repeating the verdict word.
    if trust.falsifiability:
        # Strip the long "This ranking holds UNLESS " prefix to a compact form.
        cond = trust.falsifiability[0]
        for prefix in (
            "This ranking holds UNLESS ",
            "This ranking becomes MORE ROBUST if ",
            "This ranking is critically weakened by ",
            "This ranking is notably affected if ",
            "This ranking is marginally affected by ",
            "This ranking\'s confidence is reduced by ",
        ):
            if cond.startswith(prefix):
                cond = cond[len(prefix):].strip()
                break
        # Cap the condition to 90 chars.
        if len(cond) > 90:
            cond = cond[:87].rstrip() + "…"
        lines.append(f"KEY CONDITION: {cond}")

    return "\n".join(lines) if lines else "(no signals)"


def _tier_label(rank: int) -> str:
    """Map rank to a tier adjective the LLM can use for tone."""
    if rank <= 5:
        return "top-5 — exceptional fit"
    if rank <= 15:
        return "top-15 — strong fit"
    if rank <= 40:
        return "mid-tier — solid but with gaps"
    if rank <= 70:
        return "lower-mid — notable weaknesses"
    return "bottom-tier — poor fit"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt constants
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a technical recruiter writing a 2-sentence candidate brief. "
    "Use ONLY the facts listed below — no invented claims. "
    "Sentence 1: state the single most important fact (strength or risk) "
    "specific to THIS candidate — name the exact skill, company, or number. "
    "Do NOT open with 'The strongest signal that explains the position'. "
    "Start directly with the fact (e.g. '82% skill match across FAISS...' or "
    "'Inactive 120 days — availability unclear...'). "
    "Sentence 2: give the key counterpoint or the KEY CONDITION that would change this assessment. "
    "Be concise. No filler phrases."
)

_USER_TEMPLATE = """\
Job: Senior AI Eng | 5-9yr | product-co | retrieval/ranking skills
Rank #{rank} of 100 | {tier} | {verdict} ({confidence}% confidence)
Candidate: {yoe}yr exp | composite score {composite:.3f}
Facts:
{signal_block}
Brief:"""


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class LLMReranker:
    """Local GGUF LLM used exclusively for post-ranking justification."""

    @staticmethod
    def download_model(repo_id: str, filename: str, local_dir: str) -> None:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading %s from %s to %s …", filename, repo_id, local_dir)
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

    def __init__(
        self,
        model_path: str,
        n_threads: int = 8,
        n_ctx: int = 512,
        verbose: bool = False,
    ) -> None:
        self._model_path = model_path
        self._n_threads = n_threads
        self._n_ctx = n_ctx
        self._verbose = verbose
        self._llm = None

    def _load(self) -> None:
        if self._llm is not None:
            return
        # pyrefly: ignore [missing-import]
        from llama_cpp import Llama
        logger.info("Loading LLM for justification …")
        t0 = time.perf_counter()
        self._llm = Llama(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            verbose=self._verbose,
        )
        logger.info("LLM loaded in %.2fs", time.perf_counter() - t0)

    # ── Core inference ────────────────────────────────────────────────────────

    def _justify_one(
        self,
        rank: int,
        trust: TrustVerdict,
        fallback: str,
        candidate: Optional[CandidateFeatureVector] = None,
        composite_score: float = 0.0,
    ) -> str:
        """
        Generate a 2-sentence justification for one candidate using trust signals.

        Parameters
        ----------
        rank            : Final rank (1–100)
        trust           : TrustVerdict with advocate + skeptic signals
        fallback        : Rule-based reasoning string to return on LLM failure
        candidate       : CandidateFeatureVector for YOE in the prompt header
        composite_score : Normalised composite score [0.10–1.00] for the header

        Returns
        -------
        str — 2-sentence recruiter brief
        """
        signal_block = _build_signal_block(trust)
        tier = _tier_label(rank)
        yoe = f"{candidate.years_of_experience:.1f}" if candidate else "?"

        prompt = _USER_TEMPLATE.format(
            rank=rank,
            tier=tier,
            verdict=trust.verdict,
            confidence=int(round(trust.confidence_pct)),
            yoe=yoe,
            composite=composite_score,
            signal_block=signal_block,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            out = self._llm.create_chat_completion(
                messages=messages,
                temperature=0.4,     # lower temperature for more factual, less random phrasing
                max_tokens=100,       # slightly more room for fact-rich sentences
                stop=["\n\n", "Sentence 3", "3.", "\nJob:"],  # prevent runaway / prompt echo
            )
            text = out["choices"][0]["message"]["content"].strip()

            # Sanity: reject if too short or echoes known generic/prompt phrases.
            # These patterns indicate the model is repeating instructions rather
            # than grounding in the candidate-specific facts.
            _BANNED_OPENERS = (
                "Write",
                "You are",
                "Job:",
                "The strongest signal that explains the position",
                "The single most important fact specific to this candidate is",
                "The key strength is the candidate",  # another observed echo
            )
            if len(text) < 30 or any(text.startswith(p) for p in _BANNED_OPENERS):
                logger.debug("LLM output rejected (too short or generic echo): %r", text[:80])
                return fallback

            # Hard cap at 320 chars for CSV column safety
            return text[:320]

        except Exception as exc:
            logger.debug("LLM justify_one failed: %s", exc)
            return fallback

    # ── Batch justification (public API) ──────────────────────────────────────

    def justify_candidates(
        self,
        candidates: list[CandidateFeatureVector],
        jd: JDIntent,
        ranks: dict[str, int],
        trust_verdicts: Optional[dict[str, TrustVerdict]] = None,
        fallbacks: Optional[dict[str, str]] = None,
        top_n: Optional[int] = None,
        composite_scores: Optional[dict[str, float]] = None,
    ) -> dict[str, str]:
        """
        Generate 2-sentence justifications for top-N candidates.

        Parameters
        ----------
        candidates       : top-100 CandidateFeatureVectors, in rank order
        jd               : JDIntent (used only for fallback blurb)
        ranks            : dict[candidate_id -> rank_int]
        trust_verdicts   : dict[candidate_id -> TrustVerdict] from the trust layer.
                           If None or a candidate is missing, falls back to fallback string.
        fallbacks        : dict[candidate_id -> rule_based_reasoning_str].
                           Used when LLM call fails or trust verdict is unavailable.
        top_n            : If set, only run LLM inference for candidates whose rank is
                           <= top_n. Candidates ranked beyond top_n automatically receive
                           the rule-based fallback string (no LLM call, zero latency).
                           None means run for all candidates (legacy behaviour).
        composite_scores : dict[candidate_id -> normalised_composite_score] from
                           the pipeline's score normalisation step. Used to populate
                           the candidate header in the prompt. Pass None to omit.

        Returns
        -------
        dict[candidate_id -> justification_str]
        """
        self._load()

        trust_verdicts = trust_verdicts or {}
        fallbacks = fallbacks or {}
        results: dict[str, str] = {}
        t0 = time.perf_counter()

        logger.info(
            "LLM: generating signal-grounded justifications for %d candidates …",
            len(candidates),
        )

        llm_count = 0
        skipped_count = 0

        for i, cfv in enumerate(candidates, start=1):
            cid = cfv.candidate_id
            rank = ranks.get(cid, i)
            fallback = fallbacks.get(cid, f"Ranked based on composite score (rank {rank}).")

            # Skip LLM for candidates ranked beyond top_n — use rule-based fallback.
            if top_n is not None and rank > top_n:
                results[cid] = fallback
                skipped_count += 1
                continue

            trust = trust_verdicts.get(cid)
            if trust is None:
                # No trust verdict available — use fallback directly
                logger.debug("LLM: no trust verdict for %s, using fallback", cid)
                results[cid] = fallback
                skipped_count += 1
                continue

            # Retrieve normalised composite score from the trust verdict's
            # confidence_pct as a proxy when the score dict isn't passed.
            # The composite score for the candidate header comes from the
            # composite_scores dict if provided, else falls back to 0.0.
            composite = composite_scores.get(cid, 0.0) if composite_scores else 0.0

            justification = self._justify_one(
                rank=rank,
                trust=trust,
                fallback=fallback,
                candidate=cfv,
                composite_score=composite,
            )
            results[cid] = justification
            llm_count += 1

            if llm_count % 10 == 0 or i == len(candidates):
                elapsed = time.perf_counter() - t0
                rate = llm_count / elapsed if elapsed > 0 and llm_count > 0 else 1.0
                remaining_llm = max(0, (top_n or len(candidates)) - llm_count)
                eta = remaining_llm / rate if rate > 0 else 0.0
                logger.info(
                    "LLM: %d/%d done | %d skipped (rule-based) | %.1fs elapsed | ETA %.0fs",
                    llm_count, top_n or len(candidates), skipped_count, elapsed, eta,
                )

        elapsed = time.perf_counter() - t0
        logger.info(
            "LLM: justified %d candidates via LLM, %d via rule-based, in %.1fs (%.2f s/LLM-candidate)",
            llm_count, skipped_count, elapsed,
            elapsed / max(1, llm_count),
        )
        return results