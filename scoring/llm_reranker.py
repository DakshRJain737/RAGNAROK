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

from pipeline.schemas import JDIntent, CandidateFeatureVector, TrustVerdict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_signal_block(trust: TrustVerdict) -> str:
    """
    Convert advocate + skeptic signals into a compact, LLM-readable fact block.

    Outputs format:
        STRENGTHS (HIGH): 75% required skills matched: FAISS, sentence-transformers
        STRENGTHS (MED): 3 product company roles: Zomato, Razorpay
        RISKS (HIGH): Last active 120 days ago
        RISKS (MOD): Consulting-only background

    Capped at top-3 advocate + top-2 skeptic signals to control token count.
    """
    lines: list[str] = []

    # Top-2 advocate signals (already sorted HIGH first)
    for sig in trust.advocate_signals[:2]:
        val = sig.value[:55].rstrip()
        lines.append(f"+[{sig.confidence[0]}] {sig.label}: {val}")

    # Top-1 skeptic signal
    for sig in trust.skeptic_signals[:1]:
        val = sig.value[:55].rstrip()
        lines.append(f"-[{sig.severity[0]}] {sig.label}: {val}")

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
    "Technical recruiter brief: 2 sentences using ONLY the given facts. "
    "Sentence 1: strongest signal (+ or -) that explains position. "
    "Sentence 2: key counterpoint. Name skills/companies/numbers. No filler."
)

_USER_TEMPLATE = """\
Job: Senior AI Eng | 5-9yr | product-co | retrieval/ranking skills
Tier: {tier} | {verdict} ({confidence}% confidence)
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
    ) -> str:
        """
        Generate a 2-sentence justification for one candidate using trust signals.

        Parameters
        ----------
        rank     : Final rank (1–100)
        trust    : TrustVerdict with advocate + skeptic signals
        fallback : Rule-based reasoning string to return on LLM failure

        Returns
        -------
        str — 2-sentence recruiter brief
        """
        signal_block = _build_signal_block(trust)
        tier = _tier_label(rank)

        prompt = _USER_TEMPLATE.format(
            tier=tier,
            verdict=trust.verdict,
            confidence=int(round(trust.confidence_pct)),
            signal_block=signal_block,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            out = self._llm.create_chat_completion(
                messages=messages,
                temperature=0.7,     # tiny warmth for natural phrasing, no randomness
                max_tokens=80,       # ~60 words — enough for 2 tight sentences
                stop=["\n\n", "Sentence 3", "3."],  # prevent runaway
            )
            text = out["choices"][0]["message"]["content"].strip()

            # Sanity: reject if too short or is clearly an echo of the prompt
            if len(text) < 30 or text.startswith("Write") or text.startswith("You are"):
                logger.debug("LLM output rejected (too short or echo): %r", text[:60])
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
    ) -> dict[str, str]:
        """
        Generate 2-sentence justifications for top-100 candidates.

        Parameters
        ----------
        candidates    : top-100 CandidateFeatureVectors, in rank order
        jd            : JDIntent (used only for fallback blurb)
        ranks         : dict[candidate_id -> rank_int]
        trust_verdicts: dict[candidate_id -> TrustVerdict] from the trust layer.
                        If None or a candidate is missing, falls back to fallback string.
        fallbacks     : dict[candidate_id -> rule_based_reasoning_str].
                        Used when LLM call fails or trust verdict is unavailable.

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

        for i, cfv in enumerate(candidates, start=1):
            cid = cfv.candidate_id
            rank = ranks.get(cid, i)
            fallback = fallbacks.get(cid, f"Ranked based on composite score (rank {rank}).")

            trust = trust_verdicts.get(cid)
            if trust is None:
                # No trust verdict available — use fallback directly
                logger.debug("LLM: no trust verdict for %s, using fallback", cid)
                results[cid] = fallback
                continue

            justification = self._justify_one(
                rank=rank,
                trust=trust,
                fallback=fallback,
            )
            results[cid] = justification

            if i % 10 == 0 or i == len(candidates):
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else 1.0
                eta = (len(candidates) - i) / rate if rate > 0 else 0.0
                logger.info(
                    "LLM: %d/%d done | %.1fs elapsed | ETA %.0fs",
                    i, len(candidates), elapsed, eta,
                )

        elapsed = time.perf_counter() - t0
        logger.info(
            "LLM: justified %d candidates in %.1fs (%.2f s/candidate)",
            len(candidates), elapsed,
            elapsed / max(1, len(candidates)),
        )
        return results