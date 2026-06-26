from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import time
from itertools import islice
from typing import Optional

from pipeline.schemas import JDIntent, CandidateFeatureVector, TrustVerdict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_VAL_MAX: int = 120
_ADVOCATE_SIGNALS_IN_PROMPT: int = 3
_SKEPTIC_SIGNALS_IN_PROMPT: int = 2

_FALSIFIABILITY_PREFIXES: tuple[str, ...] = (
    "This ranking holds UNLESS ",
    "This ranking becomes MORE ROBUST if ",
    "This ranking is critically weakened by ",
    "This ranking is notably affected if ",
    "This ranking is marginally affected by ",
    "This ranking's confidence is reduced by ",
)

_BANNED_OPENERS: tuple[str, ...] = (
    "Write",
    "You are",
    "Job:",
    "The strongest signal that explains the position",
    "The single most important fact specific to this candidate is",
    "The key strength is the candidate",
)

_SYSTEM_PROMPT = (
    "Recruiter brief: 2 sentences, facts only, \u226460 words total. "
    "S1: strongest single signal for this candidate — name exact skill, company, or number. "
    "S2: key counterpoint or condition that would change this assessment. "
    "Do NOT start with banned phrases like 'The strongest signal'. "
    "Start directly with the fact (e.g. '82% skill match\u2026' or 'Inactive 120 days\u2026')."
)

_USER_TEMPLATE = """\
Job: Senior AI Eng | 5-9yr | product-co | retrieval/ranking skills
Rank #{rank} of 100 | {tier} | {verdict} ({confidence}% confidence)
Candidate: {yoe}yr exp | composite score {composite:.3f}
Facts:
{signal_block}
Brief:"""


def _smart_truncate(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    sliced = value[:max_len].rstrip()
    last_comma = sliced.rfind(", ")
    if last_comma >= max_len // 2:
        return sliced[:last_comma] + "…"
    return sliced + "…"


def _build_signal_block(trust: TrustVerdict) -> str:
    lines: list[str] = []

    adv_signals = list(islice(trust.advocate_signals, _ADVOCATE_SIGNALS_IN_PROMPT))
    if adv_signals:
        lines.append("STRENGTHS")
        for sig in adv_signals:
            val = _smart_truncate(sig.value, _SIGNAL_VAL_MAX)
            lines.append(f"  [{sig.confidence:<3}] {sig.label}: {val}")

    skep_signals = list(islice(trust.skeptic_signals, _SKEPTIC_SIGNALS_IN_PROMPT))
    if skep_signals:
        lines.append("RISKS")
        for sig in skep_signals:
            val = _smart_truncate(sig.value, _SIGNAL_VAL_MAX)
            sev_tag = sig.severity.replace("MODERATE", "MOD ")
            lines.append(f"  [{sev_tag:<3}] {sig.label}: {val}")

    if trust.falsifiability:
        cond = trust.falsifiability[0]
        for prefix in _FALSIFIABILITY_PREFIXES:
            if cond.startswith(prefix):
                cond = cond[len(prefix):].strip()
                break
        if len(cond) > 90:
            cond = cond[:87].rstrip() + "…"
        lines.append(f"KEY CONDITION: {cond}")

    return "\n".join(lines) if lines else "(no signals)"


def _tier_label(rank: int) -> str:
    if rank <= 5:
        return "top-5 — exceptional fit"
    if rank <= 15:
        return "top-15 — strong fit"
    if rank <= 40:
        return "mid-tier — solid but with gaps"
    if rank <= 70:
        return "lower-mid — notable weaknesses"
    return "bottom-tier — poor fit"


def _build_messages(
    rank: int,
    trust: TrustVerdict,
    candidate: Optional[CandidateFeatureVector],
    composite_score: float,
) -> list[dict]:
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
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

_worker_llm = None
_worker_logger = None


def _pool_initializer(
    model_path: str,
    n_ctx: int,
    n_threads: int,
    verbose: bool,
) -> None:
    global _worker_llm, _worker_logger
    _worker_logger = logging.getLogger(__name__)

    from llama_cpp import Llama
    pid = os.getpid()
    _worker_logger.info("Worker PID %d: loading model …", pid)
    t0 = time.perf_counter()
    _worker_llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=verbose,
        use_mlock=False,
    )
    _worker_logger.info("Worker PID %d: model ready in %.2fs", pid, time.perf_counter() - t0)


def _pool_infer(task: tuple) -> tuple[str, str]:
    cid, messages, fallback = task

    if _worker_llm is None:
        # Should never happen if initializer ran, but be defensive.
        return cid, fallback

    try:
        out = _worker_llm.create_chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=80,
            stop=["\n\n", "Sentence 3", "3.", "\nJob:"],
        )
        text = out["choices"][0]["message"]["content"].strip()

        if len(text) < 30 or any(text.startswith(p) for p in _BANNED_OPENERS):
            if _worker_logger:
                _worker_logger.debug(
                    "Worker: output rejected (short/banned): %r", text[:80]
                )
            return cid, fallback

        return cid, text[:320]

    except Exception as exc:
        if _worker_logger:
            _worker_logger.debug("Worker: inference failed for %s: %s", cid, exc)
        return cid, fallback


# Main class
class LLMReranker:

    @staticmethod
    def download_model(repo_id: str, filename: str, local_dir: str) -> None:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading %s from %s to %s …", filename, repo_id, local_dir)
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

    def __init__(
        self,
        model_path: str,
        n_threads: int = 4,            # kept for API compat; see n_threads_per_worker
        n_ctx: int = 512,
        verbose: bool = False,
        max_workers: int = 4,          # number of parallel worker processes
        n_threads_per_worker: int = 2, # llama_cpp threads inside each worker
    ) -> None:
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._verbose = verbose
        self._num_workers = max_workers
        self._n_threads_per_worker = n_threads_per_worker or max(1, n_threads // max_workers)

        self._pool: Optional[mp.pool.Pool] = None
        self._pool_ctx = mp.get_context("spawn")  # spawn is safe on all platforms

    # ── Pool lifecycle ────────────────────────────────────────────────────

    def preload(self) -> None:
        if self._pool is not None:
            return

        logger.info(
            "LLM: starting %d worker processes (n_threads_per_worker=%d) …",
            self._num_workers,
            self._n_threads_per_worker,
        )
        t0 = time.perf_counter()

        self._pool = self._pool_ctx.Pool(
            processes=self._num_workers,
            initializer=_pool_initializer,
            initargs=(
                self._model_path,
                self._n_ctx,
                self._n_threads_per_worker,
                self._verbose,
            ),
        )

        dummy = [("__warmup__", [], "__warmup__")] * self._num_workers
        self._pool.map(_pool_infer, dummy)

        logger.info(
            "LLM: all %d workers ready in %.2fs",
            self._num_workers,
            time.perf_counter() - t0,
        )

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
            logger.info("LLM: worker pool shut down.")

    def __enter__(self) -> "LLMReranker":
        self.preload()
        return self

    def __exit__(self, *_) -> None:
        self.shutdown()

    def _ensure_pool(self) -> None:
        if self._pool is None:
            self.preload()

    # ── Batch justification (public API) ─────────────────────────────────

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

        self._ensure_pool()

        trust_verdicts = trust_verdicts or {}
        fallbacks = fallbacks or {}
        _composite: dict[str, float] = composite_scores or {}

        if composite_scores is not None and not composite_scores:
            logger.warning(
                "LLM: composite_scores passed as empty dict; "
                "all prompt headers will show composite=0.000"
            )

        results: dict[str, str] = {}
        t0 = time.perf_counter()

        tasks: list[tuple[str, list[dict], str]] = []

        for i, cfv in enumerate(candidates, start=1):
            cid = cfv.candidate_id
            rank = ranks.get(cid, i)

            raw_fallback = fallbacks.get(cid)
            if raw_fallback is None:
                logger.warning(
                    "LLM: no fallback for candidate %s (rank %d) — using placeholder.",
                    cid,
                    rank,
                )
                raw_fallback = f"Ranked based on composite score (rank {rank})."

            if top_n is not None and rank > top_n:
                results[cid] = raw_fallback
                continue

            trust = trust_verdicts.get(cid)
            if trust is None:
                logger.debug("LLM: no trust verdict for %s, using fallback", cid)
                results[cid] = raw_fallback
                continue

            composite = _composite.get(cid, 0.0)
            messages = _build_messages(rank, trust, cfv, composite)
            tasks.append((cid, messages, raw_fallback))

        skipped_count = len(results)
        logger.info(
            "LLM: dispatching %d candidates to %d workers (%d skipped) …",
            len(tasks),
            self._num_workers,
            skipped_count,
        )

        if not tasks:
            return results

        chunksize = max(1, math.ceil(len(tasks) / self._num_workers))

        completed = 0
        for cid, justification in self._pool.imap_unordered(
            _pool_infer, tasks, chunksize=chunksize
        ):
            results[cid] = justification
            completed += 1

            if completed % 10 == 0:
                elapsed = time.perf_counter() - t0
                rate = completed / elapsed if elapsed > 0 else 1.0
                remaining = max(0, len(tasks) - completed)
                eta = remaining / rate if rate > 0 else 0.0
                logger.info(
                    "LLM: %d/%d done | %d skipped | %.1fs elapsed | ETA %.0fs",
                    completed,
                    len(tasks),
                    skipped_count,
                    elapsed,
                    eta,
                )

        elapsed = time.perf_counter() - t0
        logger.info(
            "LLM: justified %d via workers, %d via fallback, in %.1fs "
            "(%.2f s/candidate wall-clock, %.1f× speedup over serial)",
            completed,
            skipped_count,
            elapsed,
            elapsed / max(1, completed),
            (completed * (elapsed / max(1, completed)) * self._num_workers) / max(elapsed, 0.001),
        )
        return results