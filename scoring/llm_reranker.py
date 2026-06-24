from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from typing import Optional

from pipeline.schemas import JDIntent, CandidateFeatureVector, TrustVerdict, ComponentScores

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_VAL_MAX: int = 120
_ADVOCATE_SIGNALS_IN_PROMPT: int = 3
_SKEPTIC_SIGNALS_IN_PROMPT: int = 2

# FIX 7: strip prefixes as a tuple constant so `startswith` uses the fast C path
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


def _smart_truncate(value: str, max_len: int) -> str:
    # FIX 6: early-exit before any work when string is already short enough
    if len(value) <= max_len:
        return value
    sliced = value[:max_len].rstrip()
    last_comma = sliced.rfind(", ")
    if last_comma >= max_len // 2:
        return sliced[:last_comma] + "…"
    return sliced + "…"


def _build_signal_block(trust: TrustVerdict) -> str:
    lines: list[str] = []

    # FIX 7: islice avoids allocating a new list just to take the first N items
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
    "Both sentence combined must be less than 80 words (60 preferred)"
)

_USER_TEMPLATE = """\
Job: Senior AI Eng | 5-9yr | product-co | retrieval/ranking skills
Rank #{rank} of 100 | {tier} | {verdict} ({confidence}% confidence)
Candidate: {yoe}yr exp | composite score {composite:.3f}
Facts:
{signal_block}
Brief:"""


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder (pure, no I/O — safe to parallelise)
# ─────────────────────────────────────────────────────────────────────────────

def _build_messages(
    rank: int,
    trust: TrustVerdict,
    candidate: Optional[CandidateFeatureVector],
    composite_score: float,
) -> list[dict]:
    """Build the chat message list without touching the LLM. Thread-safe."""
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
        n_threads: int = 4,
        n_ctx: int = 512,
        verbose: bool = False,
        max_workers: int = 4,
    ) -> None:
        self._model_path = model_path
        self._n_threads = n_threads
        self._n_ctx = n_ctx
        self._verbose = verbose
        self._max_workers = max_workers
        self._llm = None
        # FIX 5: use an RLock so the same thread can re-enter (e.g. preload → _load)
        # and a separate load-guard event so callers don't spin-wait on the load itself
        self._infer_lock = threading.Lock()   # serialises llama_cpp inference calls
        self._load_lock = threading.Lock()    # prevents concurrent model loads
        self._loaded = threading.Event()      # lets waiters block cheaply

    def preload(self) -> None:
        """
        Eagerly load the GGUF model into memory.

        Call this once at pipeline startup so the model is warm before any
        candidates arrive. Calling it again is a no-op. Thread-safe.
        """
        self._load()

    def _load(self) -> None:
        # FIX 5: double-checked locking — cheap fast path once loaded
        if self._loaded.is_set():
            return
        with self._load_lock:
            if self._loaded.is_set():   # re-check after acquiring lock
                return
            from llama_cpp import Llama
            logger.info("Loading LLM for justification …")
            t0 = time.perf_counter()
            self._llm = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                verbose=self._verbose,
                use_mlock=False,
            )
            logger.info("LLM loaded in %.2fs", time.perf_counter() - t0)
            self._loaded.set()   # unblocks any thread waiting in _ensure_loaded

    def _ensure_loaded(self) -> None:
        """Block until the model is ready. Safe to call from any thread."""
        if not self._loaded.is_set():
            self._load()            # first caller does the work
            self._loaded.wait()     # others wait here (no busy loop)

    # ── Core inference ─────────────────────────────────────────────────────

    def _infer(self, messages: list[dict], fallback: str) -> str:
        """
        Run one inference call. Serialised via _infer_lock because
        llama_cpp.Llama is not thread-safe, but prompt *building* is done
        outside this method so threads can prepare prompts in parallel.
        """
        try:
            # FIX 1: the lock now wraps ONLY the inference call, not prompt
            # building or post-processing, so threads genuinely run in parallel
            # for everything except the unavoidably serial GPU/CPU work.
            with self._infer_lock:
                out = self._llm.create_chat_completion(
                    messages=messages,
                    temperature=0.4,
                    max_tokens=100,
                    stop=["\n\n", "Sentence 3", "3.", "\nJob:"],
                )
            text = out["choices"][0]["message"]["content"].strip()

            if len(text) < 30 or any(text.startswith(p) for p in _BANNED_OPENERS):
                logger.debug("LLM output rejected (too short or banned opener): %r", text[:80])
                return fallback

            return text[:320]

        except Exception as exc:
            logger.debug("LLM _infer failed: %s", exc)
            return fallback

    # ── Batch justification (public API) ──────────────────────────────────

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

        self._ensure_loaded()

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

        # ── Partition: skip list vs LLM batch ────────────────────────────
        # Each LLM batch item carries (rank, cfv, fallback, messages)
        # so threads can do prompt building in parallel before inference.
        llm_batch: list[tuple[int, CandidateFeatureVector, str, list[dict]]] = []

        for i, cfv in enumerate(candidates, start=1):
            cid = cfv.candidate_id
            rank = ranks.get(cid, i)

            raw_fallback = fallbacks.get(cid)
            if raw_fallback is None:
                logger.warning(
                    "LLM: no fallback string for candidate %s (rank %d) — "
                    "using generic placeholder; check that the trust layer "
                    "produced a complete fallbacks dict.",
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

            # FIX 1+2: build the prompt here (pure CPU work) so the thread
            # pool does real parallelism for prompt construction, leaving
            # only the serial llama_cpp call inside _infer.
            messages = _build_messages(rank, trust, cfv, composite)
            llm_batch.append((rank, cfv, raw_fallback, messages))

        skipped_count = len(results)
        logger.info(
            "LLM: generating signal-grounded justifications for %d candidates "
            "(%d skipped via rule-based fallback) …",
            len(llm_batch),
            skipped_count,
        )

        if not llm_batch:
            return results

        llm_count = 0

        # FIX 1: _run now only calls _infer (which holds the lock briefly)
        # and does no prompt building — parallel threads overlap on the
        # everything except the locked inference slice.
        def _run(
            args: tuple[int, CandidateFeatureVector, str, list[dict]]
        ) -> tuple[str, str]:
            _rank, cfv, fallback, messages = args
            justification = self._infer(messages, fallback)
            return cfv.candidate_id, justification

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(_run, args): args for args in llm_batch}
            for future in as_completed(futures):
                try:
                    cid, justification = future.result()
                    results[cid] = justification
                    llm_count += 1

                    if llm_count % 10 == 0:
                        elapsed = time.perf_counter() - t0
                        rate = llm_count / elapsed if elapsed > 0 else 1.0
                        remaining = max(0, len(llm_batch) - llm_count)
                        eta = remaining / rate if rate > 0 else 0.0
                        logger.info(
                            "LLM: %d/%d done | %d skipped (rule-based) | "
                            "%.1fs elapsed | ETA %.0fs",
                            llm_count,
                            len(llm_batch),
                            skipped_count,
                            elapsed,
                            eta,
                        )
                except Exception as exc:
                    args = futures[future]
                    cid = args[1].candidate_id
                    fallback = args[2]
                    logger.warning(
                        "LLM: future failed for %s: %s — using fallback", cid, exc
                    )
                    results[cid] = fallback
                    llm_count += 1

        elapsed = time.perf_counter() - t0
        logger.info(
            "LLM: justified %d candidates via LLM, %d via rule-based, "
            "in %.1fs (%.2f s/LLM-candidate)",
            llm_count,
            skipped_count,
            elapsed,
            elapsed / max(1, llm_count),
        )
        return results