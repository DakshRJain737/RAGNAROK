from __future__ import annotations

import logging
import time
from pathlib import Path

import config
from pipeline.candidate_parser import CandidateParser
from pipeline.jd_parser import JDParser
from indexing.honeypot_registry import HoneypotFilter
from indexing.trajectory_builder import TrajectoryAnalyzer
from indexing.faiss_builder import FaissIndex
from indexing.bm25_builder import BM25Index
from indexing.feature_store import FeatureStore
from scoring.honeypot_filter import HoneypotCleanup


# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
)
logger = logging.getLogger("build_indexes")


# ── Timer helper ──────────────────────────────────────────────────────────────

class _Stage:
    """Context manager that logs the wall-clock time of each pipeline stage."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._t0: float = 0.0

    def __enter__(self) -> "_Stage":
        logger.info("━━━ [START] %s ━━━", self.name)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        elapsed = time.perf_counter() - self._t0
        logger.info("━━━ [DONE]  %s — %.1fs ━━━\n", self.name, elapsed)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    wall_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("RAGnarok — Full Index Build")
    logger.info("Source : %s", config.CANDIDATES_JSONL)
    logger.info("Output : %s", config.INDEXES_DIR)
    logger.info("=" * 70 + "\n")

    # Ensure output directory exists
    config.INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Parse JSONL ──────────────────────────────────────────────────
    with _Stage("1/7  Parse candidates.jsonl"):
        parser = CandidateParser()
        candidates = parser.build_candidate_list_from_jsonl(config.CANDIDATES_JSONL)
        logger.info("Parsed %d candidates", len(candidates))

    # ── Stage 2: Honeypot detection ───────────────────────────────────────────
    with _Stage("2/7  Honeypot detection"):
        honeypot_filter = HoneypotFilter()
        honeypot_filter.run_honeypot_filters(candidates)

        n_honeypots = sum(1 for c in candidates if c.is_honeypot)
        logger.info("Flagged %d honeypot candidates", n_honeypots)

        honeypot_cleanup = HoneypotCleanup()
        candidates = honeypot_cleanup.cleanup_candidates(candidates)
        logger.info("Clean pool: %d candidates (after removing honeypots)", len(candidates))

    # ── Stage 3: Trajectory analysis ──────────────────────────────────────────
    with _Stage("3/7  Trajectory analysis"):
        trajectory_analyzer = TrajectoryAnalyzer()
        trajectory_analyzer.build_all_feature_vector(candidates)
        logger.info("Trajectory vectors built for %d candidates", len(candidates))

    # ── Stage 4: Parse job description (needed by some scorers / JD encoder) ──
    with _Stage("4/7  Parse job description"):
        jd_parser = JDParser()
        # encode=True loads the bi-encoder to create the JD embedding vector.
        # This is stored inside `intent` and used downstream by rank.py.
        intent = jd_parser.parse(config.JD_PATH, encode=True)
        logger.info("Job description parsed and encoded")

    # ── Stage 5: FAISS dense index ────────────────────────────────────────────
    with _Stage("5/7  FAISS dense index (IVF256)"):
        faiss_index = FaissIndex()
        faiss_index.build(candidates, save=True)
        logger.info(
            "FAISS index: %d vectors  →  %s",
            faiss_index.total_vectors,
            config.FAISS_INDEX_PATH,
        )

    # ── Stage 6: BM25 keyword index ───────────────────────────────────────────
    with _Stage("6/7  BM25 keyword index"):
        bm25 = BM25Index()
        bm25.build(candidates, save=True)
        logger.info(
            "BM25 index: %d candidates, %d unique tokens  →  %s",
            len(bm25._id_map),
            bm25.vocab_size,
            config.BM25_INDEX_PATH,
        )

    # ── Stage 7: Feature store ────────────────────────────────────────────────
    with _Stage("7/7  Feature store (30-dim)"):
        feature_store = FeatureStore()
        matrix = feature_store.build(candidates, save=True)
        logger.info(
            "FeatureStore: shape=%s  →  %s",
            matrix.shape,
            config.FEATURE_STORE_PATH,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - wall_start
    logger.info("=" * 70)
    logger.info("All indexes built successfully in %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("")
    logger.info("Index files written to: %s", config.INDEXES_DIR)
    for f in sorted(config.INDEXES_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 ** 2)
        logger.info("  %-35s  %.1f MB", f.name, size_mb)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
