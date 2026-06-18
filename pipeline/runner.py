"""
pipeline/runner.py — End-to-end pipeline orchestrator for RAGnarok.

Runs all ranking stages in order and returns a list of RankedCandidate objects.

Stage order:
    1. Load pre-built indexes (FAISS, BM25, feature store, trajectory, honeypot)
    2. Encode JD query via bi-encoder
    3. Run 5 retrieval paths in parallel
    4. RRF fusion → top-60 pool
    5. Honeypot filter → top-50
    6. Cross-encoder rerank
    7. Composite scoring (0.40×skill + 0.35×career + 0.25×behavioral)
    8. Trust layer (Advocate + Skeptic + Verdict)
    9. Reasoning generation
    10. Assemble RankedCandidate list

Called by:
    api/routes/rank.py  →  PipelineRunner(jd, candidates).run(top_k)
    rank.py             →  direct CLI invocation
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import config
from pipeline.schemas import (
    JDIntent,
    CandidateFeatureVector,
    RankedCandidate,
    ComponentScores,
)

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Orchestrates the full ranking pipeline.

    Args:
        jd:         Parsed JDIntent (from pipeline/jd_parser.py).
        candidates: List of CandidateFeatureVector (from pipeline/candidate_parser.py).

    Usage:
        runner = PipelineRunner(jd=intent, candidates=candidates)
        ranked, timings = runner.run(top_k=100)
    """

    def __init__(
        self,
        jd: JDIntent,
        candidates: list[CandidateFeatureVector],
    ) -> None:
        self._jd = jd
        self._candidates = candidates
        self._candidate_store: dict[str, CandidateFeatureVector] = {
            c.candidate_id: c for c in candidates
        }

    def run(
        self,
        top_k: int = 100,
    ) -> tuple[list[RankedCandidate], dict[str, float]]:
        """
        Run the full pipeline. Returns (ranked_candidates, stage_timings_ms).
        """
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        # ── 1. Honeypot filter (mark flagged candidates) ──────────────────
        t0 = time.perf_counter()
        try:
            from indexing.honeypot_registry import HoneypotFilter
            hpf = HoneypotFilter()
            hpf.run_honeypot_filters(self._candidates)
        except Exception as e:
            logger.warning("Honeypot filter unavailable: %s", e)
        timings["honeypot_filter"] = (time.perf_counter() - t0) * 1000

        # Remove honeypots from active pool (keep reference for output)
        clean_candidates = [c for c in self._candidates if not c.is_honeypot]
        honeypot_ids = {c.candidate_id for c in self._candidates if c.is_honeypot}
        logger.info("Honeypot filter: %d removed, %d clean", len(honeypot_ids), len(clean_candidates))

        # ── 2. Load indexes ───────────────────────────────────────────────
        t0 = time.perf_counter()
        faiss_index = bm25_index = feature_store = trajectory_store = None
        try:
            from indexing.faiss_builder import FaissIndex
            faiss_index = FaissIndex()
            faiss_index.load()
        except Exception as e:
            logger.warning("FAISS index unavailable: %s", e)
        try:
            from indexing.bm25_builder import BM25Index
            bm25_index = BM25Index()
            bm25_index.load()
        except Exception as e:
            logger.warning("BM25 index unavailable: %s", e)
        try:
            from indexing.feature_store import FeatureStore
            feature_store = FeatureStore()
            feature_store.load()
        except Exception as e:
            logger.warning("Feature store unavailable: %s", e)
        timings["load_indexes"] = (time.perf_counter() - t0) * 1000

        # ── 3. Run 5 retrieval paths ──────────────────────────────────────
        t0 = time.perf_counter()
        all_retrieval_results = []

        # Path 1: Semantic (FAISS)
        try:
            from retrieval.semantic_path import SemanticPath
            sp = SemanticPath(faiss_index)
            all_retrieval_results.extend(sp.retrieve(self._jd, top_k=config.SEMANTIC_PATH_TOP_K))
        except Exception as e:
            logger.warning("Semantic path failed: %s", e)

        # Path 2: Keyword (BM25)
        try:
            from retrieval.keyword_path import KeywordPath
            kp = KeywordPath(bm25_index)
            all_retrieval_results.extend(kp.retrieve(self._jd, top_k=config.KEYWORD_PATH_TOP_K))
        except Exception as e:
            logger.warning("Keyword path failed: %s", e)

        # Path 3: Ontology
        try:
            from retrieval.ontology_path import OntologyPath
            op = OntologyPath(self._candidate_store)
            all_retrieval_results.extend(op.retrieve(self._jd, top_k=config.ONTOLOGY_PATH_TOP_K))
        except Exception as e:
            logger.warning("Ontology path failed: %s", e)

        # Path 4: Trajectory
        try:
            from retrieval.trajectory_path import TrajectoryPath
            tp = TrajectoryPath(self._candidate_store)
            all_retrieval_results.extend(tp.retrieve(self._jd, top_k=config.TRAJECTORY_PATH_TOP_K))
        except Exception as e:
            logger.warning("Trajectory path failed: %s", e)

        # Path 5: Signal (behavioral)
        try:
            from retrieval.signal_path import SignalPath
            sigp = SignalPath(feature_store, self._candidate_store)
            all_retrieval_results.extend(sigp.retrieve(self._jd, top_k=config.SIGNAL_PATH_TOP_K))
        except Exception as e:
            logger.warning("Signal path failed: %s", e)

        timings["retrieval_paths"] = (time.perf_counter() - t0) * 1000
        logger.info("Retrieval: %d total results across all paths", len(all_retrieval_results))

        # ── 4. RRF Fusion ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        rrf_pool = []
        try:
            from retrieval.rrf_fusion import RRFFusion
            rrf = RRFFusion()
            rrf_pool = rrf.fuse(all_retrieval_results, top_n=config.RRF_POOL_SIZE)
        except Exception as e:
            logger.warning("RRF fusion failed: %s — using direct retrieval results", e)
            # Fallback: deduplicate and take top candidates
            seen = set()
            for r in all_retrieval_results:
                if r.candidate_id not in seen and r.candidate_id not in honeypot_ids:
                    seen.add(r.candidate_id)
                    from pipeline.schemas import RRFResult
                    rrf_pool.append(RRFResult(
                        candidate_id=r.candidate_id,
                        rrf_score=r.path_score,
                        paths_present=[r.path_name],
                    ))
                if len(rrf_pool) >= config.RRF_POOL_SIZE:
                    break
        timings["rrf_fusion"] = (time.perf_counter() - t0) * 1000
        logger.info("RRF pool: %d candidates", len(rrf_pool))

        # Remove honeypots from RRF pool
        rrf_pool = [r for r in rrf_pool if r.candidate_id not in honeypot_ids]
        rrf_pool = rrf_pool[:config.CROSS_ENCODER_TOP_K]

        # ── 5. Cross-encoder rerank ───────────────────────────────────────
        t0 = time.perf_counter()
        try:
            from scoring.cross_encoder import CrossEncoderReranker
            ce = CrossEncoderReranker()
            rrf_pool = ce.rerank(rrf_pool, self._jd, self._candidate_store)
        except Exception as e:
            logger.warning("Cross-encoder unavailable: %s — skipping CE blend", e)
        timings["cross_encoder"] = (time.perf_counter() - t0) * 1000

        # ── 6. Composite scoring ──────────────────────────────────────────
        t0 = time.perf_counter()
        composite_results = []
        try:
            from scoring.behavioral import BehavioralScorer
            from scoring.composite import CompositeScorer
            bscorer = BehavioralScorer()
            scorer = CompositeScorer(self._jd, self._candidate_store, bscorer)
            composite_results = scorer.rank(rrf_pool)
        except Exception as e:
            logger.error("Composite scoring failed: %s", e)
            raise RuntimeError(f"Composite scoring error: {e}") from e
        timings["composite_scoring"] = (time.perf_counter() - t0) * 1000
        logger.info("Composite scored %d candidates", len(composite_results))

        # ── 7. Trust layer ────────────────────────────────────────────────
        t0 = time.perf_counter()
        trust_verdicts: dict = {}
        try:
            from trust.verdict import TrustVerdictEngine
            trust_engine = TrustVerdictEngine()
            for cs in composite_results:
                cfv = self._candidate_store.get(cs.candidate_id)
                if cfv:
                    verdict = trust_engine.evaluate(cfv, cs, self._jd)
                    trust_verdicts[cs.candidate_id] = verdict
        except Exception as e:
            logger.warning("Trust layer unavailable: %s — skipping", e)
        timings["trust_layer"] = (time.perf_counter() - t0) * 1000

        # ── 8. Reasoning generation ───────────────────────────────────────
        t0 = time.perf_counter()
        reasonings: dict[str, str] = {}
        try:
            from trust.reasoning_generator import ReasoningGenerator
            rgen = ReasoningGenerator()
            for cs in composite_results:
                verdict = trust_verdicts.get(cs.candidate_id)
                cfv = self._candidate_store.get(cs.candidate_id)
                reasonings[cs.candidate_id] = rgen.generate(cfv, cs, verdict, self._jd)
        except Exception as e:
            logger.warning("Reasoning generator unavailable: %s — using fallback", e)
            for cs in composite_results:
                cfv = self._candidate_store.get(cs.candidate_id)
                yoe = f"{cfv.years_of_experience:.0f}y" if cfv else "?"
                reasonings[cs.candidate_id] = (
                    f"Ranked #{cs.candidate_id} with composite score {cs.final_score:.4f}. "
                    f"Skill: {cs.skill_match_score:.2f}, Career: {cs.career_quality_score:.2f}, "
                    f"Behavioral: {cs.behavioral_score:.2f}. "
                    f"Experience: {yoe}. "
                    f"Retrieved via: {', '.join(cs.paths_present)}."
                )
        timings["reasoning"] = (time.perf_counter() - t0) * 1000

        # ── 9. Assemble RankedCandidate list ──────────────────────────────
        t0 = time.perf_counter()
        ranked: list[RankedCandidate] = []

        for rank_pos, cs in enumerate(composite_results[:top_k], start=1):
            cfv = self._candidate_store.get(cs.candidate_id)
            verdict = trust_verdicts.get(cs.candidate_id)
            reasoning = reasonings.get(cs.candidate_id, "")

            # Build pipeline.schemas.ComponentScores for trust layer compatibility
            schema_comp = ComponentScores(
                candidate_id=cs.candidate_id,
                skill_score=cs.skill_match_score,
                career_score=cs.career_quality_score,
                behavioral_score=cs.behavioral_score,
                required_skill_coverage=0.0,
                nice_to_have_coverage=0.0,
                ontology_skills_matched=[],
                yoe_score=0.0,
                trajectory_velocity=cs.trajectory_velocity,
                product_co_flag=cfv.has_product_co_experience if cfv else False,
                consulting_only_flag=cfv.is_consulting_only if cfv else False,
                location_bonus=cs.location_bonus_applied,
                recency_score=0.0,
                notice_period_score=0.0,
                uncertainty_penalty=cs.uncertainty_penalty_applied,
                signal_count=5,
            )

            ranked.append(RankedCandidate(
                candidate_id=cs.candidate_id,
                rank=rank_pos,
                final_score=cs.final_score,
                reasoning=reasoning,
                components=schema_comp,
                trust=verdict,
                feature_vector=cfv,
            ))

        timings["assemble"] = (time.perf_counter() - t0) * 1000
        timings["total"] = (time.perf_counter() - total_start) * 1000

        logger.info(
            "Pipeline complete: %d candidates ranked in %.1fms (top score=%.4f)",
            len(ranked),
            timings["total"],
            ranked[0].final_score if ranked else 0.0,
        )
        return ranked, timings
