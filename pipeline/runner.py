from __future__ import annotations

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            honeypot_ids = HoneypotFilter.load_honeypots()
            for c in self._candidates:
                if c.candidate_id in honeypot_ids:
                    c.is_honeypot = True
        except Exception as e:
            logger.warning("Honeypot filter unavailable: %s", e)
        timings["honeypot_filter"] = (time.perf_counter() - t0) * 1000

        # Remove honeypots from active pool (keep reference for output)
        clean_candidates = [c for c in self._candidates if not c.is_honeypot]
        honeypot_ids = {c.candidate_id for c in self._candidates if c.is_honeypot}
        logger.info("Honeypot filter: %d removed, %d clean", len(honeypot_ids), len(clean_candidates))

        # ── 2. Load indexes ───────────────────────────────────────────────
        t0 = time.perf_counter()
        # Indexes are now loaded lazily inside the paths via from_disk()
        timings["load_indexes"] = (time.perf_counter() - t0) * 1000

        # ── 3. Run 5 retrieval paths IN PARALLEL ─────────────────────
        # Each path loads its own index and runs independently;
        # no shared state between paths, so threads are safe here.
        # Expected speedup: ~3-5x over serial (disk I/O + scoring overlap).
        t0 = time.perf_counter()
        path_results: dict[str, list] = {}
        path_errors: dict[str, Exception] = {}

        def _run_semantic():
            from retrieval.semantic_path import SemanticPath
            sp = SemanticPath.from_disk()
            return "semantic", sp.retrieve(self._jd, top_k=config.SEMANTIC_PATH_TOP_K)

        def _run_keyword():
            from retrieval.keyword_path import KeywordPath
            kp = KeywordPath.from_disk()
            return "keyword", kp.retrieve(self._jd, top_k=config.KEYWORD_PATH_TOP_K)

        def _run_ontology():
            from retrieval.ontology_path import OntologyPath
            op = OntologyPath.from_skill_map()
            skills_map = OntologyPath.build_skills_map(self._candidates)
            return "ontology", op.retrieve(self._jd, candidate_skills_map=skills_map, top_k=config.ONTOLOGY_PATH_TOP_K)

        def _run_trajectory():
            from retrieval.trajectory_path import TrajectoryPath
            tp = TrajectoryPath.from_disk()
            return "trajectory", tp.retrieve(self._jd, top_k=config.TRAJECTORY_PATH_TOP_K)

        def _run_signal():
            from retrieval.signal_path import SignalPath
            sigp = SignalPath.from_disk()
            return "signal", sigp.retrieve(top_k=config.SIGNAL_PATH_TOP_K)

        _retrieval_fns = [_run_semantic, _run_keyword, _run_ontology, _run_trajectory, _run_signal]
        _REQUIRED_PATHS = {"semantic", "keyword", "ontology", "trajectory", "signal"}

        with ThreadPoolExecutor(max_workers=5, thread_name_prefix="retrieval") as exe:
            futures = {exe.submit(fn): fn.__name__ for fn in _retrieval_fns}
            for fut in as_completed(futures):
                fname = futures[fut]
                try:
                    path_name, res = fut.result()
                    path_results[path_name] = res
                    logger.debug("Retrieval path '%s': %d results", path_name, len(res))
                except Exception as exc:
                    path_errors[fname] = exc
                    logger.error("Retrieval path '%s' failed: %s", fname, exc)

        # Raise on any critical path failure
        for fname, exc in path_errors.items():
            raise RuntimeError(f"Retrieval path {fname} failed: {exc}") from exc

        all_retrieval_results = [r for res in path_results.values() for r in res]
        timings["retrieval_paths"] = (time.perf_counter() - t0) * 1000
        logger.info(
            "Retrieval: %d total results across %d paths (parallel)",
            len(all_retrieval_results), len(path_results),
        )

        # ── 4. RRF Fusion ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        rrf_pool = []
        try:
            from retrieval.rrf_fusion import RRFFusion
            rrf = RRFFusion()
            rrf_pool = rrf.fuse(path_results)
        except Exception as e:
            logger.error("RRF fusion failed: %s", e)
            raise RuntimeError(f"RRF fusion failed: {e}") from e
        timings["rrf_fusion"] = (time.perf_counter() - t0) * 1000
        logger.info("RRF pool: %d candidates", len(rrf_pool))

        # Remove honeypots from RRF pool
        rrf_pool = [r for r in rrf_pool if r.candidate_id not in honeypot_ids]
        rrf_pool = rrf_pool[:config.CROSS_ENCODER_TOP_K]

        # ── 5. Cross-encoder rerank + LLM pool pre-warm (overlapped) ──────
        # Spawn the LLM worker pool in a background thread NOW so model
        # loading (~15-30s) overlaps with cross-encoder scoring.
        # By the time we reach step 8 the pool is already hot.
        t0 = time.perf_counter()
        _llm_reranker_ref: list = []  # mutable container so thread can write
        _llm_preload_error: list = []  # capture any preload exception

        def _preload_llm():
            if not getattr(config, "LLM_RERANKER_ENABLED", True):
                return
            try:
                from scoring.llm_reranker import LLMReranker
                reranker = LLMReranker(
                    model_path=config.LLM_MODEL_PATH,
                    n_ctx=getattr(config, "LLM_N_CTX", 512),
                    max_workers=getattr(config, "LLM_MAX_WORKERS", 4),
                    n_threads_per_worker=getattr(config, "LLM_N_THREADS_PER_WORKER", 2),
                )
                reranker.preload()  # blocks until all workers are warm
                _llm_reranker_ref.append(reranker)
                logger.info("LLM: pool pre-warmed in background thread.")
            except Exception as exc:
                _llm_preload_error.append(exc)
                logger.warning("LLM pre-warm failed (will retry inline): %s", exc)

        _llm_thread = threading.Thread(target=_preload_llm, name="llm-preload", daemon=True)
        _llm_thread.start()
        logger.info("LLM: pool pre-warm started in background — overlapping with cross-encoder.")

        try:
            from scoring.cross_encoder import CrossEncoderReranker
            ce = CrossEncoderReranker()
            rrf_pool = ce.rerank(rrf_pool, self._jd, self._candidate_store)
        except Exception as e:
            logger.error("Cross-encoder unavailable: %s", e)
            raise RuntimeError(f"Cross-encoder unavailable: {e}") from e
        timings["cross_encoder"] = (time.perf_counter() - t0) * 1000

        # ── 6. Composite scoring ──────────────────────────────────────────
        t0 = time.perf_counter()
        composite_results = []
        # Cache behavioral_scorer so trust layer can reuse it (avoids double-scoring).
        _behavioral_scorer_instance = None
        try:
            from scoring.behavioral import BehavioralScorer
            from scoring.composite import CompositeScorer
            _behavioral_scorer_instance = BehavioralScorer()
            scorer = CompositeScorer(self._jd, self._candidate_store, _behavioral_scorer_instance)
            composite_results = scorer.rank(rrf_pool)
        except Exception as e:
            logger.error("Composite scoring failed: %s", e)
            raise RuntimeError(f"Composite scoring error: {e}") from e
        timings["composite_scoring"] = (time.perf_counter() - t0) * 1000
        logger.info("Composite scored %d candidates", len(composite_results))


        # ── 6c. Min-max score normalization ──────────────────────────────
        # Spreads the score range to [0.1, 1.0] for non-disqualified candidates,
        # making the score column more discriminative and useful for evaluation.
        # Disqualified (score == 0.0) candidates stay at 0.0.
        import dataclasses
        non_zero_scores = [r.final_score for r in composite_results if r.final_score > 0.0]
        if len(non_zero_scores) >= 2:
            s_min = min(non_zero_scores)
            s_max = max(non_zero_scores)
            score_range = s_max - s_min
            if score_range > 1e-6:
                composite_results = [
                    dataclasses.replace(
                        r,
                        final_score=round(
                            0.10 + 0.90 * (r.final_score - s_min) / score_range, 6
                        ),
                    ) if r.final_score > 0.0 else r
                    for r in composite_results
                ]
                logger.info(
                    "Score normalization: range [%.4f, %.4f] → [0.10, 1.00] "
                    "(%d non-zero candidates)",
                    s_min, s_max, len(non_zero_scores),
                )

        # ── 7. Trust layer & Reasoning ────────────────────────────────────
        t0 = time.perf_counter()
        trust_verdicts: dict = {}
        reasonings: dict[str, str] = {}
        schema_components: dict[str, ComponentScores] = {}

        top_candidates = []
        for cs in composite_results[:top_k]:
            cfv = self._candidate_store.get(cs.candidate_id)
            if cfv:
                top_candidates.append(cfv)

        try:
            from trust.advocate import build_advocate_signals
            from trust.skeptic import build_skeptic_signals
            from trust.verdict import build_verdict
            from trust.reasoning_generator import generate_reasoning
            from scoring.skill_match import SkillMatchScorer
            from scoring.career_quality import CareerQualityScorer

            # ── Run 3 independent scorers in parallel ──────────────────────
            # SkillMatch, CareerQuality, Behavioral each iterate over top-100
            # independently — no shared mutable state. 3-worker pool gives ~3x
            # speedup on the trust-layer scoring phase (~8s → ~3s).
            def _score_skills():
                return SkillMatchScorer().score_all(top_candidates, self._jd)

            def _score_career():
                return CareerQualityScorer(self._jd).score_all(top_candidates)

            def _score_behavioral():
                return _behavioral_scorer_instance.score_all(top_candidates)

            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="trust") as tex:
                f_skill = tex.submit(_score_skills)
                f_career = tex.submit(_score_career)
                f_behavioral = tex.submit(_score_behavioral)
                skill_results = f_skill.result()
                career_results = f_career.result()
                behavioral_results = f_behavioral.result()

            for rank_pos, cs in enumerate(composite_results[:top_k], start=1):
                cid = cs.candidate_id
                cfv = self._candidate_store.get(cid)
                if not cfv:
                    continue

                s_res = skill_results.get(cid)
                c_res = career_results.get(cid)
                b_res = behavioral_results.get(cid)

                missing = [
                    name for name, res in
                    [("skill", s_res), ("career", c_res), ("behavioral", b_res)]
                    if not res
                ]
                if missing:
                    logger.warning(
                        "Trust: missing scorer results for %s (rank %d): %s — "
                        "no trust verdict will be built; rule-based fallback will be used.",
                        cid, rank_pos, missing,
                    )

                if s_res and c_res and b_res:
                    schema_comp = ComponentScores(
                        candidate_id=cid,
                        skill_score=cs.skill_match_score,
                        career_score=cs.career_quality_score,
                        behavioral_score=cs.behavioral_score,
                        required_skill_coverage=s_res.required_score,
                        nice_to_have_coverage=s_res.nice_to_have_score,
                        ontology_skills_matched=[],
                        yoe_score=c_res.yoe_score,
                        trajectory_velocity=cs.trajectory_velocity,
                        product_co_flag=cfv.has_product_co_experience,
                        consulting_only_flag=cfv.is_consulting_only,
                        location_bonus=cs.location_bonus_applied,
                        recency_score=b_res.recency_score,
                        notice_period_score=b_res.notice_period_score,
                        uncertainty_penalty=cs.uncertainty_penalty_applied,
                        signal_count=b_res.signal_count,
                    )
                    schema_components[cid] = schema_comp

                    adv_signals = build_advocate_signals(cfv, schema_comp, s_res, self._jd)
                    skep_signals = build_skeptic_signals(cfv, schema_comp, c_res, b_res, s_res, self._jd)
                    verdict = build_verdict(cfv, schema_comp, adv_signals, skep_signals)
                    trust_verdicts[cid] = verdict

                    reasonings[cid] = generate_reasoning(rank_pos, verdict, cfv, schema_comp)
        except Exception as e:
            logger.error("Trust layer unavailable: %s", e)
            raise RuntimeError(f"Trust layer unavailable: {e}") from e

        for cs in composite_results[:top_k]:
            cid = cs.candidate_id
            if cid not in reasonings:
                cfv = self._candidate_store.get(cid)
                yoe = f"{cfv.years_of_experience:.0f}y" if cfv else "?"
                reasonings[cid] = (
                    f"Ranked #{cid} with composite score {cs.final_score:.4f}. "
                    f"Skill: {cs.skill_match_score:.2f}, Career: {cs.career_quality_score:.2f}, "
                    f"Behavioral: {cs.behavioral_score:.2f}. "
                    f"Experience: {yoe}. "
                    f"Retrieved via: {', '.join(cs.paths_present)}."
                )

        timings["trust_layer"] = (time.perf_counter() - t0) * 1000

        # ── 8. LLM Justification layer ─────────────────────────────────
        # Pool was pre-warmed in the background thread during step 5 (CE).
        # We just join the thread to make sure pre-warm is complete, then
        # dispatch tasks immediately — no model-load wait here.
        t0 = time.perf_counter()
        try:
            if getattr(config, "LLM_RERANKER_ENABLED", True):
                # Wait for the pre-warm thread to finish (usually already done).
                _llm_thread.join()

                if _llm_preload_error and not _llm_reranker_ref:
                    # Pre-warm failed; try a fresh inline load as fallback.
                    logger.warning(
                        "LLM pre-warm failed earlier: %s — retrying inline.",
                        _llm_preload_error[0],
                    )
                    from scoring.llm_reranker import LLMReranker
                    _inline_reranker = LLMReranker(
                        model_path=config.LLM_MODEL_PATH,
                        n_ctx=getattr(config, "LLM_N_CTX", 512),
                        max_workers=getattr(config, "LLM_MAX_WORKERS", 4),
                        n_threads_per_worker=getattr(config, "LLM_N_THREADS_PER_WORKER", 2),
                    )
                    _llm_reranker_ref.append(_inline_reranker)

                if not _llm_reranker_ref:
                    raise RuntimeError("LLM reranker could not be initialised.")

                llm_reranker = _llm_reranker_ref[0]

                top100_cfvs = [
                    self._candidate_store[cs.candidate_id]
                    for cs in composite_results[:top_k]
                    if cs.candidate_id in self._candidate_store
                ]
                ranks_map: dict[str, int] = {
                    cs.candidate_id: (pos + 1)
                    for pos, cs in enumerate(composite_results[:top_k])
                }
                llm_top_n = getattr(config, "LLM_RERANKER_TOP_N", 50)
                logger.info(
                    "LLM: dispatching top-%d candidates (pool already warm);"
                    " ranks %d–%d use rule-based reasoning.",
                    llm_top_n, llm_top_n + 1, top_k,
                )
                composite_scores_map: dict[str, float] = {
                    cs.candidate_id: cs.final_score
                    for cs in composite_results[:top_k]
                }
                llm_justifications = llm_reranker.justify_candidates(
                    candidates=top100_cfvs,
                    jd=self._jd,
                    ranks=ranks_map,
                    trust_verdicts=dict(trust_verdicts),
                    fallbacks=dict(reasonings),
                    top_n=llm_top_n,
                    composite_scores=composite_scores_map,
                )
                for cid, justification in llm_justifications.items():
                    if justification:
                        reasonings[cid] = justification

                llm_reranker.shutdown()
                logger.info(
                    "LLM: justified %d/%d top candidates.",
                    len(llm_justifications), top_k,
                )
        except Exception as e:
            logger.warning("LLM justification unavailable, using rule-based reasoning: %s", e)
        timings["llm_justification"] = (time.perf_counter() - t0) * 1000

        # ── 9. Assemble RankedCandidate list ──────────────────────────────
        t0 = time.perf_counter()
        ranked: list[RankedCandidate] = []

        for rank_pos, cs in enumerate(composite_results[:top_k], start=1):
            cid = cs.candidate_id
            cfv = self._candidate_store.get(cid)
            
            schema_comp = schema_components.get(cid)
            if not schema_comp:
                schema_comp = ComponentScores(
                    candidate_id=cid,
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
                candidate_id=cid,
                rank=rank_pos,
                final_score=cs.final_score,
                reasoning=reasonings.get(cid, ""),
                components=schema_comp,
                trust=trust_verdicts.get(cid),
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
