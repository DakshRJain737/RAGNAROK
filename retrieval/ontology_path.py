"""
retrieval/ontology_path.py
--------------------------
Retrieval Path 3: Ontology graph domain-transfer rescue (Tier-5 candidates).

Problem it solves:
    Paths 1 (FAISS) and 2 (BM25) both need text or embedding overlap with
    JD terms. A candidate whose profile reads:

        "Built a real-time recommendation system at Swiggy. XGBoost ranker
         with collaborative filtering and feature engineering. A/B tested
         click-through vs dwell-time objectives."

    …scores near zero on both dense and sparse retrieval for a JD that
    requires "information retrieval", "vector search", "ranking evaluation"
    — because none of those exact phrases appear in their profile.

    This path rescues them. The domain-transfer edge:
        "recommendation systems" → "information retrieval"
    means a candidate with RecSys skills is a plausible fit for an IR role.
    SkillGraph.rank_by_domain_transfer() performs the BFS walk and scoring;
    this file is the clean runner-facing adapter.

Architecture position:
    Path 3 of 5 in the parallel retrieval stage.
    Results feed into rrf_fusion.py with a 1.3x bonus weight
    (config.RRF_ONTOLOGY_PATH_BONUS) to give domain-transfer candidates
    a fair chance against candidates who appear in multiple paths.

This file is intentionally thin — all graph logic lives in
ontology/graph_traversal.py. This file handles:
    - Lifecycle management of the SkillGraph
    - Adapting graph output (list[tuple[str, float]]) to list[RetrievalResult]
    - The build_skills_map() helper used by pipeline/runner.py
    - Logging and timing

Consumed by:
    retrieval/rrf_fusion.py       (merges results from all 5 paths)
    pipeline/runner.py            (Path 3 of the ranking pipeline)

Dependencies:
    config.py                     ONTOLOGY_PATH_TOP_K, SKILL_MAP_PATH
    pipeline/schemas.py           JDIntent, RetrievalResult, CandidateFeatureVector
    ontology/graph_traversal.py   SkillGraph (all heavy logic here)
    stdlib                        logging, pathlib, time, typing
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import config
from ontology.graph_traversal import SkillGraph
from pipeline.schemas import CandidateFeatureVector, JDIntent, RetrievalResult

logger = logging.getLogger(__name__)


class OntologyPath:
    """
    Ontology graph domain-transfer retrieval path (Path 3 of 5).

    Scores all candidates by how well their skills map into JD required skills
    via the domain-transfer edges in skill_map.json:

        "recommendation systems" → "information retrieval"
        "nlp"                    → "information retrieval"
        "search engineering"     → "ranking"
        …

    A candidate is rescued when they have source-domain skills whose
    domain-transfer edges point to at least one JD required skill.

    Typical production usage (pipeline/runner.py):

        # Build once at startup from skill_map.json
        path = OntologyPath()

        # Build candidate skills map once from all feature vectors
        skills_map = OntologyPath.build_skills_map(all_feature_vectors)

        # Retrieve top-20 domain-transfer candidates per JD
        results = path.retrieve(jd_intent, candidate_skills_map=skills_map)

    Unit-test usage (no runner infrastructure needed):

        graph = SkillGraph(skill_map_path)
        path  = OntologyPath(skill_graph=graph)
        results = path.retrieve(
            jd_intent,
            candidate_skills_map={
                "CAND_0000001": frozenset({"recommendation systems", "python"}),
                "CAND_0000002": frozenset({"marketing", "excel"}),
            },
        )
    """

    PATH_NAME: str = "ontology"

    def __init__(
        self,
        skill_graph: Optional[SkillGraph] = None,
        skill_map_path: Optional[Path] = None,
    ) -> None:
        """
        Initialise the ontology path.

        Args:
            skill_graph:    Pre-loaded SkillGraph. Takes priority over
                            skill_map_path when supplied. Pass a pre-built
                            instance to share it across paths in runner.py
                            and avoid loading skill_map.json twice.
            skill_map_path: Path to skill_map.json.
                            Defaults to config.SKILL_MAP_PATH.
                            Ignored when skill_graph is supplied.

        Raises:
            FileNotFoundError: skill_map.json not found at resolved path
                               (raised by SkillGraph.__init__).
        """
        if skill_graph is not None:
            self._graph: SkillGraph = skill_graph
        else:
            effective_path = skill_map_path or config.SKILL_MAP_PATH
            self._graph = SkillGraph(skill_map_path=effective_path)

        logger.info("OntologyPath initialised — %r", self._graph)

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_skill_map(
        cls,
        skill_map_path: Optional[Path] = None,
    ) -> "OntologyPath":
        """
        Named constructor: build from a skill_map.json path.

        Equivalent to OntologyPath(skill_map_path=…) but reads more
        clearly in runner.py alongside SemanticPath.from_disk() and
        KeywordPath.from_disk().

        Args:
            skill_map_path: Override for config.SKILL_MAP_PATH.

        Returns:
            Ready OntologyPath instance.
        """
        return cls(skill_map_path=skill_map_path)

    # ------------------------------------------------------------------ #
    # Primary retrieve method                                              #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        jd_intent: JDIntent,
        candidate_skills_map: dict[str, frozenset[str]],
        top_k: int = config.ONTOLOGY_PATH_TOP_K,
        exclude_ids: Optional[set[str]] = None,
        bfs_depth: int = 1,
    ) -> list[RetrievalResult]:
        """
        Score all candidates by domain-transfer alignment with JD required skills.

        Unlike Paths 1 (FAISS) and 2 (BM25), this path must iterate the
        entire candidate pool — there is no pre-built ANN or inverted index.
        This is fast in practice (O(N × K) set intersections where K = ~20
        required skills) and runs in < 200 ms for 100 K candidates.

        Args:
            jd_intent:            Parsed JDIntent. Uses required_skills as
                                  the BFS seed — NOT expanded_required.
                                  The graph performs its own ontology expansion
                                  via the domain_transfers section.
            candidate_skills_map: {candidate_id: frozenset[skill_names_lower]}
                                  Built with OntologyPath.build_skills_map().
                                  All candidates to score must be present here.
            top_k:                Maximum results to return.
                                  Defaults to config.ONTOLOGY_PATH_TOP_K (20).
            exclude_ids:          Optional set of candidate_ids to skip.
                                  NOTE: Do NOT pre-filter here. Deduplication
                                  is handled by rrf_fusion.py. This param is
                                  provided for testing cross-path recall only.
            bfs_depth:            BFS depth passed to SkillGraph. Default 1
                                  (direct domain transfers). Depth 2 adds
                                  indirect transfers but increases noise.

        Returns:
            list[RetrievalResult] sorted by domain-transfer score descending,
            length ≤ top_k. Only candidates with score > 0.0 are included.

            path_name    = "ontology"
            path_score   ∈ (0.0, 1.0]  (from SkillGraph.score_candidate_skills)
            rank_in_path = 1-indexed position in this path's results

        Raises:
            ValueError: top_k < 1.
            TypeError:  candidate_skills_map is not a dict.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        if not isinstance(candidate_skills_map, dict):
            raise TypeError(
                "candidate_skills_map must be dict[str, frozenset[str]], "
                f"got {type(candidate_skills_map).__name__}."
            )

        if not jd_intent.required_skills:
            logger.warning(
                "OntologyPath.retrieve: jd_intent.required_skills is empty. "
                "No domain-transfer scoring possible. Returning []."
            )
            return []

        if not candidate_skills_map:
            logger.warning(
                "OntologyPath.retrieve: candidate_skills_map is empty. "
                "Returning []."
            )
            return []

        t0 = time.perf_counter()

        # Delegate all graph logic to SkillGraph
        ranked: list[tuple[str, float]] = self._graph.rank_by_domain_transfer(
            candidate_skills_map=candidate_skills_map,
            jd_required_skills=jd_intent.required_skills,
            top_k=top_k,
            exclude_ids=exclude_ids,
            bfs_depth=bfs_depth,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Adapt (candidate_id, score) tuples → RetrievalResult objects
        results: list[RetrievalResult] = [
            RetrievalResult(
                candidate_id=candidate_id,
                path_score=score,
                path_name=self.PATH_NAME,
                rank_in_path=rank + 1,
            )
            for rank, (candidate_id, score) in enumerate(ranked)
        ]

        logger.info(
            "OntologyPath.retrieve: %d candidates evaluated, "
            "%d rescued (top_k=%d, bfs_depth=%d, %.1f ms)",
            len(candidate_skills_map) - len(exclude_ids or set()),
            len(results),
            top_k,
            bfs_depth,
            elapsed_ms,
        )

        return results

    # ------------------------------------------------------------------ #
    # Single-candidate scoring (testing / debugging)                      #
    # ------------------------------------------------------------------ #

    def score_single(
        self,
        candidate_skills: frozenset[str],
        jd_intent: JDIntent,
        bfs_depth: int = 1,
    ) -> float:
        """
        Score a single candidate's skill set against the JD.

        Useful for:
          - Unit tests verifying specific candidates are rescued
          - Debugging why a candidate was or was not retrieved
          - Smoke tests validating the domain-transfer graph

        Args:
            candidate_skills: frozenset of lowercase skill names.
                              Typically CandidateFeatureVector.skill_names_lower.
            jd_intent:        Parsed JDIntent (uses required_skills).
            bfs_depth:        BFS depth for rescue map. Default 1.

        Returns:
            Float in [0.0, 1.0]. 0.0 means no domain-transfer alignment.
        """
        if not jd_intent.required_skills:
            return 0.0
        rescue_map = self._graph.build_jd_rescue_map(
            jd_intent.required_skills, bfs_depth=bfs_depth
        )
        return self._graph.score_candidate_skills(candidate_skills, rescue_map)

    # ------------------------------------------------------------------ #
    # Static helper — used by pipeline/runner.py                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_skills_map(
        feature_vectors: list[CandidateFeatureVector],
    ) -> dict[str, frozenset[str]]:
        """
        Build the candidate_skills_map from parsed CandidateFeatureVector objects.

        This is the expected integration point in pipeline/runner.py:

            all_fvecs = candidate_parser.parse_all(candidates_path)
            skills_map = OntologyPath.build_skills_map(all_fvecs)
            results = ontology_path.retrieve(jd_intent, skills_map)

        Args:
            feature_vectors: All CandidateFeatureVector objects loaded for
                             this ranking run (100 K at full scale).

        Returns:
            {candidate_id: frozenset[skill_names_lower]} — skill_names_lower
            is pre-built on CandidateFeatureVector for O(1) lookup.
        """
        return {
            fv.candidate_id: fv.skill_names_lower
            for fv in feature_vectors
        }

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    def explain_candidate(
        self,
        candidate_id: str,
        candidate_skills: frozenset[str],
        jd_intent: JDIntent,
    ) -> dict[str, object]:
        """
        Return a human-readable explanation of why a candidate was (or was not)
        rescued by the ontology path.

        Used by trust/advocate.py and ui/components/score_breakdown.py to
        surface domain-transfer matches in the recruiter-facing UI.

        Returns:
            dict with keys:
                "score":            float domain-transfer score
                "matched_via":      list[str] source skills that transferred in
                "covered_jd_skills":list[str] JD required skills covered
                "rescue_sources":   dict[str, list[str]] full rescue map
        """
        if not jd_intent.required_skills:
            return {
                "score": 0.0,
                "matched_via": [],
                "covered_jd_skills": [],
                "rescue_sources": {},
            }

        rescue_map = self._graph.build_jd_rescue_map(jd_intent.required_skills)
        score = self._graph.score_candidate_skills(candidate_skills, rescue_map)

        matched_via: list[str] = []
        covered_jd_skills: list[str] = []

        for jd_skill, sources in rescue_map.items():
            if jd_skill in candidate_skills:
                covered_jd_skills.append(jd_skill)
                matched_via.append(f"{jd_skill} (direct)")
            else:
                hits = candidate_skills & sources
                if hits:
                    covered_jd_skills.append(jd_skill)
                    for h in sorted(hits):
                        matched_via.append(f"{h} → {jd_skill} (transfer)")

        return {
            "score": score,
            "matched_via": matched_via,
            "covered_jd_skills": covered_jd_skills,
            "rescue_sources": {k: sorted(v) for k, v in rescue_map.items() if v},
        }

    def __repr__(self) -> str:
        return f"OntologyPath(graph={self._graph!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_ontology(
    jd_intent: JDIntent,
    candidate_skills_map: dict[str, frozenset[str]],
    top_k: int = config.ONTOLOGY_PATH_TOP_K,
    skill_map_path: Optional[Path] = None,
) -> list[RetrievalResult]:
    """
    One-shot convenience: build OntologyPath and retrieve top-K candidates.

    Creates a new OntologyPath (and SkillGraph) on each call.
    For repeated calls, use OntologyPath.from_skill_map() once and reuse.

    Args:
        jd_intent:            Parsed JDIntent with required_skills populated.
        candidate_skills_map: {candidate_id: frozenset[skill_names_lower]}.
        top_k:                Number of results to return.
        skill_map_path:       Override for config.SKILL_MAP_PATH.

    Returns:
        list[RetrievalResult] sorted by domain-transfer score descending.
    """
    path = OntologyPath.from_skill_map(skill_map_path=skill_map_path)
    return path.retrieve(jd_intent, candidate_skills_map, top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test — python -m retrieval.ontology_path
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    print("=" * 65)
    print("OntologyPath — smoke test: Tier-5 RecSys rescue")
    print("=" * 65)

    # ── Build path from skill_map.json ────────────────────────────────────
    try:
        path = OntologyPath.from_skill_map()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print(f"\nBuilt: {path}\n")

    # ── Synthetic candidate pool ──────────────────────────────────────────
    # CAND_0000031-analog: RecSys engineer — no RAG/FAISS/IR keywords at all
    TIER5_ID = "CAND_0000031"
    IRRELEVANT_ID = "CAND_0000002"

    candidate_pool: dict[str, frozenset[str]] = {
        # Tier-5 target: RecSys engineer, product company
        # Has ZERO JD keywords (no "information retrieval", "faiss", "rag")
        TIER5_ID: frozenset({
            "recommendation systems", "collaborative filtering",
            "xgboost", "feature engineering", "a/b testing",
            "python", "spark",
        }),
        # Strong direct match: has JD-required IR/embedding skills
        "CAND_0000043": frozenset({
            "elasticsearch", "opensearch", "embeddings", "information retrieval",
            "haystack", "python", "faiss",
        }),
        # NLP engineer — partial domain transfer
        "CAND_0000015": frozenset({
            "nlp", "natural language processing", "hugging face transformers",
            "pytorch", "python", "text classification",
        }),
        # Search engineer — strong domain transfer
        "CAND_0000038": frozenset({
            "search engineering", "ranking", "python",
            "elasticsearch", "feature engineering",
        }),
        # Irrelevant: marketing / operations / consulting
        IRRELEVANT_ID: frozenset({
            "marketing", "operations", "excel", "project management",
            "tcs", "infosys", "customer support",
        }),
        # Accountant — completely off-domain
        "CAND_0000005": frozenset({
            "accounting", "tally", "excel", "powerpoint",
        }),
    }

    # ── Build mock JDIntent ───────────────────────────────────────────────
    mock_jd = JDIntent(
        required_skills=[
            "information retrieval", "ranking", "embeddings",
            "evaluation framework", "python",
        ],
        nice_to_have_skills=["mlops", "learning to rank"],
        disqualifier_skills=["computer vision"],
        expanded_required=[],
        yoe_min=4.0, yoe_max=12.0, yoe_ideal_min=5.0, yoe_ideal_max=9.0,
        preferred_locations=["noida", "pune"],
        relocation_accepted=True,
        disqualify_consulting_only=True,
        disqualify_no_production=True,
        embedding=None,
        raw_text="Senior AI Engineer. Requires IR, ranking, embeddings.",
    )

    # ── Run retrieve() ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    results = path.retrieve(mock_jd, candidate_pool, top_k=config.ONTOLOGY_PATH_TOP_K)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Retrieved {len(results)} results in {elapsed_ms:.1f} ms\n")

    result_ids = [r.candidate_id for r in results]
    for r in results:
        tag = ""
        if r.candidate_id == TIER5_ID:
            tag = "  ← Tier-5 RecSys rescue (no RAG/FAISS keywords)"
        elif r.candidate_id == IRRELEVANT_ID:
            tag = "  ← SHOULD NOT APPEAR"
        print(
            f"  rank={r.rank_in_path}  {r.candidate_id}"
            f"  score={r.path_score:.3f}{tag}"
        )

    # ── Acceptance criterion 1: Tier-5 RecSys candidate rescued ──────────
    assert TIER5_ID in result_ids, (
        f"FAIL: Tier-5 candidate {TIER5_ID} (RecSys, no IR keywords) "
        f"was NOT rescued. Results: {result_ids}.\n"
        "Check that 'recommendation systems' → 'information retrieval' "
        "exists in skill_map.json domain_transfers."
    )
    print(f"\n[PASS] Tier-5 RecSys candidate {TIER5_ID} rescued  ✓")

    # ── Acceptance criterion 2: irrelevant candidate not in results ───────
    assert IRRELEVANT_ID not in result_ids, (
        f"FAIL: Irrelevant candidate {IRRELEVANT_ID} "
        f"(marketing/ops) appeared in results: {result_ids}"
    )
    print(f"[PASS] Irrelevant candidate {IRRELEVANT_ID} excluded  ✓")

    # ── Acceptance criterion 3: scores sorted descending ─────────────────
    for i in range(len(results) - 1):
        assert results[i].path_score >= results[i + 1].path_score, (
            f"FAIL: scores not sorted descending at index {i}: "
            f"{results[i].path_score} < {results[i + 1].path_score}"
        )
    print("[PASS] Scores sorted descending  ✓")

    # ── Acceptance criterion 4: path_name = "ontology" ───────────────────
    assert all(r.path_name == OntologyPath.PATH_NAME for r in results)
    print("[PASS] All results have path_name='ontology'  ✓")

    # ── Acceptance criterion 5: rank_in_path sequential from 1 ───────────
    assert [r.rank_in_path for r in results] == list(range(1, len(results) + 1))
    print("[PASS] rank_in_path is 1-indexed sequential  ✓")

    # ── Acceptance criterion 6: scores in (0, 1] ─────────────────────────
    assert all(0.0 < r.path_score <= 1.0 for r in results), (
        f"FAIL: score out of (0, 1] range: "
        f"{[r.path_score for r in results]}"
    )
    print("[PASS] All path_scores in (0.0, 1.0]  ✓")

    # ── Acceptance criterion 7: score_single matches pool score ──────────
    single = path.score_single(candidate_pool[TIER5_ID], mock_jd)
    tier5_result = next(r for r in results if r.candidate_id == TIER5_ID)
    assert abs(single - tier5_result.path_score) < 1e-6, (
        f"FAIL: score_single={single:.6f} != "
        f"pool score={tier5_result.path_score:.6f}"
    )
    print("[PASS] score_single() is consistent with retrieve()  ✓")

    # ── Acceptance criterion 8: explain_candidate output ─────────────────
    explanation = path.explain_candidate(TIER5_ID, candidate_pool[TIER5_ID], mock_jd)
    assert explanation["score"] > 0.0
    assert len(explanation["matched_via"]) > 0
    assert len(explanation["covered_jd_skills"]) > 0
    print(f"[PASS] explain_candidate: {explanation['matched_via'][:2]}  ✓")

    # ── Acceptance criterion 9: build_skills_map helper ──────────────────
    # Simulate what runner.py does — not a real CandidateFeatureVector,
    # but check the static method signature with a mock object.
    import dataclasses  # noqa: PLC0415

    @dataclasses.dataclass
    class _MockFV:
        candidate_id: str
        skill_names_lower: frozenset

    mock_fvs = [
        _MockFV(cid, skills) for cid, skills in candidate_pool.items()
    ]
    built_map = OntologyPath.build_skills_map(mock_fvs)  # type: ignore[arg-type]
    assert built_map == candidate_pool
    print("[PASS] build_skills_map() produces correct dict  ✓")

    # ── Acceptance criterion 10: empty pool returns [] ────────────────────
    empty_results = path.retrieve(mock_jd, {})
    assert empty_results == []
    print("[PASS] Empty candidate pool returns []  ✓")

    print("\nAll smoke-test assertions passed.")
#--Test END-->