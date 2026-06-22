from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import config
from indexing.trajectory_builder import TrajectoryAnalyzer
from pipeline.schemas import CandidateFeatureVector, JDIntent

logger = logging.getLogger(__name__)

_W_PRODUCT_CO = 0.20
_W_YOE        = 0.30
_W_STABILITY  = 0.25
_W_DOMAIN     = 0.25

assert abs(_W_PRODUCT_CO + _W_YOE + _W_STABILITY + _W_DOMAIN - 1.0) < 1e-9

_CONSULTING_FIRMS: frozenset[str] = frozenset(
    f.lower().strip() for f in config.CONSULTING_FIRMS
)
_PRODUCT_INDUSTRIES: frozenset[str] = frozenset(
    i.lower().strip() for i in config.PRODUCT_INDUSTRIES
)
_DOMAIN_PENALTY_INDUSTRIES: frozenset[str] = frozenset(
    i.lower().strip()
    for i in getattr(config, "DOMAIN_PENALTY_INDUSTRIES", [])
)

_JOB_HOPPER_AVG_TENURE_YEARS: float = getattr(config, "JOB_HOPPER_AVG_TENURE_YEARS", 1.5)
_JOB_HOPPER_PENALTY: float = getattr(config, "JOB_HOPPER_PENALTY", 0.75)
_STABILITY_CAP_YEARS: float = 3.0


@dataclass(slots=True)
class CareerQualityResult:
    candidate_id:               str
    career_quality_score:       float
    product_co_score:           float
    yoe_score:                  float
    stability_score:            float
    domain_match_score:         float
    is_consulting_only:         bool
    consulting_penalty_applied: bool


class CareerQualityScorer:

    def __init__(self, jd: JDIntent) -> None:
        if not isinstance(jd, JDIntent):
            raise TypeError(f"jd must be JDIntent, got {type(jd).__name__}.")
        self._jd = jd
        self._trajectory = TrajectoryAnalyzer()

    def score(self, candidate: CandidateFeatureVector) -> CareerQualityResult:
        product_co    = self._product_co_score(candidate)
        yoe           = self._yoe_score(candidate)
        stability     = self._stability_score(candidate)
        domain        = self._domain_match_score(candidate)
        is_consulting = candidate.is_consulting_only

        raw = (
            _W_PRODUCT_CO * product_co
            + _W_YOE       * yoe
            + _W_STABILITY * stability
            + _W_DOMAIN    * domain
        )

        consulting_penalty_applied = False
        if is_consulting:
            raw *= config.CONSULTING_ONLY_PENALTY
            consulting_penalty_applied = True

        return CareerQualityResult(
            candidate_id=candidate.candidate_id,
            career_quality_score=round(float(max(0.0, min(1.0, raw))), 6),
            product_co_score=round(product_co, 6),
            yoe_score=round(yoe, 6),
            stability_score=round(stability, 6),
            domain_match_score=round(domain, 6),
            is_consulting_only=is_consulting,
            consulting_penalty_applied=consulting_penalty_applied,
        )

    def score_all(
        self,
        candidates: list[CandidateFeatureVector],
    ) -> dict[str, CareerQualityResult]:
        if not isinstance(candidates, list):
            raise TypeError(
                f"candidates must be list[CandidateFeatureVector], "
                f"got {type(candidates).__name__}."
            )
        t0 = time.perf_counter()
        results = {c.candidate_id: self.score(c) for c in candidates}
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        n_consulting = sum(1 for r in results.values() if r.is_consulting_only)
        mean_score   = (
            sum(r.career_quality_score for r in results.values()) / len(results)
            if results else 0.0
        )
        logger.info(
            "CareerQualityScorer: scored %d candidates in %.1f ms "
            "(consulting_only=%d, mean_score=%.3f).",
            len(results), elapsed_ms, n_consulting, mean_score,
        )
        return results

    def _product_co_score(self, candidate: CandidateFeatureVector) -> float:
        history = candidate.career_history
        if not history:
            return 0.0
        total_months = sum(j.duration_months for j in history)
        if total_months == 0:
            return 0.0
        product_months = sum(
            j.duration_months
            for j in history
            if j.industry_lower in config.PRODUCT_INDUSTRIES
        )
        return float(product_months) / float(total_months)

    def _yoe_score(self, candidate: CandidateFeatureVector) -> float:
        return float(self._trajectory.yoe_score(candidate))

    def _stability_score(self, candidate: CandidateFeatureVector) -> float:
        avg_tenure, _, _ = self._trajectory.calculate_tenure_metrics(candidate)
        score = min(avg_tenure / _STABILITY_CAP_YEARS, 1.0)
        if avg_tenure < _JOB_HOPPER_AVG_TENURE_YEARS:
            score *= _JOB_HOPPER_PENALTY
        return float(max(0.0, score))

    def _domain_match_score(self, candidate: CandidateFeatureVector) -> float:
        history = candidate.career_history
        if not history:
            return 1.0
        total_months = sum(j.duration_months for j in history)
        if total_months == 0:
            return 1.0

        # Penalty-only: pure out-of-domain exposure reduces score from 1.0 baseline.
        # Does not overlap with _product_co_score (which measures product-co time).
        penalty_months = sum(
            j.duration_months for j in history
            if j.industry_lower in _DOMAIN_PENALTY_INDUSTRIES
        )
        if penalty_months == 0:
            return 1.0

        # Weight recent penalty jobs more heavily than old ones.
        # Sort ascending by start_date so index 0 = oldest.
        sorted_history = sorted(history, key=lambda j: j.start_date)
        n = len(sorted_history)
        weighted_penalty = 0.0
        weighted_total = 0.0
        for idx, job in enumerate(sorted_history):
            recency_weight = 1.0 + (idx / n)  # ranges 1.0 (oldest) → ~2.0 (newest)
            weighted_total += job.duration_months * recency_weight
            if job.industry_lower in _DOMAIN_PENALTY_INDUSTRIES:
                weighted_penalty += job.duration_months * recency_weight

        if weighted_total == 0:
            return 1.0

        penalty_ratio = weighted_penalty / weighted_total
        return float(max(0.0, 1.0 - penalty_ratio))
    
    def __repr__(self) -> str:
        return (
            f"CareerQualityScorer("
            f"yoe_ideal=[{self._jd.yoe_ideal_min}, {self._jd.yoe_ideal_max}])"
            )

def score_career_quality(
    candidates: list[CandidateFeatureVector],
    jd: JDIntent,
) -> dict[str, CareerQualityResult]:
    return CareerQualityScorer(jd).score_all(candidates)