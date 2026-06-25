from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

import config
from pipeline.schemas import CandidateFeatureVector, CareerEntry

logger = logging.getLogger(__name__)

_SENIORITY_KEYWORDS: list[tuple[int, tuple[str, ...]]] = [
    (5, ("chief", "ceo", "cto", "cfo", "coo", "vp", "vice president",
         "president", "founder", "co-founder", "director", "head of")),
    (4, ("principal", "manager", "staff")),
    (3, ("senior", "sr.", "sr ", "lead")),
    (1, ("junior", "associate", "entry", "trainee", "apprentice", "intern")),
]
_DEFAULT_LEVEL = 2

# Minimum denominator for promotions/year — guards against divide-by-zero
# and unstable rates for very short (<6 month) career histories.
_MIN_YEARS_FOR_RATE = 0.5


def _seniority_level(title: str) -> int:
    t = title.lower()
    for level, keywords in _SENIORITY_KEYWORDS:
        if any(kw in t for kw in keywords):
            return level
    return _DEFAULT_LEVEL


def count_promotions(career_history: list[CareerEntry]) -> int:
    if len(career_history) < 2:
        return 0

    ordered = sorted(career_history, key=lambda e: e.start_date)
    promotions = 0
    prev_level = _seniority_level(ordered[0].title)
    for entry in ordered[1:]:
        level = _seniority_level(entry.title)
        if level > prev_level:
            promotions += 1
        prev_level = level
    return promotions


def _effective_years(candidate: CandidateFeatureVector) -> float:
    years = candidate.years_of_experience
    if years <= 0:
        years = candidate.total_career_months / 12.0
    return max(years, _MIN_YEARS_FOR_RATE)


def promotions_per_year(candidate: CandidateFeatureVector) -> float:
    return count_promotions(candidate.career_history) / _effective_years(candidate)


def trajectory_velocity_score(rate: float) -> float:
    floor = config.TRAJECTORY_PROMOTIONS_PER_YEAR_FLOOR
    cap = config.TRAJECTORY_PROMOTIONS_PER_YEAR_CAP
    if cap <= floor:
        logger.warning("TRAJECTORY cap <= floor; returning 0.0 for all candidates.")
        return 0.0
    normalized = (rate - floor) / (cap - floor)
    return float(np.clip(normalized, 0.0, 1.0))


@dataclass(frozen=True)
class TrajectoryResult:
    candidate_id: str
    num_promotions: int
    years_of_experience: float
    promotions_per_year: float
    trajectory_velocity: float               # ComponentScores.trajectory_velocity
    percentile_rank: Optional[float] = None  # 0-100, set by score_all()


class TrajectoryVelocityScorer:
    def score(self, candidate: CandidateFeatureVector) -> TrajectoryResult:
        years = _effective_years(candidate)
        n_promo = count_promotions(candidate.career_history)
        rate = n_promo / years
        return TrajectoryResult(
            candidate_id=candidate.candidate_id,
            num_promotions=n_promo,
            years_of_experience=years,
            promotions_per_year=rate,
            trajectory_velocity=trajectory_velocity_score(rate),
        )

    def score_all(self, candidates: list[CandidateFeatureVector]) -> list[TrajectoryResult]:
        results = [self.score(c) for c in candidates]
        if not results:
            return results

        rates = np.array([r.promotions_per_year for r in results], dtype=np.float64)
        return [
            replace(
                r,
                percentile_rank=float((rates <= r.promotions_per_year).mean() * 100.0),
            )
            for r in results
        ]
