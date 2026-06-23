"""
trust/skeptic.py — Skeptic agent for the adversarial trust layer.

ROLE
----
The Skeptic independently argues *against* the ranking.  It scans every
available signal for evidence that the candidate is a weaker fit than their
composite score suggests — availability problems, career red flags, missing
required skills, profile sparsity, domain disqualifiers.

The Skeptic does not try to be fair.  Its job is to find problems.
verdict.py weighs the Skeptic's output against the Advocate's output to
produce a balanced, calibrated final assessment.

CONTRACT
--------
Input:
  candidate      : CandidateFeatureVector    — normalised candidate record
  scores         : ComponentScores           — pre-computed scoring breakdown
  career_result  : CareerQualityResult       — career scorer breakdown
  behavioral_result: BehavioralResult        — behavioral scorer breakdown
  skill_result   : SkillMatchResult          — skill scorer breakdown
  jd             : JDIntent                  — structured job description

Output:
  list[SkepticSignal]  — each signal has a label, severity, and value.
                         Severity is "HIGH" | "MODERATE" | "LOW".
                         List is ordered: HIGH first, then MODERATE, then LOW.
                         Empty list is valid (very strong, low-risk candidates).

SEVERITY RULES (from config.py thresholds)
-------------------------------------------
  HIGH     : Hard disqualifier or clear red flag from the JD
             (consulting-only, inactivity > 90d, response rate < 0.15,
              hard skill disqualifier, zero product-company experience).
  MODERATE : Meaningful concern but not disqualifying alone
             (notice > 60d, low stability, soft skill disqualifier,
              YOE out of band, missing ≥40% required skills).
  LOW      : Weak signal worth noting but unlikely to flip a ranking
             (no GitHub, low profile completeness, poor domain match).

RISK CATALOGUE (14 checks)
---------------------------
  1.  Inactivity                — days since last platform activity
  2.  Low recruiter response    — historical recruiter response rate
  3.  Long notice period        — days of notice required
  4.  Consulting-only career    — ALL companies are consulting firms (JD disqualifier)
  5.  No product-co experience  — zero time at product companies
  6.  Hard skill disqualifier   — domain-wrong skills at high proficiency
  7.  Soft skill disqualifier   — domain-adjacent wrong skills at low proficiency
  8.  Missing required skills   — required skill coverage below thresholds
  9.  YOE out of band           — years of experience outside JD target
  10. Low career stability       — job-hopping signal from stability_score
  11. Sparse profile             — too few signal types for confident scoring
  12. Profile incompleteness     — raw completeness score below threshold
  13. No GitHub linked           — open-source signal missing
  14. Poor domain match          — career history outside relevant industries

DESIGN NOTES
------------
- All claimed facts in output MUST trace to input data.  The value field
  carries the specific number/company/count — reasoning_generator.py reads
  it verbatim.
- Deterministic: same input → same output, same order.
- The function does NOT read ComponentScores.skill_score etc. for risk checks.
  It reads the richer breakdowns (CareerQualityResult, BehavioralResult,
  SkillMatchResult) to produce factual, specific value strings.
- Parallel structure to advocate.py: one private _scan_* function per risk,
  same _add_optional/_sort_signals pattern, same public entry point.

DEPENDENCIES
------------
  config                    : threshold constants
  pipeline.schemas          : CandidateFeatureVector, ComponentScores,
                              SkepticSignal, JDIntent
  scoring.career_quality    : CareerQualityResult
  scoring.behavioral        : BehavioralResult
  scoring.skill_match       : SkillMatchResult

No I/O.  No network.  No side-effects.  Pure function.
"""

from __future__ import annotations

import collections
import datetime
import logging
from typing import Optional

import config
from pipeline.schemas import (
    CandidateFeatureVector,
    ComponentScores,
    JDIntent,
    SkepticSignal,
)
from scoring.behavioral import BehavioralResult
from scoring.career_quality import CareerQualityResult
from scoring.skill_match import SkillMatchResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_HIGH: str = "HIGH"
_MOD: str = "MODERATE"
_LOW: str = "LOW"

# ─────────────────────────────────────────────────────────────────────────────
# RISK THRESHOLDS (sourced from config — never hardcode)
# ─────────────────────────────────────────────────────────────────────────────

# Inactivity
_INACTIVITY_HIGH_DAYS: int = config.SKEPTIC_HIGH_RISK_INACTIVITY_DAYS        # 90
_INACTIVITY_MOD_DAYS: int = 30   # 30–90 days → MODERATE concern

# Recruiter response rate
_RESPONSE_HIGH_RISK: float = config.SKEPTIC_HIGH_RISK_RESPONSE_RATE          # 0.15
_RESPONSE_MOD_RISK: float = 0.40  # 0.15–0.40 → MODERATE concern

# Notice period
_NOTICE_HIGH_DAYS: int = config.NOTICE_PERIOD_MAX                             # 90
_NOTICE_MOD_DAYS: int = config.SKEPTIC_MODERATE_NOTICE_DAYS                  # 60

# Required skill coverage
_SKILL_COVERAGE_HIGH_RISK: float = 0.30   # < 30% required → HIGH
_SKILL_COVERAGE_MOD_RISK: float = 0.60    # < 60% required → MODERATE

# Career stability
_STABILITY_MOD_RISK: float = 0.40   # stability_score < 0.40 → MODERATE

# No product-company experience
_PRODUCT_CO_HIGH_RISK: float = 0.0    # exactly zero → HIGH
_PRODUCT_CO_MOD_RISK: float = 0.30   # < 30% product-co time → MODERATE

# Profile incompleteness
_COMPLETENESS_LOW_RISK: float = 50.0  # raw score 0–100; < 50 → LOW signal

# Domain match
_DOMAIN_MOD_RISK: float = 0.20   # domain_match_score < 0.20 → LOW signal

# Sparse profile: signal count below this → MODERATE sparsity flag
_SPARSE_SIGNAL_COUNT: int = 4


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sort_signals(signals: list[SkepticSignal]) -> list[SkepticSignal]:
    """
    Sort signals HIGH → MODERATE → LOW, stable within each tier.

    verdict.py expects the most critical risks first so it can classify
    ROBUST / CONTESTED / FRAGILE by scanning from the top.
    """
    _order = {_HIGH: 0, _MOD: 1, _LOW: 2}
    return sorted(signals, key=lambda s: _order.get(s.severity, 3))


def _add_optional(
    target: list[SkepticSignal],
    signal: Optional[SkepticSignal],
) -> None:
    """Append signal to target only if it is not None."""
    if signal is not None:
        target.append(signal)


def _days_to_human(days: int) -> str:
    """Convert a day count to a human-readable string for value fields."""
    if days == 0:
        return "today"
    if days == 1:
        return "1 day ago"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        weeks = days // 7
        return f"~{weeks} week(s) ago"
    months = days // 30
    return f"~{months} month(s) ago"


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL RISK SCANNERS
# Each returns Optional[SkepticSignal].  None means no risk detected.
# Each function is narrow and independently unit-testable.
# ─────────────────────────────────────────────────────────────────────────────

def _scan_inactivity(
    candidate: CandidateFeatureVector,
) -> Optional[SkepticSignal]:
    """
    Risk 1: Platform inactivity.

    A candidate who hasn't been active for 90+ days is, for hiring purposes,
    probably not in the market regardless of how strong their profile is.
    The JD explicitly says to down-weight unavailable candidates.

    Threshold sources: config.SKEPTIC_HIGH_RISK_INACTIVITY_DAYS (90).
    MODERATE band: 30–89 days (meaningful lag but not disqualifying).
    """
    days = candidate.signals.days_since_active

    if days > _INACTIVITY_HIGH_DAYS:
        return SkepticSignal(
            label="Platform inactivity",
            severity=_HIGH,
            value=f"Last active {_days_to_human(days)} ({days} days) — well above 90-day risk threshold",
        )
    if days > _INACTIVITY_MOD_DAYS:
        return SkepticSignal(
            label="Platform inactivity",
            severity=_MOD,
            value=f"Last active {_days_to_human(days)} ({days} days) — some inactivity lag",
        )
    return None


def _scan_response_rate(
    candidate: CandidateFeatureVector,
) -> Optional[SkepticSignal]:
    """
    Risk 2: Low recruiter response rate.

    A candidate who historically ignores recruiter outreach will likely do
    so again.  Low response rate makes the best-on-paper candidate effectively
    unreachable.

    Thresholds: config.SKEPTIC_HIGH_RISK_RESPONSE_RATE (0.15).
    MODERATE band: 0.15–0.40 (engages occasionally but unreliable).
    """
    rate = candidate.signals.recruiter_response_rate

    if rate < _RESPONSE_HIGH_RISK:
        return SkepticSignal(
            label="Low recruiter response rate",
            severity=_HIGH,
            value=(
                f"{rate:.0%} historical response rate to recruiter outreach "
                f"— below {_RESPONSE_HIGH_RISK:.0%} HIGH-risk threshold"
            ),
        )
    if rate < _RESPONSE_MOD_RISK:
        return SkepticSignal(
            label="Low recruiter response rate",
            severity=_MOD,
            value=(
                f"{rate:.0%} historical response rate "
                f"— moderate engagement risk"
            ),
        )
    return None


def _scan_notice_period(
    candidate: CandidateFeatureVector,
) -> Optional[SkepticSignal]:
    """
    Risk 3: Long notice period.

    The JD says "we'd love sub-30-day notice; we can buy out up to 30 days.
    30+ day notice candidates are still in scope but the bar gets higher."
    > 90 days is a meaningful operational risk, not just a preference.

    We compute the earliest possible start date so the LLM has a concrete,
    quotable fact rather than just a threshold reference.

    Thresholds: config.NOTICE_PERIOD_MAX (90), config.SKEPTIC_MODERATE_NOTICE_DAYS (60).
    """
    days = candidate.signals.notice_period_days

    # Compute earliest start date as a human-readable month string.
    today = datetime.date.today()
    earliest_start = today + datetime.timedelta(days=days)
    start_str = earliest_start.strftime("%b %Y")  # e.g. "Sep 2025"

    if days > _NOTICE_HIGH_DAYS:
        return SkepticSignal(
            label="Long notice period",
            severity=_HIGH,
            value=(
                f"{days}-day notice — earliest start: ~{start_str}; "
                f"JD states notice >90 days significantly raises the hiring bar"
            ),
        )
    if days > _NOTICE_MOD_DAYS:
        return SkepticSignal(
            label="Long notice period",
            severity=_MOD,
            value=(
                f"{days}-day notice — earliest start: ~{start_str}; "
                f"above JD preferred ≤30 days (confirm buyout feasibility)"
            ),
        )
    return None


def _scan_consulting_only(
    career_result: CareerQualityResult,
    candidate: CandidateFeatureVector,
) -> Optional[SkepticSignal]:
    """
    Risk 4: Consulting-only career background.

    The JD is explicit: "People who have only worked at consulting firms
    (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) —
    we will not move forward."  This is always HIGH severity when triggered.

    We name the specific firms found so the recruiter can verify.
    """
    if not career_result.is_consulting_only:
        return None

    # Collect the consulting firm names from career history.
    consulting_companies: list[str] = sorted({
        entry.company
        for entry in candidate.career_history
        if entry.company_lower in config.CONSULTING_FIRMS
    })

    companies_str = ", ".join(consulting_companies[:4])
    if len(consulting_companies) > 4:
        companies_str += f" (+{len(consulting_companies) - 4} more)"

    return SkepticSignal(
        label="Consulting-only career background",
        severity=_HIGH,
        value=(
            f"All career history at consulting firms: {companies_str} — "
            f"JD explicitly disqualifies consulting-only backgrounds"
        ),
    )


def _scan_no_product_experience(
    career_result: CareerQualityResult,
) -> Optional[SkepticSignal]:
    """
    Risk 5: No meaningful product-company experience.

    The JD rewards product-company experience (food-tech, fintech, SaaS,
    AI/ML) and penalises pure services careers.  Zero product-co time is
    HIGH; very low product-co fraction is MODERATE.

    Note: this check fires independently of _scan_consulting_only.
    A candidate can be non-consulting but still have zero product-co time
    (e.g. defence/government sector career).
    """
    score = career_result.product_co_score

    if score <= _PRODUCT_CO_HIGH_RISK:
        return SkepticSignal(
            label="No product-company experience",
            severity=_HIGH,
            value=(
                "0% of career time at product companies — "
                "JD weights product-company experience heavily"
            ),
        )
    if score < _PRODUCT_CO_MOD_RISK:
        return SkepticSignal(
            label="Limited product-company experience",
            severity=_MOD,
            value=(
                f"{score:.0%} of career time at product companies — "
                f"below {_PRODUCT_CO_MOD_RISK:.0%} moderate threshold"
            ),
        )
    return None


def _scan_hard_disqualifier(
    skill_result: SkillMatchResult,
) -> Optional[SkepticSignal]:
    """
    Risk 6: Hard skill disqualifier.

    The JD disqualifies candidates whose primary expertise is in the wrong
    domain (computer vision, speech, robotics) at high proficiency levels.
    skill_match.py's _check_disqualifiers() handles the detection;
    we surface it here as a HIGH risk with the specific matched skills.
    """
    if not skill_result.hard_disqualifier:
        return None

    disq_skills = ", ".join(skill_result.matched_disqualifiers[:5])

    return SkepticSignal(
        label="Hard domain disqualifier",
        severity=_HIGH,
        value=(
            f"Expert/advanced proficiency in out-of-domain skills: {disq_skills} — "
            f"JD explicitly disqualifies CV/speech/robotics-primary backgrounds"
        ),
    )


def _scan_soft_disqualifier(
    skill_result: SkillMatchResult,
) -> Optional[SkepticSignal]:
    """
    Risk 7: Soft skill disqualifier.

    Same domain concern as Risk 6 but at lower proficiency (beginner/
    intermediate).  Still worth flagging because it may indicate the
    candidate is pivoting from a wrong domain, not a fit candidate who
    happens to have peripheral exposure.
    """
    if not skill_result.soft_disqualifier:
        return None
    # Don't double-fire if hard disqualifier already fired on same skills.
    if skill_result.hard_disqualifier:
        return None

    disq_skills = ", ".join(skill_result.matched_disqualifiers[:5])

    return SkepticSignal(
        label="Soft domain disqualifier",
        severity=_MOD,
        value=(
            f"Low-proficiency exposure to out-of-domain skills: {disq_skills} — "
            f"may indicate domain pivot rather than IR/NLP specialisation"
        ),
    )


def _scan_missing_required_skills(
    skill_result: SkillMatchResult,
    jd: JDIntent,
) -> Optional[SkepticSignal]:
    """
    Risk 8: Missing required skills.

    Uses required_score from SkillMatchResult (cluster-weighted coverage)
    rather than a raw count, because the JD weights retrieval/ranking
    clusters much more heavily than LLM clusters.

    We also compute the explicit list of unmatched required skills to give
    the recruiter a concrete list to probe in interviews.
    """
    coverage = skill_result.required_score
    matched_set: frozenset[str] = frozenset(
        s.lower() for s in skill_result.matched_required
    )
    unmatched: list[str] = [
        s for s in jd.required_skills
        if s.lower() not in matched_set
    ]

    if coverage < _SKILL_COVERAGE_HIGH_RISK:
        unmatched_str = ", ".join(unmatched[:6])
        if len(unmatched) > 6:
            unmatched_str += f" (+{len(unmatched) - 6} more)"
        return SkepticSignal(
            label="Missing required skills",
            severity=_HIGH,
            value=(
                f"Required skill coverage: {coverage:.0%} "
                f"(below {_SKILL_COVERAGE_HIGH_RISK:.0%} threshold). "
                f"Unmatched: {unmatched_str}"
            ),
        )

    if coverage < _SKILL_COVERAGE_MOD_RISK:
        unmatched_str = ", ".join(unmatched[:4])
        if len(unmatched) > 4:
            unmatched_str += f" (+{len(unmatched) - 4} more)"
        return SkepticSignal(
            label="Partial required skill coverage",
            severity=_MOD,
            value=(
                f"Required skill coverage: {coverage:.0%} "
                f"(below {_SKILL_COVERAGE_MOD_RISK:.0%} threshold). "
                f"Gaps: {unmatched_str}"
                if unmatched_str else
                f"Required skill coverage: {coverage:.0%} "
                f"— partial match, some capability clusters weak"
            ),
        )

    return None


def _scan_yoe_out_of_band(
    candidate: CandidateFeatureVector,
    career_result: CareerQualityResult,
) -> Optional[SkepticSignal]:
    """
    Risk 9: Years of experience outside the JD band.

    The JD targets 5–9 years.  Under-banded candidates may lack production
    depth; over-banded candidates may be overqualified or title-chasers.
    We only fire when yoe_score is meaningfully low (< 0.40) because mild
    out-of-band is penalised by the score already.

    Uses config.YOE_BAND_MIN / YOE_BAND_MAX (the soft outer limits).
    """
    yoe = candidate.years_of_experience
    yoe_score = career_result.yoe_score

    # Only fire when the score is meaningfully low.
    if yoe_score >= 0.40:
        return None

    ideal_min = config.YOE_BAND_IDEAL_MIN   # 5.0
    ideal_max = config.YOE_BAND_IDEAL_MAX   # 9.0
    band_min = config.YOE_BAND_MIN          # 4.0
    band_max = config.YOE_BAND_MAX          # 12.0

    if yoe < band_min:
        return SkepticSignal(
            label="Under-experienced for role",
            severity=_MOD,
            value=(
                f"{yoe:.1f} years of experience — below soft floor of "
                f"{band_min:.0f} years (JD target: {ideal_min:.0f}–{ideal_max:.0f})"
            ),
        )
    if yoe > band_max:
        return SkepticSignal(
            label="Over-experienced — possible title-chaser risk",
            severity=_LOW,
            value=(
                f"{yoe:.1f} years of experience — above soft ceiling of "
                f"{band_max:.0f} years; JD explicitly flags title-chasers"
            ),
        )

    # In-band but still low yoe_score — mild concern.
    return SkepticSignal(
        label="YOE band fit weak",
        severity=_LOW,
        value=(
            f"{yoe:.1f} years — within soft band but YOE score low "
            f"({yoe_score:.2f}); check adjacent experience quality"
        ),
    )


def _scan_job_hopping(
    career_result: CareerQualityResult,
    candidate: CandidateFeatureVector,
) -> Optional[SkepticSignal]:
    """
    Risk 10: Low career stability / job-hopping.

    stability_score in CareerQualityResult encodes average tenure length
    relative to the 3-year cap, with an additional penalty for avg tenure
    < 1.5 years.  The JD says "we need someone who plans to be here for 3+
    years" — job-hopping history is a direct signal against that.
    """
    stability = career_result.stability_score

    if stability >= _STABILITY_MOD_RISK:
        return None

    # Count roles to give a concrete fact.
    total_roles = len(candidate.career_history)
    yoe = max(candidate.years_of_experience, 1.0)
    avg_tenure_years = yoe / total_roles if total_roles > 0 else yoe

    return SkepticSignal(
        label="Job-hopping pattern",
        severity=_MOD,
        value=(
            f"Stability score {stability:.2f} (below {_STABILITY_MOD_RISK:.2f} threshold) — "
            f"~{avg_tenure_years:.1f} avg years per role across {total_roles} roles; "
            f"JD requires 3+ year commitment"
        ),
    )


def _scan_sparse_profile(
    behavioral_result: BehavioralResult,
    scores: ComponentScores,
) -> Optional[SkepticSignal]:
    """
    Risk 11: Sparse profile — too few signal types for confident scoring.

    When signal_count < _SPARSE_SIGNAL_COUNT the composite score carries an
    uncertainty_penalty multiplier.  We surface this so the recruiter knows
    the score is less reliable than usual and warrants extra screening.
    """
    signal_count = behavioral_result.signal_count
    penalty = scores.uncertainty_penalty

    if signal_count >= _SPARSE_SIGNAL_COUNT:
        return None

    return SkepticSignal(
        label="Sparse profile — low scoring confidence",
        severity=_MOD,
        value=(
            f"Only {signal_count} of {config.MIN_SIGNAL_TYPES_FOR_FULL_CONFIDENCE} "
            f"expected signal types populated — "
            f"composite score carries {penalty:.0%} confidence multiplier"
        ),
    )


def _scan_profile_incompleteness(
    candidate: CandidateFeatureVector,
) -> Optional[SkepticSignal]:
    """
    Risk 12: Low profile completeness.

    Raw completeness < 50% is a LOW signal — it doesn't disqualify the
    candidate but suggests their profile may not represent them well.
    Recruiters should ask the candidate to complete their profile before
    final decision.
    """
    completeness = candidate.signals.profile_completeness_score

    if completeness >= _COMPLETENESS_LOW_RISK:
        return None

    return SkepticSignal(
        label="Incomplete profile",
        severity=_LOW,
        value=(
            f"Profile completeness: {completeness:.0f}% — "
            f"below {_COMPLETENESS_LOW_RISK:.0f}% threshold; "
            f"scoring reliability may be reduced"
        ),
    )


def _scan_no_github(
    candidate: CandidateFeatureVector,
) -> Optional[SkepticSignal]:
    """
    Risk 13: No GitHub linked.

    The JD notes open-source contributions as a nice-to-have and says
    "people whose work has been entirely on closed-source proprietary systems
    for 5+ years without external validation (papers, talks, open-source) —
    we need to see how you think."  Missing GitHub is a LOW flag, not
    disqualifying, but worth noting for senior-level roles.
    """
    if candidate.signals.has_github:
        return None

    return SkepticSignal(
        label="No GitHub linked",
        severity=_LOW,
        value=(
            "GitHub not connected to Redrob profile — "
            "JD notes open-source / external validation as important signal"
        ),
    )


def _scan_domain_mismatch(
    career_result: CareerQualityResult,
) -> Optional[SkepticSignal]:
    """
    Risk 14: Poor domain match.

    domain_match_score is the fraction of career time in relevant industries
    minus a penalty for time in explicitly wrong industries.
    Low domain match means the candidate has spent most of their career
    outside AI/ML, fintech, SaaS, e-commerce — domains where this role's
    skills transfer.
    """
    domain_score = career_result.domain_match_score

    if domain_score >= _DOMAIN_MOD_RISK:
        return None

    return SkepticSignal(
        label="Limited domain relevance",
        severity=_LOW,
        value=(
            f"Domain match score: {domain_score:.2f} — "
            f"most career time outside AI/ML-adjacent industries; "
            f"check depth of applied ML context"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_skeptic_signals(
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
    career_result: CareerQualityResult,
    behavioral_result: BehavioralResult,
    skill_result: SkillMatchResult,
    jd: JDIntent,
) -> list[SkepticSignal]:
    """
    Build the complete list of risk signals for one candidate.

    This is the sole public entry point for trust/skeptic.py.
    Calls all individual risk scanners, sorts by severity, and returns a
    clean list ready for verdict.py consumption.

    Parameters
    ----------
    candidate        : Fully parsed CandidateFeatureVector.
    scores           : ComponentScores breakdown from scoring/composite.py.
    career_result    : CareerQualityResult from scoring/career_quality.py.
    behavioral_result: BehavioralResult from scoring/behavioral.py.
    skill_result     : SkillMatchResult from scoring/skill_match.py.
    jd               : Structured JD intent from pipeline/jd_parser.py.

    Returns
    -------
    list[SkepticSignal], sorted HIGH → MODERATE → LOW.
    Empty list if no risks detected (valid for strong, available candidates).

    Raises
    ------
    TypeError   : If any argument is not of the expected type.
    ValueError  : If candidate_id is inconsistent across inputs.
    """
    # ── Type guards (fail-fast) ───────────────────────────────────────────────
    if not isinstance(candidate, CandidateFeatureVector):
        raise TypeError(
            f"candidate must be CandidateFeatureVector, got {type(candidate).__name__}"
        )
    if not isinstance(scores, ComponentScores):
        raise TypeError(
            f"scores must be ComponentScores, got {type(scores).__name__}"
        )
    if not isinstance(career_result, CareerQualityResult):
        raise TypeError(
            f"career_result must be CareerQualityResult, got {type(career_result).__name__}"
        )
    if not isinstance(behavioral_result, BehavioralResult):
        raise TypeError(
            f"behavioral_result must be BehavioralResult, "
            f"got {type(behavioral_result).__name__}"
        )
    if not isinstance(skill_result, SkillMatchResult):
        raise TypeError(
            f"skill_result must be SkillMatchResult, got {type(skill_result).__name__}"
        )
    if not isinstance(jd, JDIntent):
        raise TypeError(f"jd must be JDIntent, got {type(jd).__name__}")

    # ── ID consistency guard ──────────────────────────────────────────────────
    cid = candidate.candidate_id
    for label, other_id in [
        ("scores", scores.candidate_id),
        ("career_result", career_result.candidate_id),
        ("behavioral_result", behavioral_result.candidate_id),
        ("skill_result", skill_result.candidate_id),
    ]:
        if other_id != cid:
            raise ValueError(
                f"ID mismatch: candidate.candidate_id={cid!r} "
                f"but {label}.candidate_id={other_id!r}"
            )

    signals: list[SkepticSignal] = []

    # ── Risk 1: Inactivity ────────────────────────────────────────────────────
    _add_optional(signals, _scan_inactivity(candidate))

    # ── Risk 2: Low recruiter response rate ───────────────────────────────────
    _add_optional(signals, _scan_response_rate(candidate))

    # ── Risk 3: Long notice period ────────────────────────────────────────────
    _add_optional(signals, _scan_notice_period(candidate))

    # ── Risk 4: Consulting-only background ───────────────────────────────────
    _add_optional(signals, _scan_consulting_only(career_result, candidate))

    # ── Risk 5: No product-company experience ─────────────────────────────────
    _add_optional(signals, _scan_no_product_experience(career_result))

    # ── Risk 6: Hard skill disqualifier ──────────────────────────────────────
    _add_optional(signals, _scan_hard_disqualifier(skill_result))

    # ── Risk 7: Soft skill disqualifier ──────────────────────────────────────
    _add_optional(signals, _scan_soft_disqualifier(skill_result))

    # ── Risk 8: Missing required skills ──────────────────────────────────────
    _add_optional(signals, _scan_missing_required_skills(skill_result, jd))

    # ── Risk 9: YOE out of band ───────────────────────────────────────────────
    _add_optional(signals, _scan_yoe_out_of_band(candidate, career_result))

    # ── Risk 10: Job-hopping ──────────────────────────────────────────────────
    _add_optional(signals, _scan_job_hopping(career_result, candidate))

    # ── Risk 11: Sparse profile ───────────────────────────────────────────────
    _add_optional(signals, _scan_sparse_profile(behavioral_result, scores))

    # ── Risk 12: Profile incompleteness ──────────────────────────────────────
    _add_optional(signals, _scan_profile_incompleteness(candidate))

    # ── Risk 13: No GitHub ────────────────────────────────────────────────────
    _add_optional(signals, _scan_no_github(candidate))

    # ── Risk 14: Domain mismatch ─────────────────────────────────────────────
    _add_optional(signals, _scan_domain_mismatch(career_result))

    # ── Sort and return ───────────────────────────────────────────────────────
    result = _sort_signals(signals)

    logger.debug(
        "skeptic: %s → %d risks (HIGH=%d, MODERATE=%d, LOW=%d)",
        cid,
        len(result),
        sum(1 for s in result if s.severity == _HIGH),
        sum(1 for s in result if s.severity == _MOD),
        sum(1 for s in result if s.severity == _LOW),
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY HELPERS (consumed by verdict.py)
# ─────────────────────────────────────────────────────────────────────────────

def count_by_severity(signals: list[SkepticSignal]) -> dict[str, int]:
    """
    Return a count dict: {"HIGH": n, "MODERATE": n, "LOW": n}.

    Convenience function for verdict.py so it does not filter inline.
    The HIGH count is the primary input to the ROBUST/CONTESTED/FRAGILE
    classification in config.VERDICT_FRAGILE_HIGH_RISK_COUNT.
    """
    counts: collections.Counter[str] = collections.Counter(
        s.severity for s in signals
    )
    return {
        _HIGH: counts[_HIGH],
        _MOD:  counts[_MOD],
        _LOW:  counts[_LOW],
    }


def top_risks(
    signals: list[SkepticSignal],
    n: int = 3,
) -> list[SkepticSignal]:
    """
    Return the top-n highest-severity risks.

    Assumes list is already sorted (build_skeptic_signals guarantees this).
    Used by reasoning_generator.py to build the honest-concern sentence
    and by verdict.py to generate falsifiability conditions.
    """
    return signals[:n]