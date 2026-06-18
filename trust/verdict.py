"""
trust/verdict.py — Verdict synthesis for the adversarial trust layer.

ROLE
----
verdict.py is the final stage of the trust layer.  It receives the outputs
of advocate.py and skeptic.py, weighs them against each other, and produces
a single TrustVerdict that:

  1. Classifies the ranking as ROBUST, CONTESTED, or FRAGILE.
  2. Assigns a flip_risk level: LOW, MEDIUM, or HIGH.
  3. Computes a confidence_pct (0–100) that reflects how much of the
     composite score is backed by strong, non-contradicted signals.
  4. Generates a falsifiability contract: 2–3 specific, verifiable conditions
     that, if discovered during interviews or reference checks, would change
     the ranking.

The falsifiability contract is template-driven (no LLM) and derives every
claim directly from the skeptic signal value strings — so nothing it says
can be a hallucination.

CONTRACT
--------
Input:
  candidate        : CandidateFeatureVector
  scores           : ComponentScores
  advocate_signals : list[AdvocateSignal]    ← from advocate.py
  skeptic_signals  : list[SkepticSignal]     ← from skeptic.py

Output:
  TrustVerdict  (from pipeline/schemas.py)

VERDICT CLASSIFICATION (config-driven)
---------------------------------------
  FRAGILE   → HIGH skeptic risks ≥ VERDICT_FRAGILE_HIGH_RISK_COUNT   (2)
  CONTESTED → HIGH skeptic risks == VERDICT_CONTESTED_HIGH_RISK_COUNT (1)
  ROBUST    → 0 HIGH skeptic risks

FLIP RISK MAPPING
-----------------
  FRAGILE   → "HIGH"
  CONTESTED → "MEDIUM"
  ROBUST    → "LOW"

CONFIDENCE PERCENTAGE FORMULA
------------------------------
The confidence_pct is computed as a weighted blend of positive and negative
signal mass, not a copy of the composite score.

  positive_mass = HIGH advocate × 1.0 + MEDIUM advocate × 0.6
  negative_mass = HIGH skeptic  × 1.2 + MODERATE skeptic × 0.5

  raw_confidence = positive_mass / max(positive_mass + negative_mass, 1.0)
  confidence_pct = clamp(raw_confidence × 100, 10.0, 95.0)

Clamped at 10 (never fully confident a candidate is worthless) and 95
(never fully confident a ranking is unflippable — real interviews can
always surface surprises).

FALSIFIABILITY CONTRACT
-----------------------
Generated from the top HIGH and MODERATE skeptic signals.  Each condition
is a one-sentence "This ranking holds UNLESS ..." statement that references
the specific fact in SkepticSignal.value.  2 conditions minimum, 3 maximum.

If there are fewer than 2 skeptic signals, positive falsifiability conditions
are generated instead (conditions that would STRENGTHEN the ranking — e.g.
"If the candidate's GitHub activity confirms open-source depth in IR systems,
this ranking becomes more robust").

DESIGN NOTES
------------
- Deterministic: same inputs → same TrustVerdict, same order.
- No LLM.  No I/O.  No network.  Pure function.
- The `falsifiability` list items are plain strings — reasoning_generator.py
  embeds them verbatim into the recruiter brief.
- All numeric constants that appear in logic come from config.py.
  The only exception is the confidence clamp (10/95) and the signal weights
  (1.0/0.6/1.2/0.5) which are scoring choices internal to this module
  and documented inline.

DEPENDENCIES
------------
  config              : verdict classification thresholds
  pipeline.schemas    : CandidateFeatureVector, ComponentScores,
                        AdvocateSignal, SkepticSignal, TrustVerdict
  trust.advocate      : count_by_confidence, top_signals
  trust.skeptic       : count_by_severity, top_risks

No imports from scoring/ or indexing/ — verdict.py works only with
pre-computed signal lists passed in by the caller (pipeline/runner.py).
"""

from __future__ import annotations

import logging
from typing import Sequence

import config
from pipeline.schemas import (
    AdvocateSignal,
    CandidateFeatureVector,
    ComponentScores,
    SkepticSignal,
    TrustVerdict,
)
from trust.advocate import count_by_confidence
from trust.advocate import top_signals as top_advocate_signals
from trust.skeptic import count_by_severity
from trust.skeptic import top_risks as top_skeptic_risks

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT CONSTANTS (from config)
# ─────────────────────────────────────────────────────────────────────────────

_FRAGILE_HIGH_RISK_COUNT: int = config.VERDICT_FRAGILE_HIGH_RISK_COUNT      # 2
_CONTESTED_HIGH_RISK_COUNT: int = config.VERDICT_CONTESTED_HIGH_RISK_COUNT  # 1

# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE FORMULA WEIGHTS
# (internal to this module — not in config because they are not user-tunable
#  ranking weights, they are signal-mass accounting choices)
# ─────────────────────────────────────────────────────────────────────────────

_ADVOCATE_HIGH_WEIGHT: float = 1.0    # HIGH advocate signal mass
_ADVOCATE_MED_WEIGHT: float = 0.6     # MEDIUM advocate signal mass
_SKEPTIC_HIGH_WEIGHT: float = 1.2     # HIGH skeptic risk mass (weighted harder)
_SKEPTIC_MOD_WEIGHT: float = 0.5      # MODERATE skeptic risk mass

_CONFIDENCE_FLOOR: float = 10.0       # Never output < 10% confidence
_CONFIDENCE_CEIL: float = 95.0        # Never output > 95% confidence

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT / FLIP-RISK STRINGS
# ─────────────────────────────────────────────────────────────────────────────

_ROBUST: str = "ROBUST"
_CONTESTED: str = "CONTESTED"
_FRAGILE: str = "FRAGILE"

_FLIP_LOW: str = "LOW"
_FLIP_MED: str = "MEDIUM"
_FLIP_HIGH: str = "HIGH"


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _classify_verdict(n_high_risks: int) -> tuple[str, str]:
    """
    Map the count of HIGH-severity skeptic risks to verdict + flip_risk.

    Returns (verdict, flip_risk).

    Config thresholds:
      VERDICT_FRAGILE_HIGH_RISK_COUNT   = 2  → FRAGILE / HIGH
      VERDICT_CONTESTED_HIGH_RISK_COUNT = 1  → CONTESTED / MEDIUM
      0 high risks                          → ROBUST / LOW
    """
    if n_high_risks >= _FRAGILE_HIGH_RISK_COUNT:
        return _FRAGILE, _FLIP_HIGH
    if n_high_risks >= _CONTESTED_HIGH_RISK_COUNT:
        return _CONTESTED, _FLIP_MED
    return _ROBUST, _FLIP_LOW


def _compute_confidence(
    advocate_counts: dict[str, int],
    skeptic_counts: dict[str, int],
) -> float:
    """
    Compute confidence_pct as a signal-mass ratio, clamped to [10, 95].

    positive_mass = HIGH_advocate × 1.0 + MEDIUM_advocate × 0.6
    negative_mass = HIGH_skeptic  × 1.2 + MODERATE_skeptic × 0.5

    confidence = positive_mass / (positive_mass + negative_mass) × 100

    A candidate with zero advocates and 2 HIGH risks scores ~10% (floor).
    A candidate with 4 HIGH advocates and 0 risks scores ~95% (ceiling).
    """
    positive = (
        advocate_counts.get("HIGH", 0) * _ADVOCATE_HIGH_WEIGHT
        + advocate_counts.get("MEDIUM", 0) * _ADVOCATE_MED_WEIGHT
    )
    negative = (
        skeptic_counts.get("HIGH", 0) * _SKEPTIC_HIGH_WEIGHT
        + skeptic_counts.get("MODERATE", 0) * _SKEPTIC_MOD_WEIGHT
    )

    total = positive + negative
    if total <= 0:
        # No signals at all — maximum uncertainty.
        return _CONFIDENCE_FLOOR

    raw_pct = (positive / total) * 100.0
    return float(max(_CONFIDENCE_FLOOR, min(_CONFIDENCE_CEIL, raw_pct)))


def _generate_falsifiability(
    skeptic_signals: list[SkepticSignal],
    advocate_signals: list[AdvocateSignal],
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
    verdict: str,
) -> list[str]:
    """
    Generate 2–3 falsifiability conditions for the ranking.

    Strategy:
      - Take the top skeptic risks (HIGH first, then MODERATE).
      - For each, produce a specific "This ranking holds UNLESS ..." condition
        derived directly from the SkepticSignal.value field.
      - If fewer than 2 skeptic signals exist, supplement with positive
        conditions ("This ranking becomes MORE ROBUST if ...") derived from
        the top advocate signals.
      - Always produce exactly 2–3 conditions.

    Every condition references a specific fact (a number, a company name, a
    days count) sourced from the signal.value string — no invented claims.
    """
    conditions: list[str] = []

    # ── Negative conditions from skeptic risks ────────────────────────────────
    priority_risks = top_skeptic_risks(skeptic_signals, n=3)

    for risk in priority_risks:
        condition = _skeptic_signal_to_condition(risk)
        if condition:
            conditions.append(condition)
        if len(conditions) >= 3:
            break

    # ── Supplement with positive conditions if we have fewer than 2 ──────────
    if len(conditions) < 2:
        priority_advocates = top_advocate_signals(advocate_signals, n=3)
        for signal in priority_advocates:
            condition = _advocate_signal_to_positive_condition(signal)
            if condition:
                conditions.append(condition)
            if len(conditions) >= 2:
                break

    # ── Final fallback: generic condition based on verdict tier ───────────────
    if len(conditions) < 2:
        conditions.extend(
            _generic_conditions(verdict, candidate, scores, len(conditions))
        )

    # Return exactly 2–3 conditions (cap at 3).
    return conditions[:3]


def _skeptic_signal_to_condition(signal: SkepticSignal) -> str:
    """
    Convert one SkepticSignal into a falsifiability condition string.

    The condition is always phrased as a specific, interview-actionable check:
    "This ranking holds UNLESS [specific risk is confirmed]."

    The signal.value field contains the concrete fact (days, rate, companies)
    so the condition is never vague.
    """
    label = signal.label
    value = signal.value
    sev = signal.severity

    severity_prefix = (
        "critically" if sev == "HIGH"
        else "notably" if sev == "MODERATE"
        else "marginally"
    )

    # Map specific risk labels to tailored condition templates.
    label_lower = label.lower()

    if "inactivity" in label_lower or "inactive" in label_lower:
        return (
            f"This ranking holds UNLESS the inactivity proves permanent — "
            f"verify candidate is actively looking ({value})"
        )

    if "response rate" in label_lower:
        return (
            f"This ranking holds UNLESS outreach goes unanswered — "
            f"reach out via multiple channels given {value}"
        )

    if "notice period" in label_lower:
        return (
            f"This ranking is {severity_prefix} affected if the notice period "
            f"cannot be shortened — confirm buyout possibility ({value})"
        )

    if "consulting" in label_lower:
        return (
            f"This ranking holds UNLESS prior product-context work can be "
            f"demonstrated despite the consulting background ({value})"
        )

    if "product-company" in label_lower or "product company" in label_lower:
        return (
            f"This ranking holds UNLESS the candidate can demonstrate "
            f"product-facing impact equivalent to product-company exposure ({value})"
        )

    if "hard domain" in label_lower or "disqualifier" in label_lower:
        return (
            f"This ranking is {severity_prefix} weakened by domain mismatch — "
            f"interview must confirm NLP/IR depth beyond: {value}"
        )

    if "soft domain" in label_lower:
        return (
            f"This ranking holds UNLESS the domain-adjacent skills signal "
            f"a pivot rather than genuine IR specialisation ({value})"
        )

    if "missing required" in label_lower or "partial required" in label_lower:
        return (
            f"This ranking holds UNLESS a technical screen confirms the "
            f"skill gaps are compensated by adjacent depth — {value}"
        )

    if "under-experienced" in label_lower or "yoe" in label_lower:
        return (
            f"This ranking holds UNLESS the candidate's production scope "
            f"exceeds what YOE alone suggests ({value})"
        )

    if "job-hopping" in label_lower or "stability" in label_lower:
        return (
            f"This ranking holds UNLESS the candidate commits to a 3+ year "
            f"tenure; pattern suggests risk ({value})"
        )

    if "sparse" in label_lower or "confidence" in label_lower:
        return (
            f"This ranking's confidence is reduced by sparse profile data — "
            f"a technical interview carries extra weight here ({value})"
        )

    if "incomplete" in label_lower:
        return (
            f"This ranking holds UNLESS profile incompleteness hides a "
            f"disqualifying gap — request profile completion ({value})"
        )

    if "github" in label_lower:
        return (
            f"This ranking holds UNLESS no external code/open-source work "
            f"exists to validate technical depth ({value})"
        )

    if "domain" in label_lower:
        return (
            f"This ranking holds UNLESS domain-relevant project work can be "
            f"confirmed through interview ({value})"
        )

    # Generic fallback for unknown risk labels.
    return (
        f"This ranking is {severity_prefix} affected by: {label} — "
        f"verify: {value}"
    )


def _advocate_signal_to_positive_condition(signal: AdvocateSignal) -> str:
    """
    Convert one AdvocateSignal into a positive falsifiability condition.

    Used when skeptic risks are few — the falsifiability contract then focuses
    on what would make the ranking even more robust, rather than what could
    undermine it.  Phrased as "This ranking becomes MORE ROBUST if ...".
    """
    label = signal.label
    value = signal.value
    label_lower = label.lower()

    if "skill" in label_lower and "cluster" in label_lower:
        return (
            f"This ranking becomes MORE ROBUST if a technical interview "
            f"confirms depth in: {value}"
        )

    if "product-company" in label_lower:
        return (
            f"This ranking becomes MORE ROBUST if reference checks confirm "
            f"the candidate drove product outcomes, not just delivery: {value}"
        )

    if "trajectory" in label_lower:
        return (
            f"This ranking becomes MORE ROBUST if the trajectory reflects "
            f"genuine promotion on merit, not tenure: {value}"
        )

    if "github" in label_lower:
        return (
            f"This ranking becomes MORE ROBUST if the GitHub activity reflects "
            f"IR/NLP depth: {value}"
        )

    if "assessment" in label_lower:
        return (
            f"This ranking becomes MORE ROBUST if the assessment score "
            f"reflects current ability, not a coached result: {value}"
        )

    return (
        f"This ranking becomes MORE ROBUST if confirmed by interview: {value}"
    )


def _generic_conditions(
    verdict: str,
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
    already_have: int,
) -> list[str]:
    """
    Last-resort fallback conditions when signal lists are very sparse.

    Generates generic but still factual conditions using ComponentScores values.
    Returns only as many conditions as needed to reach a total of 2.
    """
    needed = 2 - already_have
    conditions: list[str] = []

    yoe = candidate.years_of_experience
    skill_score = scores.skill_score

    if needed >= 1:
        conditions.append(
            f"This ranking holds UNLESS the technical screen reveals "
            f"the {skill_score:.0%} skill score overstates practical ability "
            f"for this role's retrieval and ranking systems requirements"
        )
    if needed >= 2:
        conditions.append(
            f"This ranking holds UNLESS reference checks reveal a pattern "
            f"inconsistent with the {yoe:.1f} years of experience on file"
        )

    return conditions


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_verdict(
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
    advocate_signals: Sequence[AdvocateSignal],
    skeptic_signals: Sequence[SkepticSignal],
) -> TrustVerdict:
    """
    Synthesise Advocate and Skeptic outputs into a final TrustVerdict.

    This is the sole public entry point for trust/verdict.py.
    Called by pipeline/runner.py after advocate.py and skeptic.py have run.

    Parameters
    ----------
    candidate        : Fully parsed CandidateFeatureVector.
    scores           : ComponentScores from scoring/composite.py.
    advocate_signals : list[AdvocateSignal] from trust/advocate.py.
    skeptic_signals  : list[SkepticSignal]  from trust/skeptic.py.

    Returns
    -------
    TrustVerdict  — fully populated, ready for reasoning_generator.py.

    Raises
    ------
    TypeError  : Any argument is not of the expected type.
    ValueError : candidate_id mismatch between candidate and scores.
    """
    # ── Type guards ───────────────────────────────────────────────────────────
    if not isinstance(candidate, CandidateFeatureVector):
        raise TypeError(
            f"candidate must be CandidateFeatureVector, "
            f"got {type(candidate).__name__}"
        )
    if not isinstance(scores, ComponentScores):
        raise TypeError(
            f"scores must be ComponentScores, got {type(scores).__name__}"
        )
    if not isinstance(advocate_signals, (list, tuple)):
        raise TypeError(
            f"advocate_signals must be a sequence, "
            f"got {type(advocate_signals).__name__}"
        )
    if not isinstance(skeptic_signals, (list, tuple)):
        raise TypeError(
            f"skeptic_signals must be a sequence, "
            f"got {type(skeptic_signals).__name__}"
        )

    # ── ID consistency ────────────────────────────────────────────────────────
    if scores.candidate_id != candidate.candidate_id:
        raise ValueError(
            f"ID mismatch: candidate.candidate_id={candidate.candidate_id!r} "
            f"but scores.candidate_id={scores.candidate_id!r}"
        )

    # ── Coerce to lists (defensive — Sequence is acceptable in but list is
    #    cleaner for downstream processing and logging) ────────────────────────
    adv_list: list[AdvocateSignal] = list(advocate_signals)
    skep_list: list[SkepticSignal] = list(skeptic_signals)

    # ── 1. Count signals by tier ──────────────────────────────────────────────
    adv_counts = count_by_confidence(adv_list)
    skep_counts = count_by_severity(skep_list)

    n_high_risks = skep_counts.get("HIGH", 0)

    # ── 2. Classify verdict and flip risk ─────────────────────────────────────
    verdict, flip_risk = _classify_verdict(n_high_risks)

    # ── 3. Compute confidence percentage ─────────────────────────────────────
    confidence_pct = _compute_confidence(adv_counts, skep_counts)

    # ── 4. Generate falsifiability contract ───────────────────────────────────
    falsifiability = _generate_falsifiability(
        skeptic_signals=skep_list,
        advocate_signals=adv_list,
        candidate=candidate,
        scores=scores,
        verdict=verdict,
    )

    # ── 5. Build and return TrustVerdict ─────────────────────────────────────
    trust_verdict = TrustVerdict(
        candidate_id=candidate.candidate_id,
        advocate_signals=adv_list,
        skeptic_signals=skep_list,
        verdict=verdict,
        flip_risk=flip_risk,
        confidence_pct=round(confidence_pct, 1),
        falsifiability=falsifiability,
    )

    logger.debug(
        "verdict: %s → %s | flip=%s | confidence=%.1f%% | "
        "advocate(H=%d,M=%d,L=%d) | skeptic(H=%d,MOD=%d,L=%d) | "
        "falsifiability=%d conditions",
        candidate.candidate_id,
        verdict,
        flip_risk,
        confidence_pct,
        adv_counts.get("HIGH", 0),
        adv_counts.get("MEDIUM", 0),
        adv_counts.get("LOW", 0),
        skep_counts.get("HIGH", 0),
        skep_counts.get("MODERATE", 0),
        skep_counts.get("LOW", 0),
        len(falsifiability),
    )

    return trust_verdict


# ─────────────────────────────────────────────────────────────────────────────
# BATCH HELPER (called by pipeline/runner.py for top-100 candidates)
# ─────────────────────────────────────────────────────────────────────────────

def build_verdicts_batch(
    items: list[tuple[
        CandidateFeatureVector,
        ComponentScores,
        list[AdvocateSignal],
        list[SkepticSignal],
    ]],
) -> dict[str, TrustVerdict]:
    """
    Build TrustVerdict for a batch of candidates.

    Parameters
    ----------
    items : list of 4-tuples (candidate, scores, advocate_signals, skeptic_signals).

    Returns
    -------
    dict[candidate_id → TrustVerdict]

    The batch wrapper is provided so pipeline/runner.py has a clean single
    call for all top-100 candidates without looping over build_verdict itself.
    Errors on individual candidates are caught, logged, and skipped so a
    single bad profile does not abort the entire pipeline.
    """
    results: dict[str, TrustVerdict] = {}

    for candidate, scores, adv_signals, skep_signals in items:
        try:
            verdict = build_verdict(candidate, scores, adv_signals, skep_signals)
            results[candidate.candidate_id] = verdict
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "verdict: failed for %s — %s: %s",
                getattr(candidate, "candidate_id", "<unknown>"),
                type(exc).__name__,
                exc,
            )

    logger.info(
        "verdict batch: %d/%d succeeded.",
        len(results),
        len(items),
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# INTROSPECTION HELPERS (consumed by ui/components/candidate_card.py)
# ─────────────────────────────────────────────────────────────────────────────

def verdict_badge_colour(verdict: str) -> str:
    """
    Return a CSS colour string for the verdict badge in the Streamlit UI.

    ROBUST   → green  (#34d399)
    CONTESTED → amber (#f59e0b)
    FRAGILE  → red   (#ff5c5c)

    Kept here so the UI has a single source of truth for verdict colours
    rather than duplicating the mapping in candidate_card.py.
    """
    _colours = {
        _ROBUST: "#34d399",
        _CONTESTED: "#f59e0b",
        _FRAGILE: "#ff5c5c",
    }
    return _colours.get(verdict, "#8892a4")  # grey for unknown


def summarise_verdict(trust: TrustVerdict) -> str:
    """
    Return a one-line plain-text summary of the TrustVerdict.

    Used by the Streamlit candidate card subtitle and by logging in runner.py.

    Example outputs:
      "ROBUST — confidence 82.0% | 3 HIGH advocate signals | 0 HIGH risks"
      "FRAGILE — confidence 28.0% | 1 HIGH advocate signal | 2 HIGH risks"
    """
    adv_high = sum(1 for s in trust.advocate_signals if s.confidence == "HIGH")
    skep_high = sum(1 for s in trust.skeptic_signals if s.severity == "HIGH")

    adv_noun = "signal" if adv_high == 1 else "signals"
    risk_noun = "risk" if skep_high == 1 else "risks"

    return (
        f"{trust.verdict} — confidence {trust.confidence_pct:.1f}% | "
        f"{adv_high} HIGH advocate {adv_noun} | "
        f"{skep_high} HIGH {risk_noun}"
    )