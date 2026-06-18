"""
ui/components/candidate_card.py — Streamlit candidate card component.

Renders a single ranked candidate as a rich, expandable card.
Used by ui/app.py in the results table section.

Each card shows:
    - Rank badge (gold for top-3, indigo for top-10, grey otherwise)
    - Candidate name, title, company, location, YOE
    - Composite score bar
    - Verdict badge (ROBUST / CONTESTED / FRAGILE)
    - Retrieval path dots
    - Expandable section → renders score_breakdown.py

Dependencies:
    streamlit, ui/components/score_breakdown.py
"""

from __future__ import annotations

import streamlit as st
from ui.components.score_breakdown import render_score_breakdown


# ── Palette (matches ats_platform.html CSS vars) ──────────────────────────────
_VERDICT_CSS = {
    "ROBUST":    ("✅", "#10b981", "#ecfdf5"),
    "CONTESTED": ("⚠️", "#f59e0b", "#fffbeb"),
    "FRAGILE":   ("❌", "#f43f5e", "#fff1f2"),
}
_PATH_COLORS = {
    "semantic":   "#6366f1",
    "keyword":    "#10b981",
    "ontology":   "#8b5cf6",
    "trajectory": "#f59e0b",
    "signal":     "#0ea5e9",
}
_ALL_PATHS = ["semantic", "keyword", "ontology", "trajectory", "signal"]


def _rank_badge(rank: int) -> str:
    if rank <= 3:
        return f"🥇🥈🥉"[rank - 1]
    if rank <= 10:
        return f"#{rank}"
    return f"#{rank}"


def _score_color(score: float) -> str:
    if score >= 0.9:
        return "#6366f1"
    if score >= 0.8:
        return "#10b981"
    if score >= 0.7:
        return "#f59e0b"
    return "#f43f5e"


def render_candidate_card(rc, index: int = 0) -> None:
    """
    Render a single RankedCandidate as an expandable Streamlit card.

    Args:
        rc:    pipeline.schemas.RankedCandidate (fully populated)
        index: Card index for unique widget keys
    """
    cfv   = rc.feature_vector
    comp  = rc.components
    trust = rc.trust
    rank  = rc.rank

    # ── Card container ─────────────────────────────────────────────────────────
    is_honeypot = cfv.is_honeypot if cfv else False
    verdict_info = _VERDICT_CSS.get(
        trust.verdict if trust else "CONTESTED",
        ("⚠️", "#f59e0b", "#fffbeb")
    )

    # Top-level expander (collapsed by default)
    label = _build_card_label(rc, cfv, comp, trust, rank, verdict_info, is_honeypot)

    with st.expander(label, expanded=False):
        render_score_breakdown(rc)
        _render_card_footer(rc, index)


def _build_card_label(rc, cfv, comp, trust, rank, verdict_info, is_honeypot) -> str:
    """Build the one-line expander label for the candidate card."""
    verdict_emoji, verdict_color, _ = verdict_info
    rank_str = _rank_badge(rank)

    name  = cfv.headline if cfv else rc.candidate_id
    title = (cfv.current_title + " @ " + cfv.current_company) if cfv else ""
    score = f"{rc.final_score:.4f}"
    yoe   = f"{cfv.years_of_experience:.0f}y" if cfv else ""
    loc   = cfv.location if cfv else ""
    hp    = " 🛑 HONEYPOT" if is_honeypot else ""

    verdict_str = ""
    if trust:
        verdict_str = f" | {verdict_emoji} {trust.verdict}"

    paths_present = set(p.lower() for p in (comp.paths_present if comp else []))
    path_str = "".join(
        "●" if p in paths_present else "○"
        for p in _ALL_PATHS
    )

    return (
        f"**{rank_str}** · **{name}**{hp} · {title} · "
        f"⭐ `{score}`{verdict_str} · "
        f"{yoe} · {loc} · paths: `{path_str}`"
    )


def _render_card_footer(rc, index: int) -> None:
    """Render action buttons at the bottom of the expanded card."""
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📋 Copy reasoning", key=f"copy_{index}_{rc.candidate_id}"):
            st.code(rc.reasoning)

    with col2:
        if st.button("✅ Shortlist", key=f"shortlist_{index}_{rc.candidate_id}"):
            st.success(f"✅ {rc.candidate_id} added to shortlist!")
            if "shortlisted" not in st.session_state:
                st.session_state.shortlisted = []
            if rc.candidate_id not in st.session_state.shortlisted:
                st.session_state.shortlisted.append(rc.candidate_id)

    with col3:
        if st.button("🚩 Flag for review", key=f"flag_{index}_{rc.candidate_id}"):
            st.warning(f"🚩 {rc.candidate_id} flagged for manual review.")

    with col4:
        if st.button("📊 Debug scores", key=f"debug_{index}_{rc.candidate_id}"):
            st.json({
                "candidate_id": rc.candidate_id,
                "rank": rc.rank,
                "final_score": rc.final_score,
                "components": {
                    "skill": rc.components.skill_match_score if rc.components else None,
                    "career": rc.components.career_quality_score if rc.components else None,
                    "behavioral": rc.components.behavioral_score if rc.components else None,
                    "cross_encoder": rc.components.cross_encoder_score if rc.components else None,
                    "rrf": rc.components.rrf_score if rc.components else None,
                    "paths": rc.components.paths_present if rc.components else [],
                },
                "verdict": rc.trust.verdict if rc.trust else None,
            } if rc.components else {"note": "components not available"})


# ── Table-mode rendering (compact, no expander) ───────────────────────────────

def render_candidates_table(ranked_candidates: list, max_rows: int = 100) -> None:
    """
    Render all ranked candidates as a compact interactive table with row expanders.

    Args:
        ranked_candidates: list of RankedCandidate objects
        max_rows: maximum rows to display (default 100)
    """
    if not ranked_candidates:
        st.info("No candidates to display. Run the pipeline first.")
        return

    display = [rc for rc in ranked_candidates if not (rc.feature_vector and rc.feature_vector.is_honeypot)]
    honeypots = [rc for rc in ranked_candidates if rc.feature_vector and rc.feature_vector.is_honeypot]

    # ── Summary metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ranked", len(display))
    col2.metric("ROBUST", sum(1 for rc in display if rc.trust and rc.trust.verdict == "ROBUST"))
    col3.metric("CONTESTED", sum(1 for rc in display if rc.trust and rc.trust.verdict == "CONTESTED"))
    col4.metric("Honeypots Removed", len(honeypots))

    st.markdown("---")

    # ── Filter bar ────────────────────────────────────────────────────────────
    col_filter, col_sort, col_search = st.columns([2, 2, 3])
    with col_filter:
        verdict_filter = st.selectbox(
            "Filter by Verdict",
            ["ALL", "ROBUST", "CONTESTED", "FRAGILE"],
            key="verdict_filter",
        )
    with col_sort:
        sort_by = st.selectbox(
            "Sort By",
            ["Score (↓)", "Rank (↑)", "YOE (↓)", "Behavioral (↓)"],
            key="sort_by",
        )
    with col_search:
        search_q = st.text_input(
            "Search candidates",
            placeholder="Name, company, skill…",
            key="cand_search",
        )

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = display[:max_rows]

    if verdict_filter != "ALL":
        filtered = [
            rc for rc in filtered
            if rc.trust and rc.trust.verdict == verdict_filter
        ]

    if search_q:
        q = search_q.lower()
        filtered = [
            rc for rc in filtered
            if (rc.feature_vector and (
                q in rc.feature_vector.current_company.lower()
                or q in rc.feature_vector.current_title.lower()
                or q in rc.feature_vector.location.lower()
                or any(q in s.name for s in rc.feature_vector.skills)
            ))
        ]

    if sort_by == "YOE (↓)":
        filtered.sort(key=lambda rc: -(rc.feature_vector.years_of_experience if rc.feature_vector else 0))
    elif sort_by == "Behavioral (↓)":
        filtered.sort(key=lambda rc: -(rc.components.behavioral_score if rc.components else 0))

    st.caption(f"Showing {len(filtered)} candidates")

    # ── Candidate cards ───────────────────────────────────────────────────────
    for i, rc in enumerate(filtered):
        render_candidate_card(rc, index=i)

    # ── Honeypot section ──────────────────────────────────────────────────────
    if honeypots:
        st.markdown("---")
        with st.expander(f"🛑 {len(honeypots)} Honeypot Profile(s) — Removed from Ranking"):
            st.warning(
                "These profiles were flagged by the Honeypot Registry before cross-encoder scoring. "
                "They are NOT in the submission.csv output."
            )
            for hp in honeypots:
                st.error(f"**{hp.candidate_id}** — Score: 0.0 — Reason: impossible profile detected")
