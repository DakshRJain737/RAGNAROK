"""
ui/components/score_breakdown.py — Streamlit score breakdown component.

Renders an expandable sub-score panel for a single RankedCandidate.
Consumed by ui/app.py inside each candidate card row.

Displays:
    - Skill Match score (×0.40) with sub-items: required coverage, NTH coverage,
      ontology matches, proficiency breakdown
    - Career Quality score (×0.35) with sub-items: YOE, product-co flag,
      consulting-only flag, trajectory velocity, location bonus
    - Behavioral score (×0.25) with sub-items: recency decay, notice period,
      open_to_work, recruiter response rate, GitHub activity
    - Uncertainty penalty
    - Composite formula bar
    - Honeypot status

Dependencies:
    streamlit, pipeline/schemas.py (RankedCandidate, ComponentScores, TrustVerdict)
"""

from __future__ import annotations

import streamlit as st


def render_score_breakdown(rc) -> None:
    """
    Render the expandable score breakdown for one RankedCandidate object.

    Args:
        rc: pipeline.schemas.RankedCandidate (with .components and .trust populated)
    """
    comp   = rc.components       # scoring.composite.ComponentScores or None
    trust  = rc.trust            # pipeline.schemas.TrustVerdict or None
    cfv    = rc.feature_vector   # pipeline.schemas.CandidateFeatureVector or None

    if comp is None:
        st.info("Score breakdown not available (pipeline ran in lightweight mode).")
        return

    # ── Composite formula header ───────────────────────────────────────────────
    st.markdown("#### 🎯 Composite Score Formula")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Composite Score", f"{comp.final_score:.4f}")
    col2.metric("Weighted Sum", f"{comp.weighted_sum:.4f}",
                help="0.40×skill + 0.35×career + 0.25×behavioral")
    col3.metric("Cross-Encoder", f"{comp.cross_encoder_score:.4f}",
                help="ms-marco-MiniLM-L-6-v2 pairwise score (blended at 30%)")
    col4.metric("RRF Score", f"{comp.rrf_score:.6f}",
                help="Σ 1/(60+rank) across all 5 retrieval paths")

    # Formula bar
    formula_pct = {
        "Skill (40%)":    comp.skill_match_score * 0.40,
        "Career (35%)":   comp.career_quality_score * 0.35,
        "Behavioral (25%)": comp.behavioral_score * 0.25,
    }
    st.markdown("**Score Decomposition**")
    for label, val in formula_pct.items():
        st.progress(min(1.0, val / 0.4 if "Skill" in label else val / 0.35 if "Career" in label else val / 0.25),
                    text=f"{label}: {val:.4f}")

    st.divider()

    # ── Three columns: Skill | Career | Behavioral ─────────────────────────────
    col_skill, col_career, col_beh = st.columns(3)

    # ── SKILL MATCH ───────────────────────────────────────────────────────────
    with col_skill:
        skill_color = "🟢" if comp.skill_match_score >= 0.8 else "🟡" if comp.skill_match_score >= 0.6 else "🔴"
        st.markdown(f"### {skill_color} Skill Match")
        st.metric("Score", f"{comp.skill_match_score:.4f}",
                  delta=f"Weight: ×0.40")

        if cfv:
            # Use pipeline.schemas.ComponentScores sub-scores if available
            schema_comp = getattr(rc, '_schema_components', None)
            req_cov = getattr(schema_comp, 'required_skill_coverage', None)
            nth_cov = getattr(schema_comp, 'nice_to_have_coverage', None)
            ontology = getattr(schema_comp, 'ontology_skills_matched', [])

            if req_cov is not None:
                st.progress(req_cov, text=f"Required skill coverage: {req_cov:.0%}")
            if nth_cov is not None:
                st.progress(nth_cov, text=f"Nice-to-have coverage: {nth_cov:.0%}")

            # Top skills
            st.markdown("**Candidate Skills**")
            skills_sorted = sorted(cfv.skills,
                                   key=lambda s: (s.proficiency == "expert",
                                                  s.proficiency == "advanced",
                                                  s.endorsements), reverse=True)
            for s in skills_sorted[:8]:
                prof_emoji = {"expert": "⭐", "advanced": "🔷", "intermediate": "🔹", "beginner": "○"}.get(s.proficiency, "·")
                score_str  = f" — assess: {s.assessment_score:.0f}" if s.assessment_score >= 0 else ""
                st.caption(f"{prof_emoji} **{s.name_raw}** ({s.proficiency}, {s.duration_months}m{score_str})")

            if ontology:
                st.markdown(f"**🕸️ Ontology Matches** ({len(ontology)})")
                st.caption(", ".join(ontology[:6]))

        if comp.hard_disqualifier:
            st.error("⛔ Hard disqualifier triggered — score forced to 0.0")

    # ── CAREER QUALITY ────────────────────────────────────────────────────────
    with col_career:
        career_color = "🟢" if comp.career_quality_score >= 0.7 else "🟡" if comp.career_quality_score >= 0.5 else "🔴"
        st.markdown(f"### {career_color} Career Quality")
        st.metric("Score", f"{comp.career_quality_score:.4f}",
                  delta="Weight: ×0.35")

        schema_comp = getattr(rc, '_schema_components', None)
        yoe_score  = getattr(schema_comp, 'yoe_score', None)
        traj_vel   = comp.trajectory_velocity
        prod_flag  = getattr(schema_comp, 'product_co_flag', None)
        cons_flag  = getattr(schema_comp, 'consulting_only_flag', None)
        loc_bonus  = comp.location_bonus_applied

        if yoe_score is not None:
            st.progress(yoe_score, text=f"YOE band fit (5–9y): {yoe_score:.0%}")
        if traj_vel is not None:
            st.progress(min(1.0, traj_vel), text=f"Trajectory velocity: {traj_vel:.2f}")

        if cfv:
            st.markdown("**Career Flags**")
            if prod_flag is not None:
                st.success("✅ Product-company experience") if prod_flag else st.warning("⚠️ No product-company history")
            if cons_flag is not None:
                st.error("❌ Consulting-only career (penalty applied)") if cons_flag else st.success("✅ Not consulting-only")
            if loc_bonus > 0:
                st.info(f"📍 Location bonus: +{loc_bonus:.2%} ({cfv.location})")

        if cfv and cfv.career_history:
            st.markdown("**Career History**")
            for entry in cfv.career_history[:4]:
                end = entry.end_date.isoformat() if entry.end_date else "present"
                st.caption(f"🏢 **{entry.company}** · {entry.title} · {entry.duration_months}m ({entry.start_date.year}–{end[:4]})")

    # ── BEHAVIORAL ────────────────────────────────────────────────────────────
    with col_beh:
        beh_color = "🟢" if comp.behavioral_score >= 0.75 else "🟡" if comp.behavioral_score >= 0.5 else "🔴"
        st.markdown(f"### {beh_color} Behavioral")
        st.metric("Score", f"{comp.behavioral_score:.4f}",
                  delta="Weight: ×0.25")

        unc_penalty = comp.uncertainty_penalty_applied
        if unc_penalty < 1.0:
            st.warning(f"⚠️ Uncertainty penalty: ×{unc_penalty:.2f} (sparse profile)")

        if cfv:
            sig = cfv.signals
            st.markdown("**Engagement Signals**")

            open_work = "✅ Open to work" if sig.open_to_work_flag else "❌ Not open to work"
            st.caption(open_work)

            notice = sig.notice_period_days
            notice_emoji = "✅" if notice <= 30 else "⚠️" if notice <= 60 else "❌"
            st.caption(f"{notice_emoji} Notice period: {notice}d")

            recency = getattr(rc, '_schema_components', None)
            recency_score = getattr(recency, 'recency_score', None)
            if recency_score is not None:
                st.progress(recency_score, text=f"Recency score: {recency_score:.0%} ({sig.days_since_active}d ago)")
            else:
                st.caption(f"🕐 Last active: {sig.days_since_active}d ago")

            rr = sig.recruiter_response_rate
            rr_emoji = "✅" if rr >= 0.8 else "⚠️" if rr >= 0.6 else "❌"
            st.caption(f"{rr_emoji} Recruiter response rate: {rr:.0%}")

            gh = sig.github_activity_score
            if gh >= 0:
                gh_emoji = "✅" if gh >= 70 else "⚠️" if gh >= 40 else "❌"
                st.caption(f"{gh_emoji} GitHub activity: {gh:.0f}/100")
            else:
                st.caption("❌ GitHub: not linked")

            st.caption(f"📊 Profile completeness: {sig.profile_completeness_score:.0f}%")

    st.divider()

    # ── Retrieval Paths ────────────────────────────────────────────────────────
    st.markdown("#### 🔀 Retrieval Path Coverage")
    path_cols = st.columns(5)
    path_info = [
        ("Semantic", "#6366f1"),
        ("Keyword",  "#10b981"),
        ("Ontology", "#8b5cf6"),
        ("Trajectory", "#f59e0b"),
        ("Signal",   "#0ea5e9"),
    ]
    for i, (name, color) in enumerate(path_info):
        present = name.lower() in [p.lower() for p in comp.paths_present]
        path_cols[i].markdown(
            f"{'🟢' if present else '⬜'} **{name}**",
            help=f"Path {i+1}: {'✅ Retrieved this candidate' if present else '❌ Did not retrieve'}"
        )

    # ── Trust Layer ───────────────────────────────────────────────────────────
    if trust is not None:
        st.divider()
        st.markdown("#### 🛡️ Trust Layer Analysis")

        verdict_color = {"ROBUST": "success", "CONTESTED": "warning", "FRAGILE": "error"}.get(trust.verdict, "info")
        getattr(st, verdict_color)(
            f"**{trust.verdict}** — Flip Risk: {trust.flip_risk} — Confidence: {trust.confidence_pct:.0f}%"
        )

        col_adv, col_skep = st.columns(2)
        with col_adv:
            st.markdown("**✅ Advocate Signals**")
            for sig in trust.advocate_signals:
                conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔵"}.get(sig.confidence, "·")
                st.caption(f"{conf_emoji} [{sig.confidence}] **{sig.label}** — {sig.value}")

        with col_skep:
            st.markdown("**⚠️ Skeptic Signals**")
            for sig in trust.skeptic_signals:
                sev_emoji = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🔵"}.get(sig.severity, "·")
                st.caption(f"{sev_emoji} [{sig.severity}] **{sig.label}** — {sig.value}")

        if trust.falsifiability:
            st.markdown("**📋 Falsifiability Conditions**")
            for cond in trust.falsifiability:
                st.caption(f"↩ {cond}")

    # ── Honeypot status ───────────────────────────────────────────────────────
    if cfv and cfv.is_honeypot:
        st.divider()
        st.error("🛑 **HONEYPOT DETECTED** — Profile removed from ranking (score forced to 0.0)")
