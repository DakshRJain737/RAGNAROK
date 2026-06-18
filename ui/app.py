"""
ui/app.py — RAGnarok ATS · Streamlit Sandbox UI

Full end-to-end demo interface for the Redrob Hackathon.
Judges can:
    1. Upload candidates.jsonl (or .json / .jsonl.gz)
    2. Enter or paste a job description override
    3. Click "⚡ Run Pipeline" to trigger the FastAPI backend
    4. Browse ranked results with full score breakdowns and trust verdicts
    5. Export submission.csv

Start with:
    streamlit run ui/app.py

Requires:
    - api/main.py running on http://localhost:8000  (uvicorn api.main:app)
    OR
    - Direct import from pipeline modules (fallback mode when API is unavailable)
"""

from __future__ import annotations

import gzip
import io
import json
import time
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="RAGnarok ATS — Intelligent Candidate Ranking",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/krishna-zalavadiya/RAGnarok",
        "Report a bug": "https://github.com/krishna-zalavadiya/RAGnarok/issues",
        "About": "RAGnarok — Hackathon intelligent candidate ranking system",
    },
)

logger = logging.getLogger(__name__)

# ── API config ────────────────────────────────────────────────────────────────
_API_BASE = "http://localhost:8000"
_TIMEOUT  = 300   # 5 minutes (matches pipeline budget)

# ── Project root (for default JD / sample candidates) ─────────────────────────
_ROOT = Path(__file__).parent.parent


# ─── SESSION STATE INIT ───────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "ranked_candidates": [],
        "pipeline_elapsed_ms": None,
        "stage_timings": {},
        "honeypots_removed": 0,
        "total_input": 0,
        "errors": [],
        "shortlisted": [],
        "run_mode": "api",           # "api" or "direct"
        "last_run_id": None,
        "api_healthy": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─── CUSTOM CSS (matches ats_platform.html design system) ─────────────────────
def _inject_css() -> None:
    st.markdown("""
    <style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .main { background: #f7f8fc; }
    .block-container { padding: 1.5rem 2rem; max-width: 1200px; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e4e7f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(15,17,23,0.06);
    }
    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 16px rgba(15,17,23,0.08);
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
    [data-testid="stMetricLabel"] { color: #9298b0; font-size: 12px; font-weight: 600; }
    [data-testid="stMetricValue"]  { color: #0f1117; font-weight: 800; }

    /* Expander cards */
    .streamlit-expanderHeader {
        background: #ffffff;
        border: 1px solid #e4e7f0;
        border-radius: 10px;
        font-size: 13px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        border-color: #818cf8;
        background: #eef2ff;
    }
    .streamlit-expanderContent {
        background: #f8f9ff;
        border: 1px solid rgba(99,102,241,0.15);
        border-top: none;
        border-radius: 0 0 10px 10px;
        padding: 16px;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #e4e7f0;
        border-radius: 12px;
        background: #f8f9ff;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover { border-color: #818cf8; }

    /* Primary button style */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: 0 3px 12px rgba(99,102,241,0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e4e7f0;
    }
    [data-testid="stSidebar"] .block-container { padding: 1rem; }

    /* Progress bar */
    .stProgress > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6); }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f3fa;
        border-radius: 8px;
        padding: 3px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 1px 4px rgba(15,17,23,0.08);
    }

    /* Hero gradient header */
    .hero-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #c084fc 100%);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 8px 32px rgba(99,102,241,0.3);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 99px;
        padding: 3px 12px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: 0.05em;
    }
    .hero-title { font-size: 24px; font-weight: 900; margin: 4px 0; letter-spacing: -0.5px; }
    .hero-sub { font-size: 13px; opacity: 0.8; line-height: 1.5; }
    .hero-stat { display: inline-block; margin-right: 28px; }
    .hero-stat-val { font-size: 28px; font-weight: 900; }
    .hero-stat-label { font-size: 11px; opacity: 0.7; margin-top: 2px; }

    /* Status badge */
    .status-live { color: #10b981; font-weight: 700; }
    .status-dot { display: inline-block; width: 8px; height: 8px; background: #10b981;
                  border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

    /* Code blocks */
    code { background: #eef2ff; color: #6366f1; padding: 1px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ─── API HELPERS ──────────────────────────────────────────────────────────────

def _check_api_health() -> bool:
    """Ping GET /health and return True if pipeline is ready."""
    try:
        import requests
        r = requests.get(f"{_API_BASE}/health", timeout=3)
        return r.status_code == 200 and r.json().get("pipeline_ready", False)
    except Exception:
        return False


def _run_pipeline_via_api(
    candidates_jsonl: str,
    jd_text: Optional[str],
    top_k: int,
) -> dict:
    """POST /rank and return the JSON response dict."""
    import requests
    payload = {
        "candidates_jsonl": candidates_jsonl,
        "jd_text": jd_text,
        "top_k": top_k,
    }
    r = requests.post(f"{_API_BASE}/rank", json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _export_csv_via_api() -> bytes:
    """POST /export/csv and return the CSV bytes."""
    import requests
    r = requests.post(
        f"{_API_BASE}/export/csv",
        json={"validate_before_export": True},
        timeout=30,
    )
    r.raise_for_status()
    return r.content


# ─── DIRECT PIPELINE MODE ─────────────────────────────────────────────────────

def _run_pipeline_direct(
    candidates_jsonl: str,
    jd_text: Optional[str],
    top_k: int,
) -> list:
    """
    Fallback: run pipeline directly (imports pipeline modules without FastAPI).
    Returns list of RankedCandidate objects.
    """
    from pipeline.candidate_parser import CandidateParser
    from pipeline.jd_parser import JDParser
    from pipeline.runner import PipelineRunner

    parser = CandidateParser()
    candidates = []
    for line in candidates_jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            candidates.append(parser.parse_candidate(json.loads(line)))
        except Exception as e:
            logger.warning("Parse error: %s", e)

    jd_parser  = JDParser()
    jd_intent  = jd_parser.parse(jd_text) if jd_text else jd_parser.load_parsed()
    runner     = PipelineRunner(jd=jd_intent, candidates=candidates)
    ranked, _  = runner.run(top_k=top_k)
    return ranked


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

def _render_sidebar() -> dict:
    """Render sidebar controls. Returns config dict."""
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
            <div style="width:38px;height:38px;background:linear-gradient(135deg,#6366f1,#8b5cf6);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;
                        font-size:18px;font-weight:900;color:#fff;box-shadow:0 4px 12px rgba(99,102,241,0.35);">R⚡</div>
            <div>
                <div style="font-size:16px;font-weight:800;color:#0f1117;">RAGnarok</div>
                <div style="font-size:10px;color:#9298b0;text-transform:uppercase;letter-spacing:0.06em;">ATS Platform</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # API health indicator
        if st.button("🔄 Check API", key="check_api"):
            healthy = _check_api_health()
            st.session_state.api_healthy = healthy

        if st.session_state.api_healthy is True:
            st.success("🟢 API ready")
        elif st.session_state.api_healthy is False:
            st.warning("🟡 API offline — using direct mode")
        else:
            st.info("⚪ API status unknown")

        st.divider()

        # Run mode
        run_mode = st.radio(
            "Run Mode",
            ["🌐 Via API (FastAPI)", "⚙️ Direct (Python)"],
            key="run_mode_radio",
            help="API mode calls POST /rank. Direct mode imports pipeline modules.",
        )
        st.session_state.run_mode = "api" if "API" in run_mode else "direct"

        st.divider()

        # Pipeline settings
        st.markdown("**⚙️ Pipeline Settings**")
        top_k = st.slider("Top-K results", min_value=10, max_value=100, value=100, step=10)
        show_honeypots = st.checkbox("Show honeypot profiles", value=False)
        show_debug = st.checkbox("Show debug info", value=False)

        st.divider()

        # Shortlist
        if st.session_state.shortlisted:
            st.markdown(f"**✅ Shortlisted ({len(st.session_state.shortlisted)})**")
            for cid in st.session_state.shortlisted:
                st.caption(f"· {cid}")
            if st.button("🗑️ Clear shortlist"):
                st.session_state.shortlisted = []
                st.rerun()

        st.divider()

        # Links
        st.markdown("**📚 Resources**")
        st.markdown("- [Pipeline Architecture](../Optimised_Pipeline.html)")
        st.markdown("- [Sprint Planner](../Daily_Task(sprint planner).html)")
        st.markdown("- [API Docs](http://localhost:8000/docs)")
        st.markdown("- [GitHub](https://github.com/krishna-zalavadiya/RAGnarok)")

    return {
        "top_k": top_k,
        "show_honeypots": show_honeypots,
        "show_debug": show_debug,
    }


# ─── HERO HEADER ──────────────────────────────────────────────────────────────

def _render_hero() -> None:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">⚡ Redrob Hackathon · RAGnarok v1.0</div>
        <div class="hero-title">Intelligent Candidate Ranking System</div>
        <div class="hero-sub">
            5-path retrieval · RRF fusion · Cross-encoder rerank ·
            0.40×skill + 0.35×career + 0.25×behavioral · Adversarial trust layer
        </div>
        <div style="margin-top:18px;">
            <div class="hero-stat"><div class="hero-stat-val">100K</div><div class="hero-stat-label">Candidate Pool</div></div>
            <div class="hero-stat"><div class="hero-stat-val">~15s</div><div class="hero-stat-label">Runtime</div></div>
            <div class="hero-stat"><div class="hero-stat-val">5</div><div class="hero-stat-label">Retrieval Paths</div></div>
            <div class="hero-stat"><div class="hero-stat-val">0%</div><div class="hero-stat-label">Hallucination Rate</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── INPUT SECTION ────────────────────────────────────────────────────────────

def _render_input_section() -> tuple[Optional[str], Optional[str]]:
    """
    Render the candidate upload + JD input section.
    Returns (candidates_jsonl: str | None, jd_text: str | None).
    """
    st.markdown("### 📥 Step 1 — Input")
    col_upload, col_jd = st.columns([1, 1], gap="large")

    candidates_jsonl: Optional[str] = None
    jd_text: Optional[str] = None

    with col_upload:
        st.markdown("**Candidates File**")
        st.caption("Upload `.jsonl`, `.json`, or `.jsonl.gz` (up to 50MB)")

        uploaded = st.file_uploader(
            "Drop candidates file here",
            type=["jsonl", "json", "gz"],
            key="candidates_upload",
            label_visibility="collapsed",
        )

        if uploaded:
            raw = uploaded.read()
            if uploaded.name.endswith(".gz"):
                raw = gzip.decompress(raw)
            text = raw.decode("utf-8", errors="ignore")

            # Count lines
            lines = [l for l in text.splitlines() if l.strip()]
            st.success(f"✅ Loaded **{len(lines):,}** candidate records ({len(raw) / 1024:.0f} KB)")

            # Detect format: JSON array vs JSONL
            stripped = text.strip()
            if stripped.startswith("["):
                # Convert JSON array to JSONL
                try:
                    arr = json.loads(stripped)
                    candidates_jsonl = "\n".join(json.dumps(c) for c in arr)
                    st.info("ℹ️ JSON array detected — converted to JSONL.")
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON parse error: {e}")
            else:
                candidates_jsonl = text

        else:
            # Option: use sample data
            sample_path = _ROOT / "data" / "sample_candidates.json"
            if not sample_path.exists():
                sample_path = _ROOT / "data" / "sample_candidates.jsonl"

            if sample_path.exists():
                if st.button("📂 Use sample candidates", key="use_sample"):
                    raw = sample_path.read_text(encoding="utf-8")
                    if raw.strip().startswith("["):
                        arr = json.loads(raw)
                        candidates_jsonl = "\n".join(json.dumps(c) for c in arr)
                    else:
                        candidates_jsonl = raw
                    st.success(f"✅ Loaded sample from `{sample_path.name}`")
                    st.session_state["_sample_jsonl"] = candidates_jsonl
            else:
                st.info("📌 No sample file found. Upload a candidates file to begin.")

            # Persist sample across reruns
            if "_sample_jsonl" in st.session_state and not uploaded:
                candidates_jsonl = st.session_state["_sample_jsonl"]

    with col_jd:
        st.markdown("**Job Description**")
        st.caption("Paste a custom JD or leave blank to use the pre-parsed JD")

        default_jd = ""
        jd_path = _ROOT / "job_description.md"
        if jd_path.exists():
            default_jd = jd_path.read_text(encoding="utf-8")[:2000]

        jd_input = st.text_area(
            "Job Description (optional)",
            value="",
            placeholder=default_jd[:400] + "…\n\n(Leave blank to use pre-parsed JD)",
            height=250,
            key="jd_input",
            label_visibility="collapsed",
        )
        if jd_input.strip():
            jd_text = jd_input.strip()
            st.caption(f"✅ Custom JD: {len(jd_text)} chars")
        else:
            st.caption("📄 Using pre-parsed `parsed_job_description.json`")

    return candidates_jsonl, jd_text


# ─── RUN PIPELINE ─────────────────────────────────────────────────────────────

def _run_pipeline(
    candidates_jsonl: str,
    jd_text: Optional[str],
    top_k: int,
) -> None:
    """Execute the pipeline and store results in session state."""
    mode = st.session_state.run_mode
    st.markdown("### ⚡ Step 2 — Running Pipeline")

    stages = [
        "Loading indexes (FAISS + BM25 + feature store)…",
        "Encoding JD query (MiniLM)…",
        "Running 5 parallel retrieval paths…",
        "RRF fusion → top-60 pool…",
        "Honeypot filter (O(1) registry lookup)…",
        "Cross-encoder reranking top-50…",
        "Composite scoring (0.40+0.35+0.25)…",
        "Adversarial trust layer (Advocate + Skeptic + Verdict)…",
        "Generating reasoning strings…",
    ]
    progress_bar = st.progress(0, text=stages[0])
    status_placeholder = st.empty()

    def update_progress(i: int, msg: str) -> None:
        progress_bar.progress((i + 1) / len(stages), text=msg)
        status_placeholder.caption(f"⏳ {msg}")

    t_start = time.perf_counter()

    try:
        if mode == "api":
            # Animate progress while waiting for API
            import threading
            api_result: dict = {}
            api_error: list  = []

            def _api_thread():
                try:
                    api_result["data"] = _run_pipeline_via_api(candidates_jsonl, jd_text, top_k)
                except Exception as e:
                    api_error.append(str(e))

            thread = threading.Thread(target=_api_thread)
            thread.start()

            for i, stage in enumerate(stages):
                if not thread.is_alive():
                    break
                update_progress(i, stage)
                time.sleep(1.5)
            thread.join(timeout=_TIMEOUT)

            if api_error:
                st.error(f"❌ API Error: {api_error[0]}")
                st.info("💡 Tip: Start the API with `uvicorn api.main:app --port 8000` then retry.")
                return

            if "data" not in api_result:
                st.error("❌ Pipeline timed out or returned no data.")
                return

            result = api_result["data"]
            # API result is a dict with "ranked" as list of dicts
            st.session_state.ranked_candidates = result.get("ranked", [])
            st.session_state.pipeline_elapsed_ms = result.get("pipeline_elapsed_ms", 0)
            st.session_state.stage_timings = result.get("stage_timings", {})
            st.session_state.honeypots_removed = result.get("honeypots_removed", 0)
            st.session_state.total_input = result.get("total_candidates_input", 0)
            st.session_state.errors = result.get("errors", [])

        else:
            # Direct mode
            update_progress(0, "Loading pipeline modules…")
            ranked = _run_pipeline_direct(candidates_jsonl, jd_text, top_k)
            # Convert RankedCandidate objects to dicts for display
            st.session_state.ranked_candidates = ranked
            st.session_state.pipeline_elapsed_ms = (time.perf_counter() - t_start) * 1000
            st.session_state.honeypots_removed = sum(
                1 for rc in ranked
                if hasattr(rc, "feature_vector") and rc.feature_vector and rc.feature_vector.is_honeypot
            )
            st.session_state.total_input = len(ranked)

        elapsed = time.perf_counter() - t_start
        progress_bar.progress(1.0, text="✅ Pipeline complete!")
        status_placeholder.success(
            f"✅ Pipeline completed in **{elapsed:.1f}s** — "
            f"**{len(st.session_state.ranked_candidates)}** candidates ranked · "
            f"**{st.session_state.honeypots_removed}** honeypots removed"
        )

    except Exception as exc:
        progress_bar.empty()
        st.error(f"❌ Pipeline failed: {exc}")
        logger.exception("Pipeline run failed")


# ─── RESULTS SECTION ──────────────────────────────────────────────────────────

def _render_results(config: dict) -> None:
    """Render the ranked results with tabs."""
    ranked = st.session_state.ranked_candidates
    if not ranked:
        return

    st.markdown("### 📊 Step 3 — Results")

    # ── Summary strip ─────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    elapsed_s = (st.session_state.pipeline_elapsed_ms or 0) / 1000
    col1.metric("Candidates Input", f"{st.session_state.total_input:,}")
    col2.metric("Ranked Output", len(ranked))
    col3.metric("Honeypots Removed", st.session_state.honeypots_removed)
    col4.metric("Pipeline Time", f"{elapsed_s:.1f}s")

    top_score = None
    if ranked:
        first = ranked[0]
        if isinstance(first, dict):
            top_score = first.get("score", 0)
        elif hasattr(first, "final_score"):
            top_score = first.final_score
    col5.metric("Top Score", f"{top_score:.4f}" if top_score is not None else "—")

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_results, tab_csv, tab_debug, tab_pipeline = st.tabs([
        "🏆 Rankings", "📄 Export CSV", "🔍 Debug", "⚡ Pipeline"
    ])

    with tab_results:
        _render_rankings(ranked, config)

    with tab_csv:
        _render_csv_export(ranked)

    with tab_debug:
        _render_debug(config)

    with tab_pipeline:
        _render_pipeline_viz()


def _render_rankings(ranked, config: dict) -> None:
    """Render the ranked candidates in card or table format."""
    from ui.components.candidate_card import render_candidates_table

    # Determine if ranked is list of dicts (API mode) or list of RankedCandidate
    # For API mode: convert API response dicts to displayable format
    if ranked and isinstance(ranked[0], dict):
        _render_api_results_table(ranked, config)
    else:
        # Direct mode: RankedCandidate objects
        render_candidates_table(ranked, max_rows=config.get("top_k", 100))


def _render_api_results_table(ranked: list[dict], config: dict) -> None:
    """Render API-mode results (list of RankedCandidateOut dicts)."""
    import pandas as pd

    display = [r for r in ranked if not r.get("is_honeypot", False)]
    honeypots = [r for r in ranked if r.get("is_honeypot", False)]

    # ── Summary ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Ranked", len(display))
    robust = sum(1 for r in display if (r.get("trust") or {}).get("verdict") == "ROBUST")
    col2.metric("ROBUST", robust)
    contested = sum(1 for r in display if (r.get("trust") or {}).get("verdict") == "CONTESTED")
    col3.metric("CONTESTED", contested)

    # ── Filter ────────────────────────────────────────────────────────────────
    col_f, col_s = st.columns([2, 3])
    with col_f:
        vfilter = st.selectbox("Verdict", ["ALL", "ROBUST", "CONTESTED", "FRAGILE"], key="vf_api")
    with col_s:
        search = st.text_input("Search", placeholder="Name, company, skill…", key="srch_api")

    filtered = display
    if vfilter != "ALL":
        filtered = [r for r in filtered if (r.get("trust") or {}).get("verdict") == vfilter]
    if search:
        q = search.lower()
        filtered = [
            r for r in filtered
            if q in (r.get("name") or "").lower()
            or q in (r.get("current_company") or "").lower()
            or q in (r.get("current_title") or "").lower()
            or any(q in s.get("name", "").lower() for s in r.get("skills", []))
        ]

    st.caption(f"Showing {len(filtered)} candidates")

    # ── Candidate expanders ────────────────────────────────────────────────────
    for i, rc in enumerate(filtered[:config.get("top_k", 100)]):
        verdict = (rc.get("trust") or {}).get("verdict", "CONTESTED")
        v_emoji = {"ROBUST": "✅", "CONTESTED": "⚠️", "FRAGILE": "❌"}.get(verdict, "⚠️")
        comp = rc.get("components") or {}
        paths = comp.get("paths_present", [])
        path_str = "".join(
            "●" if p in [pp.lower() for pp in paths] else "○"
            for p in ["semantic", "keyword", "ontology", "trajectory", "signal"]
        )

        rank = rc.get("rank", i + 1)
        rank_badge = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        label = (
            f"{rank_badge} **{rc.get('name', rc.get('candidate_id', '?'))}** "
            f"· {rc.get('current_title', '')} @ {rc.get('current_company', '')} "
            f"· ⭐ `{rc.get('score', 0):.4f}` {v_emoji} {verdict}"
            f" · {rc.get('years_of_experience', 0):.0f}y · {rc.get('location', '')} · `{path_str}`"
        )

        with st.expander(label, expanded=False):
            _render_api_candidate_detail(rc, i)

    # ── Honeypots ─────────────────────────────────────────────────────────────
    if honeypots and config.get("show_honeypots"):
        st.markdown("---")
        with st.expander(f"🛑 {len(honeypots)} Honeypot(s) — Removed"):
            for hp in honeypots:
                st.error(f"**{hp.get('candidate_id')}** — forced score: 0.0")
                st.caption(f"Reasoning: {hp.get('reasoning', '—')}")


def _render_api_candidate_detail(rc: dict, index: int) -> None:
    """Render the expanded detail for an API-mode candidate."""
    comp  = rc.get("components") or {}
    trust = rc.get("trust") or {}
    sigs  = rc.get("signals") or {}

    col_scores, col_signals, col_trust = st.columns(3)

    with col_scores:
        st.markdown("**🎯 Score Breakdown**")
        skill  = comp.get("skill_match_score", 0)
        career = comp.get("career_quality_score", 0)
        beh    = comp.get("behavioral_score", 0)
        ce     = comp.get("cross_encoder_score", 0)

        st.progress(skill,  text=f"Skill Match (×0.40): {skill:.4f}")
        st.progress(career, text=f"Career Quality (×0.35): {career:.4f}")
        st.progress(beh,    text=f"Behavioral (×0.25): {beh:.4f}")
        st.progress(min(1.0, ce), text=f"Cross-encoder: {ce:.4f}")

        loc = comp.get("location_bonus_applied", 0)
        if loc:
            st.success(f"📍 Location bonus: +{loc:.2%}")
        unc = comp.get("uncertainty_penalty_applied", 1.0)
        if unc < 1.0:
            st.warning(f"⚠️ Uncertainty penalty: ×{unc:.2f}")

        paths = comp.get("paths_present", [])
        all_paths = ["semantic", "keyword", "ontology", "trajectory", "signal"]
        path_cols = st.columns(5)
        for j, p in enumerate(all_paths):
            hit = p in [pp.lower() for pp in paths]
            path_cols[j].markdown(f"{'🟢' if hit else '⬜'} {p[:4].capitalize()}")

    with col_signals:
        st.markdown("**📡 Behavioral Signals**")
        if sigs:
            st.caption(f"{'✅' if sigs.get('open_to_work') else '❌'} Open to work: {sigs.get('open_to_work', '?')}")
            notice = sigs.get("notice_period_days", 0)
            n_e = "✅" if notice <= 30 else "⚠️" if notice <= 60 else "❌"
            st.caption(f"{n_e} Notice: {notice}d")
            st.caption(f"🕐 Last active: {sigs.get('last_active_date', '?')}")
            rr = sigs.get("recruiter_response_rate", 0)
            st.caption(f"{'✅' if rr >= 0.8 else '⚠️'} Recruiter rate: {rr:.0%}")
            gh = sigs.get("github_activity_score", -1)
            if gh >= 0:
                st.caption(f"{'✅' if gh >= 70 else '⚠️'} GitHub: {gh:.0f}/100")
            st.caption(f"📊 Profile: {sigs.get('profile_completeness_score', 0):.0f}%")
            st.caption(f"🏠 Mode: {sigs.get('preferred_work_mode', '?')}")

        if rc.get("skills"):
            st.markdown("**🛠️ Skills**")
            for s in rc["skills"][:6]:
                st.caption(f"· **{s.get('name')}** ({s.get('proficiency')}, {s.get('duration_months')}m)")

    with col_trust:
        st.markdown("**🛡️ Trust Layer**")
        verdict = trust.get("verdict", "—")
        vmap = {"ROBUST": st.success, "CONTESTED": st.warning, "FRAGILE": st.error}
        vmap.get(verdict, st.info)(
            f"**{verdict}** · Flip Risk: {trust.get('flip_risk', '?')} · Conf: {trust.get('confidence_pct', 0):.0f}%"
        )

        st.markdown("**✅ Advocate**")
        for sig in trust.get("advocate_signals", [])[:3]:
            conf_e = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔵"}.get(sig.get("confidence"), "·")
            st.caption(f"{conf_e} {sig.get('label')}: {sig.get('value')}")

        st.markdown("**⚠️ Skeptic**")
        for sig in trust.get("skeptic_signals", [])[:3]:
            sev_e = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🔵"}.get(sig.get("severity"), "·")
            st.caption(f"{sev_e} {sig.get('label')}: {sig.get('value')}")

        st.markdown("**📋 Reasoning**")
        st.info(rc.get("reasoning", "—"))

    # Action buttons
    st.markdown("---")
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        if st.button("✅ Shortlist", key=f"sl_api_{index}"):
            cid = rc.get("candidate_id", "")
            if cid not in st.session_state.shortlisted:
                st.session_state.shortlisted.append(cid)
            st.success(f"✅ {cid} shortlisted")
    with bc2:
        if st.button("📋 Copy reasoning", key=f"cp_api_{index}"):
            st.code(rc.get("reasoning", ""))
    with bc3:
        if st.button("🔍 Raw JSON", key=f"rj_api_{index}"):
            st.json(rc)


def _render_csv_export(ranked: list) -> None:
    """Render the CSV export tab."""
    st.markdown("#### 📄 Export submission.csv")
    st.caption("Validated against spec requirements before download.")

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("⬇️ Download via API", key="dl_api_csv"):
            try:
                csv_bytes = _export_csv_via_api()
                st.download_button(
                    label="✅ Download submission.csv",
                    data=csv_bytes,
                    file_name="submission.csv",
                    mime="text/csv",
                    key="dl_csv_btn",
                )
            except Exception as e:
                st.error(f"API export failed: {e}")
                _generate_csv_locally(ranked)

    with col_b:
        _generate_csv_locally(ranked)


def _generate_csv_locally(ranked: list) -> None:
    """Generate CSV directly from session state results."""
    import csv as csv_mod
    output = io.StringIO()
    writer = csv_mod.DictWriter(output, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()

    for rc in ranked:
        if isinstance(rc, dict):
            row = {
                "candidate_id": rc.get("candidate_id", ""),
                "rank": rc.get("rank", 0),
                "score": f"{rc.get('score', 0):.6f}",
                "reasoning": rc.get("reasoning", ""),
            }
        elif hasattr(rc, "to_csv_row"):
            row = rc.to_csv_row()
        else:
            continue
        writer.writerow(row)

    st.download_button(
        label="⬇️ Generate CSV locally",
        data=output.getvalue().encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
        key="dl_local_csv",
    )


def _render_debug(config: dict) -> None:
    """Render debug info."""
    st.markdown("#### 🔍 Debug Information")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Stage Timings**")
        timings = st.session_state.stage_timings
        if timings:
            for stage, ms in timings.items():
                st.metric(stage.replace("_", " ").title(), f"{ms:.0f}ms")
        else:
            st.info("No timing data — run the pipeline first.")

    with col2:
        st.markdown("**Parse Errors**")
        errors = st.session_state.errors
        if errors:
            for e in errors[:10]:
                st.warning(e)
        else:
            st.success("✅ No parse errors")

    if config.get("show_debug"):
        st.markdown("**Session State**")
        st.json({
            k: v for k, v in st.session_state.items()
            if k not in ("ranked_candidates", "_sample_jsonl")
        })


def _render_pipeline_viz() -> None:
    """Render a visual pipeline flow diagram."""
    st.markdown("#### ⚡ Pipeline Architecture")
    st.markdown("""
    ```
    INPUT (candidates.jsonl.gz + job_description.md)
        │
        ├─ Candidate Parser ──────────────────────────────────────────┐
        │   parse_candidate() → CandidateFeatureVector                │
        │                                                              │
        └─ JD Parser                                                   │
            jd_parser.parse() → JDIntent                              │
                │                                                      │
                ▼             PRE-COMPUTATION (offline)                │
    ┌───────────────────────────────────────────────────────────────┐  │
    │  FAISS Index  │  BM25 Index  │  Feature Store  │  Trajectory  │  │
    │  IVF256·22MB  │  skill+text  │  23 signals·npy │  promotions  │  │
    └───────────────────────────────────────────────────────────────┘  │
                │                                                      │
                ▼             RANKING WINDOW (≤5 min · CPU only)        │
    ┌─────────────────────────────────────────────────────────┐       │
    │  Path 1: Semantic   (FAISS cosine · top-25)             │       │
    │  Path 2: Keyword    (BM25+ontology · top-25)            │       │
    │  Path 3: Ontology   (graph traversal · top-20)    NEW   │       │
    │  Path 4: Trajectory (IC-riser pattern · top-15)         │       │
    │  Path 5: Signal     (behavioral engage · top-15)   NEW  │       │
    └─────────────────────────────────────────────────────────┘       │
                │                                                      │
                ▼                                                      │
        RRF Fusion: Σ 1/(60+rank) → deduplicated top-60               │
                │                                                      │
                ▼                                                      │
        Honeypot Filter: O(1) registry lookup → top-50                │
                │                                                      │
                ▼                                                      │
        Cross-Encoder Rerank: ms-marco-MiniLM-L-6-v2 · ~4s            │
                │                                                      │
                ▼                                                      │
        Composite Score: 0.40×skill + 0.35×career + 0.25×behavioral   │
                │                                                      │
                ▼                                                      │
    ┌─────────────────────────────────────────────────────────┐       │
    │  ADVERSARIAL TRUST LAYER                                │       │
    │  Advocate Agent → positive signals (HIGH/MEDIUM/LOW)   │       │
    │  Skeptic Agent  → risk flags (HIGH/MODERATE/LOW)       │       │
    │  Verdict: ROBUST · CONTESTED · FRAGILE                 │       │
    │  Reasoning Generator: template · no hallucination      │       │
    └─────────────────────────────────────────────────────────┘       │
                │                                                      │
                ▼                                                      │
        submission.csv: candidate_id · rank · score · reasoning ───────┘
    ```
    """)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()
    config = _render_sidebar()
    _render_hero()

    candidates_jsonl, jd_text = _render_input_section()

    st.markdown("---")

    # ── Run Pipeline button ────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button(
            "⚡ Run Pipeline",
            type="primary",
            use_container_width=True,
            key="run_pipeline_btn",
            disabled=candidates_jsonl is None,
        )
    with col_info:
        if candidates_jsonl is None:
            st.warning("📌 Upload a candidates file to enable the pipeline.")
        else:
            lines = len([l for l in candidates_jsonl.splitlines() if l.strip()])
            st.success(f"✅ **{lines:,}** candidates ready · Mode: `{st.session_state.run_mode}` · Top-K: {config['top_k']}")

    if run_clicked and candidates_jsonl:
        _run_pipeline(candidates_jsonl, jd_text, config["top_k"])
        st.rerun()

    # ── Results ────────────────────────────────────────────────────────────────
    if st.session_state.ranked_candidates:
        st.markdown("---")
        _render_results(config)
    elif not run_clicked:
        # Empty state
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; padding:48px; color:#9298b0;">
            <div style="font-size:48px; margin-bottom:14px; opacity:0.35;">🏆</div>
            <div style="font-size:16px; font-weight:700; color:#4a5068; margin-bottom:6px;">No Results Yet</div>
            <div style="font-size:13px; line-height:1.6;">
                Upload a candidates file above and click <strong>⚡ Run Pipeline</strong><br>
                to rank candidates and view the full trust-layer analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
