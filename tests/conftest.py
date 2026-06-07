"""
tests/conftest.py — Shared pytest fixtures for the Redrob ranking system.

This file is the test contract between Dev A and Dev B. Both teammates
write tests that import from these fixtures. Fixtures defined here are:

  sample_candidates_raw   — 50 real candidates from sample_candidates.json
                            (raw dicts, before parsing)

  synthetic_honeypots_raw — 10 synthetic impossible profiles, one per
                            honeypot rule defined in config.py

  mock_jd_intent          — A JDIntent object matching job_description.md

  good_candidate_ids      — candidate_ids that should rank in top-10
                            (manually verified from sample data)

  bad_candidate_ids       — candidate_ids that should NOT rank in top-20
                            (marketing managers, civil engineers, etc.)

Session-scoped fixtures are expensive to build and safe to share across tests.
Function-scoped fixtures are cheap and give test isolation.

Dependencies:
  - config.py
  - pipeline/schemas.py
  - data/sample_candidates.json (copied from hackathon bundle)

No imports from our own indexing/scoring/retrieval modules.
conftest.py must be importable before those modules exist.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

# Ensure project root is on sys.path for all test imports.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from pipeline.schemas import (
    JDIntent,
    RankedCandidate,
    validate_candidate_id,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_JSON_PATH = PROJECT_ROOT / "data" / "sample_candidates.json"

# Candidates from sample that should rank well for this JD.
# CAND_0000031: Ela Singh, Recommendation Systems Engineer @ Swiggy
#   — 6yr, FAISS/Pinecone/Embeddings/Sentence-Transformers expert, product-co
#   — open_to_work=True, last_active=2026-05-24, willing_to_relocate=True
# CAND_0000043: Cloud Engineer @ Swiggy
#   — Elasticsearch+OpenSearch+Haystack+LangChain, 8.3yr, product-co
EXPECTED_GOOD_CANDIDATE_IDS: list[str] = [
    "CAND_0000031",   # Primary target: Recsys engineer, Swiggy, strong retrieval skills
    "CAND_0000043",   # Cloud engineer with strong search skills
    "CAND_0000014",   # Frontend engineer with FAISS+OpenSearch skills
    "CAND_0000038",   # Java developer @ Swiggy with Weaviate
]

# Candidates that are clearly wrong fits and should not appear in top-20.
# Marketing managers, civil engineers, accountants — present in sample.
EXPECTED_BAD_CANDIDATE_IDS: list[str] = [
    "CAND_0000002",   # Operations Manager @ Wipro — no ML at all
    "CAND_0000003",   # Customer Support @ TCS — 1.1yr, wrong domain
    "CAND_0000004",   # Marketing Manager @ Dunder Mifflin — completely wrong
    "CAND_0000005",   # Accountant @ Stark Industries — no technical skills
    "CAND_0000007",   # Civil Engineer @ Wipro — wrong domain entirely
    "CAND_0000013",   # Civil Engineer @ Globex — UAE-based, wrong skills
]

# Candidate with a consulting-only background — should be penalised.
CONSULTING_ONLY_CANDIDATE_ID: str = "CAND_0000002"  # Wipro operations manager


# ─────────────────────────────────────────────────────────────────────────────
# RAW SAMPLE DATA FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_candidates_raw() -> list[dict[str, Any]]:
    """
    Load all 50 candidates from sample_candidates.json as raw dicts.

    Session-scoped: loaded once per test run. Tests must not mutate this.
    If you need to modify a candidate, use copy.deepcopy().
    """
    if not SAMPLE_JSON_PATH.exists():
        pytest.skip(
            f"sample_candidates.json not found at {SAMPLE_JSON_PATH}. "
            "Copy it from the hackathon bundle to data/sample_candidates.json."
        )

    with open(SAMPLE_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "sample_candidates.json must be a JSON array"
    assert len(data) == 50, f"Expected 50 candidates, got {len(data)}"

    # Validate all IDs are correct format
    for c in data:
        assert validate_candidate_id(c["candidate_id"]), (
            f"Bad candidate_id in sample: {c['candidate_id']}"
        )

    return data


@pytest.fixture(scope="session")
def good_candidate_raw(sample_candidates_raw) -> dict[str, Any]:
    """
    Return CAND_0000031 (Ela Singh @ Swiggy) — our primary 'ground truth' good match.
    Used in tests that need a single strong-positive example.
    """
    c = next(
        (c for c in sample_candidates_raw if c["candidate_id"] == "CAND_0000031"),
        None
    )
    assert c is not None, "CAND_0000031 not found in sample_candidates.json"
    return c


@pytest.fixture(scope="session")
def bad_candidate_raw(sample_candidates_raw) -> dict[str, Any]:
    """
    Return CAND_0000004 (Marketing Manager @ Dunder Mifflin) — a clear wrong fit.
    Used in tests that need a single strong-negative example.
    """
    c = next(
        (c for c in sample_candidates_raw if c["candidate_id"] == "CAND_0000004"),
        None
    )
    assert c is not None, "CAND_0000004 not found in sample_candidates.json"
    return c


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC HONEYPOT FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def _base_honeypot_raw() -> dict[str, Any]:
    """
    Return a base candidate dict that is a valid, non-honeypot profile.
    Honeypot fixtures mutate this to trigger exactly one rule each.
    """
    return {
        "candidate_id": "CAND_9990000",   # overwritten per honeypot
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "ML Engineer",
            "summary": "Machine learning engineer with experience in embeddings and retrieval.",
            "location": "Noida, Uttar Pradesh",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "ML Engineer",
            "current_company": "TestCo",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "TestCo",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 65,
                "is_current": True,
                "industry": "Software",
                "company_size": "201-500",
                "description": "Built recommendation systems and vector search pipelines.",
            }
        ],
        "education": [
            {
                "institution": "IIT Delhi",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "grade": "8.5 CGPA",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 72},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": 85.0,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 50,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 5.0,
            "skill_assessment_scores": {},
            "connection_count": 300,
            "endorsements_received": 50,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 25.0, "max": 45.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 60.0,
            "search_appearance_30d": 200,
            "saved_by_recruiters_30d": 10,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.7,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


@pytest.fixture(scope="session")
def synthetic_honeypots_raw() -> list[dict[str, Any]]:
    """
    10 synthetic honeypot candidates — one or more per detection rule.

    Rules covered (per config.py):
      Rule 1: experience at company predates plausible founding (2 candidates)
      Rule 2: expert proficiency + 0 duration_months (2 candidates)
      Rule 3: salary min > max (2 candidates)
      Rule 4: low completeness + skill stuffing (2 candidates)
      Rule 5: YOE wildly inconsistent with career history (2 candidates)

    Each honeypot is labelled with which rule(s) it triggers.
    """
    honeypots = []

    # ── Rule 1a: duration_months impossible relative to company tenure ──────
    # Career shows 120 months at a company, but company is only ~2yr old
    h1a = _base_honeypot_raw()
    h1a["candidate_id"] = "CAND_9990001"
    h1a["profile"]["years_of_experience"] = 10.0
    h1a["career_history"] = [
        {
            "company": "NewStartup2024",
            "title": "Senior ML Engineer",
            "start_date": "2014-01-01",   # Claims started 10 years ago
            "end_date": None,
            "duration_months": 120,        # 10 years at a company founded ~2 years ago
            "is_current": True,
            "industry": "Software",
            "company_size": "51-200",
            "description": "Built ML systems.",
        }
    ]
    h1a["_honeypot_rule"] = "Rule1a: duration impossible relative to start_date"
    honeypots.append(h1a)

    # ── Rule 1b: total career duration massively exceeds stated YOE ──────────
    h1b = _base_honeypot_raw()
    h1b["candidate_id"] = "CAND_9990002"
    h1b["profile"]["years_of_experience"] = 3.0   # Claims only 3 years
    h1b["career_history"] = [
        {
            "company": "CompanyA",
            "title": "Engineer",
            "start_date": "2010-01-01",
            "end_date": "2016-01-01",
            "duration_months": 72,    # 6 years
            "is_current": False,
            "industry": "Software",
            "company_size": "201-500",
            "description": "Engineering work.",
        },
        {
            "company": "CompanyB",
            "title": "Senior Engineer",
            "start_date": "2016-01-01",
            "end_date": None,
            "duration_months": 72,    # Another 6 years (total = 12yr vs 3yr claimed)
            "is_current": True,
            "industry": "Software",
            "company_size": "1001-5000",
            "description": "Senior engineering.",
        },
    ]
    h1b["_honeypot_rule"] = "Rule1b: career months >> stated YOE"
    honeypots.append(h1b)

    # ── Rule 2a: expert proficiency with 0 duration_months ───────────────────
    h2a = _base_honeypot_raw()
    h2a["candidate_id"] = "CAND_9990003"
    h2a["skills"] = [
        {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 0},
        {"name": "FAISS", "proficiency": "expert", "endorsements": 40, "duration_months": 0},
        {"name": "Pinecone", "proficiency": "expert", "endorsements": 35, "duration_months": 0},
        {"name": "Weaviate", "proficiency": "expert", "endorsements": 30, "duration_months": 0},
    ]
    h2a["_honeypot_rule"] = "Rule2a: multiple expert skills with 0 duration"
    honeypots.append(h2a)

    # ── Rule 2b: single expert skill, 0 months, high endorsements ────────────
    h2b = _base_honeypot_raw()
    h2b["candidate_id"] = "CAND_9990004"
    h2b["skills"] = [
        # Mix of normal + one suspicious expert+0 combo
        {"name": "Python", "proficiency": "advanced", "endorsements": 20, "duration_months": 48},
        {"name": "Embeddings", "proficiency": "expert", "endorsements": 99, "duration_months": 0},
        {"name": "Vector Search", "proficiency": "expert", "endorsements": 95, "duration_months": 0},
    ]
    h2b["_honeypot_rule"] = "Rule2b: expert+0months with suspiciously high endorsements"
    honeypots.append(h2b)

    # ── Rule 3a: salary min > max (clear impossible range) ───────────────────
    h3a = _base_honeypot_raw()
    h3a["candidate_id"] = "CAND_9990005"
    h3a["redrob_signals"]["expected_salary_range_inr_lpa"] = {
        "min": 80.0,   # min > max — impossible
        "max": 20.0,
    }
    h3a["_honeypot_rule"] = "Rule3a: salary min > max"
    honeypots.append(h3a)

    # ── Rule 3b: salary min slightly > max (edge case) ────────────────────────
    h3b = _base_honeypot_raw()
    h3b["candidate_id"] = "CAND_9990006"
    h3b["redrob_signals"]["expected_salary_range_inr_lpa"] = {
        "min": 45.1,   # min just barely > max
        "max": 45.0,
    }
    h3b["_honeypot_rule"] = "Rule3b: salary min barely > max"
    honeypots.append(h3b)

    # ── Rule 4a: very low completeness + skills stuffing ─────────────────────
    h4a = _base_honeypot_raw()
    h4a["candidate_id"] = "CAND_9990007"
    h4a["redrob_signals"]["profile_completeness_score"] = 15.0   # Very low
    h4a["profile"]["summary"] = ""
    h4a["profile"]["headline"] = ""
    # But 20 skills listed — stuffing
    h4a["skills"] = [
        {"name": f"Skill{i}", "proficiency": "advanced", "endorsements": 50, "duration_months": 24}
        for i in range(20)
    ]
    h4a["_honeypot_rule"] = "Rule4a: completeness=15 + 20 skills stuffed"
    honeypots.append(h4a)

    # ── Rule 4b: zero completeness + all expert skills ────────────────────────
    h4b = _base_honeypot_raw()
    h4b["candidate_id"] = "CAND_9990008"
    h4b["redrob_signals"]["profile_completeness_score"] = 5.0
    h4b["skills"] = [
        {"name": s, "proficiency": "expert", "endorsements": 99, "duration_months": 60}
        for s in [
            "Python", "FAISS", "Pinecone", "Weaviate", "Qdrant",
            "Elasticsearch", "OpenSearch", "BM25", "Embeddings",
            "Vector Search", "LLM Fine-tuning", "RAG", "NLP",
            "Sentence Transformers", "Cross-encoders", "NDCG",
        ]
    ]
    h4b["_honeypot_rule"] = "Rule4b: completeness=5 + 16 expert AI skills"
    honeypots.append(h4b)

    # ── Rule 5a: YOE massively understated vs career span ────────────────────
    h5a = _base_honeypot_raw()
    h5a["candidate_id"] = "CAND_9990009"
    h5a["profile"]["years_of_experience"] = 1.0   # Claims 1 year
    h5a["career_history"] = [
        {
            "company": "OldCo",
            "title": "Engineer",
            "start_date": "2010-06-01",
            "end_date": "2020-06-01",
            "duration_months": 120,  # 10 years at one company alone
            "is_current": False,
            "industry": "Software",
            "company_size": "501-1000",
            "description": "Long career.",
        }
    ]
    h5a["_honeypot_rule"] = "Rule5a: claims 1yr YOE but career_history shows 10yr"
    honeypots.append(h5a)

    # ── Rule 5b: YOE massively overstated vs career span ─────────────────────
    h5b = _base_honeypot_raw()
    h5b["candidate_id"] = "CAND_9990010"
    h5b["profile"]["years_of_experience"] = 20.0  # Claims 20 years
    h5b["career_history"] = [
        {
            "company": "NewCo",
            "title": "Junior Engineer",
            "start_date": "2025-01-01",
            "end_date": None,
            "duration_months": 5,   # Only 5 months of career history
            "is_current": True,
            "industry": "Software",
            "company_size": "11-50",
            "description": "New grad role.",
        }
    ]
    h5b["_honeypot_rule"] = "Rule5b: claims 20yr YOE but career_history shows 5 months"
    honeypots.append(h5b)

    assert len(honeypots) == 10, f"Expected 10 honeypots, built {len(honeypots)}"

    # Verify all honeypot IDs are unique and validly formatted
    ids = [h["candidate_id"] for h in honeypots]
    assert len(set(ids)) == len(ids), "Duplicate honeypot IDs"
    for hid in ids:
        assert validate_candidate_id(hid), f"Bad honeypot ID format: {hid}"

    return honeypots


@pytest.fixture(scope="session")
def all_raw_candidates(sample_candidates_raw, synthetic_honeypots_raw) -> list[dict]:
    """
    Combined pool: 50 real + 10 synthetic honeypots = 60 candidates.
    Used for tests that need a realistic mixed pool.
    """
    return sample_candidates_raw + synthetic_honeypots_raw


# ─────────────────────────────────────────────────────────────────────────────
# JD INTENT FIXTURE
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mock_jd_intent() -> JDIntent:
    """
    JDIntent built from job_description.md (Senior AI Engineer @ Redrob).

    This is a manually curated intent object — the ground truth that
    pipeline/jd_parser.py should reproduce when it runs on the actual JD.

    Used in retrieval tests (does path return good candidates?),
    scoring tests (does CAND_0000031 score high?), and trust tests.
    """
    return JDIntent(
        # ── Required skills (from JD "Things you absolutely need") ──────────
        required_skills=[
            "embeddings",
            "retrieval",
            "vector search",
            "sentence transformers",
            "faiss",
            "pinecone",
            "weaviate",
            "qdrant",
            "milvus",
            "opensearch",
            "elasticsearch",
            "python",
            "ranking",
            "ndcg",
            "mrr",
            "evaluation framework",
            "information retrieval",
        ],
        # ── Nice-to-have (from JD "Things we'd like you to have") ───────────
        nice_to_have_skills=[
            "lora",
            "qlora",
            "peft",
            "fine-tuning llms",
            "xgboost",
            "learning to rank",
            "recommendation systems",
            "distributed systems",
            "mlops",
            "mlflow",
            "weights & biases",
        ],
        # ── Disqualifiers (wrong domain) ────────────────────────────────────
        disqualifier_skills=[
            "computer vision",
            "speech recognition",
            "robotics",
            "image classification",
            "object detection",
            "yolo",
            "cnn",
        ],
        # ── BM25 expanded terms (will be built by query_expander.py) ────────
        expanded_required=[
            "embeddings", "dense retrieval", "vector database",
            "sentence transformers", "bi-encoder", "cross-encoder",
            "faiss", "pinecone", "weaviate", "qdrant", "milvus",
            "opensearch", "elasticsearch", "bm25", "hybrid search",
            "python", "ranking", "ndcg", "mrr", "map", "information retrieval",
            "recsys", "recommendation", "search infrastructure",
        ],
        # ── Experience band ──────────────────────────────────────────────────
        yoe_min=4.0,
        yoe_max=12.0,
        yoe_ideal_min=5.0,
        yoe_ideal_max=9.0,
        # ── Location ─────────────────────────────────────────────────────────
        preferred_locations=[
            "noida", "pune", "delhi", "gurgaon",
            "hyderabad", "mumbai", "bangalore", "bengaluru",
        ],
        relocation_accepted=True,
        # ── Business rules ───────────────────────────────────────────────────
        disqualify_consulting_only=True,
        disqualify_no_production=True,
        # ── Text for cross-encoder ────────────────────────────────────────────
        raw_text=(
            "Senior AI Engineer at Redrob AI. "
            "Required: embeddings-based retrieval, vector databases (FAISS, Pinecone, "
            "Weaviate, Qdrant, Elasticsearch), strong Python, evaluation frameworks "
            "(NDCG, MRR, MAP). 5-9 years experience at product companies. "
            "No consulting-only backgrounds. Location: Pune/Noida/Delhi NCR preferred."
        ),
        embedding=None,   # Set by jd_parser.py when it runs
    )


# ─────────────────────────────────────────────────────────────────────────────
# RANKED RESULT FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_ranked_list() -> list[RankedCandidate]:
    """
    A valid submission-ready list of 100 RankedCandidate objects.
    Scores are non-increasing, ranks are 1–100, IDs are unique and valid.

    Used to test validate_ranked_list() and CSV writing logic.
    """
    candidates = []
    for i in range(100):
        rank = i + 1
        # Use real IDs for ranks 1–50, synthetic for 51–100
        if rank <= 50:
            cid = f"CAND_{rank:07d}"
        else:
            cid = f"CAND_{rank + 1000:07d}"

        score = max(0.01, 1.0 - (i * 0.009))   # Non-increasing, all ≥ 0.01

        candidates.append(
            RankedCandidate(
                candidate_id=cid,
                rank=rank,
                final_score=round(score, 6),
                reasoning=f"Candidate at rank {rank} selected for AI engineering skills.",
            )
        )
    return candidates


@pytest.fixture
def invalid_ranked_list_nonmonotonic() -> list[RankedCandidate]:
    """A 2-candidate list where scores are NOT non-increasing. For negative testing."""
    return [
        RankedCandidate("CAND_0000001", 1, 0.50, "Lower ranked."),
        RankedCandidate("CAND_0000002", 2, 0.80, "Higher score at lower rank — wrong."),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG FIXTURE
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app_config():
    """
    Return the config module for tests that need to inspect constants.
    Session-scoped — config does not change during a test run.
    """
    return config


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: RAW → DICT ACCESSORS (for tests that don't want to run the parser)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def good_candidate_ids() -> list[str]:
    """IDs of candidates expected to rank well. For retrieval recall tests."""
    return EXPECTED_GOOD_CANDIDATE_IDS.copy()


@pytest.fixture(scope="session")
def bad_candidate_ids() -> list[str]:
    """IDs of candidates that should NOT appear in top-20. For precision tests."""
    return EXPECTED_BAD_CANDIDATE_IDS.copy()


@pytest.fixture(scope="session")
def honeypot_ids(synthetic_honeypots_raw) -> set[str]:
    """Set of honeypot candidate_ids. For honeypot detection tests."""
    return {h["candidate_id"] for h in synthetic_honeypots_raw}