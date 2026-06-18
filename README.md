# RAGnarok ATS — Intelligent Candidate Ranking System

> **Redrob Hackathon · DEV A: Krishna Zalavadiya**

Enterprise-grade candidate ranking pipeline: 5-path retrieval → RRF fusion → cross-encoder rerank → composite scoring → adversarial trust layer.

---

## ⚡ Quick Start (Single Command)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser — the full ATS UI loads instantly.

---

## 🖥️ UI

The UI is `ui/ats_platform.html` served directly by the FastAPI backend.

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | ATS Platform UI |
| `http://localhost:8000/docs` | Swagger / OpenAPI docs |
| `http://localhost:8000/health` | Pipeline health check |
| `http://localhost:8000/redoc` | ReDoc API reference |

---

## 📋 Prerequisites

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python precompute.py --candidates data/candidates.jsonl.gz
```

---

## 🔁 Pipeline Flow

```
candidates.jsonl
      │
      ▼
  5 Retrieval Paths (parallel)
  ├─ Path 1: Semantic  (FAISS · all-MiniLM-L6-v2)
  ├─ Path 2: Keyword   (BM25 + ontology expansion)
  ├─ Path 3: Ontology  (domain-transfer graph)
  ├─ Path 4: Trajectory (IC-riser career pattern)
  └─ Path 5: Signal    (behavioral engagement)
      │
      ▼
  RRF Fusion → top-60
      │
      ▼
  Honeypot Filter (O(1) registry)
      │
      ▼
  Cross-Encoder Rerank · top-50 (ms-marco-MiniLM-L-6-v2)
      │
      ▼
  Composite Score = 0.40×skill + 0.35×career + 0.25×behavioral
      │
      ▼
  Adversarial Trust Layer (Advocate · Skeptic · Verdict)
      │
      ▼
  submission.csv · top-100
```

---

## 🏃 Running the Pipeline via UI

1. Start the server: `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`
2. Open **http://localhost:8000** in your browser
3. Click the **📂** upload button and load your `candidates.jsonl`
4. Click **⚡ Run Pipeline**
5. Browse ranked results and click **⬇️ Export CSV** to download `submission.csv`

---

## 🏃 Running the Pipeline via CLI

```bash
python rank.py --candidates data/sample_candidates.json --out output/test.csv
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | ATS Platform UI |
| `GET` | `/health` | Pipeline health |
| `POST` | `/rank` | Run ranking pipeline |
| `GET` | `/results` | Last run results |
| `POST` | `/export/csv` | Export submission.csv |

### POST /rank — Example

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{
    "candidates_jsonl": "{\"candidate_id\":\"CAND_001\", ...}",
    "jd_text": null,
    "top_k": 100
  }'
```

---

## 🏗️ Architecture

```
RAGnarok/
├── api/
│   ├── main.py          # FastAPI app — serves UI + API
│   ├── middleware.py     # CORS, rate limit, timing, error handling
│   ├── schemas.py        # Pydantic models
│   └── routes/
│       ├── health.py     # GET /health
│       └── rank.py       # POST /rank, GET /results, POST /export/csv
├── pipeline/
│   ├── candidate_parser.py
│   ├── jd_parser.py
│   └── runner.py
├── ui/
│   └── ats_platform.html  # Full ATS UI (served at GET /)
├── requirements.txt
├── precompute.py
└── rank.py
```

---

## 📊 Constraints Met

| Constraint | Status |
|------------|--------|
| CPU-only (no GPU) | ✅ |
| ≤5 minute ranking window | ✅ (~15s estimated) |
| No network during ranking | ✅ |
| 100K candidate pool | ✅ |
| Honeypot rate <10% | ✅ |
| submission.csv monotonic scores | ✅ |
