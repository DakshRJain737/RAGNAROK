# ─────────────────────────────────────────────────────────────────────────────
# RAGnarok — Candidate Ranking System
# Base image: Python 3.11-slim — matches local venv (Python 3.11.2)
# CPU-only, no GPU needed
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential + cmake: needed to compile llama-cpp-python from source
# git: needed by some pip packages during install
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer separately.
# The torch CPU wheel uses an extra index URL — pip handles it automatically
# from the --extra-index-url flag embedded in requirements.txt via PEP 508.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ── spaCy language model ──────────────────────────────────────────────────────
RUN python -m spacy download en_core_web_sm

# ── Pre-download sentence-transformer models into the image ───────────────────
# These are baked in so there is zero network traffic at ranking time.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# ── Copy source code ──────────────────────────────────────────────────────────
COPY config.py .
COPY rank.py .
COPY precompute.py .
COPY precompute_llm.py .
COPY build_indexes.py .
COPY job_description.md .
COPY parsed_job_description.json .
COPY submission_metadata.yaml .
COPY pipeline/   ./pipeline/
COPY indexing/   ./indexing/
COPY ontology/   ./ontology/
COPY retrieval/  ./retrieval/
COPY scoring/    ./scoring/
COPY trust/      ./trust/
COPY scripts/    ./scripts/

# ── Copy pre-built indexes (built by precompute.py, checked in to repo) ───────
COPY data/indexes/ ./data/indexes/

# ── Copy the LLM model (already downloaded locally — no HuggingFace needed) ──
# The GGUF file is ~1 GB; baking it in avoids any download at runtime.
COPY models/ ./models/

# ── Create output directory ───────────────────────────────────────────────────
RUN mkdir -p output

# ── Default command ───────────────────────────────────────────────────────────
# Reads  : data/candidates.jsonl   (mount your file here at runtime)
# Writes : output/submission.csv   (mount a host directory here to retrieve it)
CMD ["python", "rank.py", \
     "--input",  "data/candidates.jsonl", \
     "--output", "output/submission.csv", \
     "--top-k",  "100"]