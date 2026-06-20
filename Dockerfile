# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — RAGnarok with LLM reranker (CPU only)
#
# Build:  docker build -t ragnarok .
# Run:    docker run --rm \
#           -v $(pwd)/data:/app/data \
#           -v $(pwd)/output:/app/output \
#           ragnarok
#
# The GGUF model must be pre-downloaded into models/ before building.
# Run locally first:
#   python precompute.py --candidates data/candidates.jsonl.gz
# This downloads the model and builds all indexes.
# Then docker build will COPY models/ into the image.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# llama-cpp-python needs a C/C++ compiler at build time
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────────
COPY requirements.txt .

# Install llama-cpp-python CPU build first (separate to avoid conflicts)
RUN pip install --no-cache-dir \
    llama-cpp-python==0.3.2 \
    huggingface-hub==0.24.0

# Install the rest of your deps
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY config.py .
COPY rank.py .
COPY pipeline/ pipeline/
COPY indexing/ indexing/
COPY retrieval/ retrieval/
COPY scoring/ scoring/
COPY trust/ trust/
COPY ontology/ ontology/

# ── Copy pre-built artifacts (built by precompute.py outside Docker) ──────────
# Indexes: built from candidates.jsonl.gz by precompute.py
COPY data/indexes/ data/indexes/

# Models: GGUF downloaded by precompute.py / LLMReranker.download_model()
COPY models/ models/

# spaCy model: downloaded by precompute.py
# If you ran: python -m spacy download en_core_web_sm
# the model is in site-packages — it's copied automatically above via pip.

# HuggingFace model cache (bi-encoder + cross-encoder)
# These were downloaded during precompute.py and cached in ~/.cache/huggingface/
# Copy from your local cache so Docker doesn't need network:
# Option A (recommended): copy the cache into image
# COPY .hf_cache/ /root/.cache/huggingface/
# Option B: mount at runtime:
# docker run -v ~/.cache/huggingface:/root/.cache/huggingface ragnarok

# ── Runtime ───────────────────────────────────────────────────────────────────
# Output is written to /app/output/submission.csv
# Mount a host directory to retrieve it:
#   docker run -v $(pwd)/output:/app/output ragnarok

CMD ["python", "rank.py", \
     "--input",  "data/candidates.jsonl.gz", \
     "--output", "output/submission.csv", \
     "--top-k",  "100"]