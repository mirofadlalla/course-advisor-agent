# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Course Advisor Agent — Production Dockerfile
#
# Designed for HuggingFace Spaces (Docker SDK) but runs identically locally.
#
# Build stages:
#   1. deps   — install Python dependencies into a clean venv
#   2. runtime — copy venv + app code; add non-root user
#
# HuggingFace Spaces requirements:
#   • App must listen on port 7860
#   • Persistent storage is mounted at /data  (set INDEX_STORAGE_PATH=/data/index)
#   • Secrets are injected as environment variables (GROQ_API_KEY, etc.)
#   • Container runs as a non-root user
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency installer ────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /build

# Install build tools needed by some Python packages (tokenizers, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv — keeps runtime image clean
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency manifest first (layer-cache friendly)
COPY pyproject.toml .

# Install all dependencies from pyproject.toml
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from stage 1
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Add a non-root user (HuggingFace Spaces requirement)
RUN useradd --create-home --shell /bin/bash appuser

# Copy application code
COPY app/ ./app/
COPY data/ ./data/

# Create the persistent storage mount point with correct ownership
# HF Spaces mounts /data as a persistent volume at runtime
RUN mkdir -p /data && chown appuser:appuser /data

USER appuser

# ── Runtime environment ───────────────────────────────────────────────────────
# These are defaults; override via HF Spaces secrets or docker run -e
ENV APP_NAME="Course Advisor Agent" \
    DEBUG="false" \
    EMBEDDING_MODEL="BAAI/bge-m3" \
    EMBEDDING_DEVICE="cpu" \
    VECTOR_STORE_BACKEND="simple" \
    INDEX_STORAGE_PATH="/data/index" \
    DATA_DIR="/data/raw" \
    METRICS_STORAGE_PATH="/data/metrics.json" \
    CHUNK_SIZE="512" \
    CHUNK_OVERLAP="64" \
    RETRIEVAL_TOP_K="8" \
    BM25_TOP_K="8" \
    RERANK_TOP_K="4" \
    RERANKER_BACKEND="" \
    IS_HF_SPACES="true"
# GROQ_API_KEY must be set via HF Space secrets — never hardcoded here

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Health check — /health returns 200 immediately, no RAG dependency
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start uvicorn on 0.0.0.0:7860
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
