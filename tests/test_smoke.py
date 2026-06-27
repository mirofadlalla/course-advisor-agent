"""
tests/test_smoke.py — Smoke Tests

Validates that:
  - Settings initialises correctly from environment variables
  - All modules are importable without side effects
  - The FastAPI app object exists and /health returns 200 without
    running the full lifespan (no embedding model, no index)
"""

import os
import pytest
from fastapi.testclient import TestClient


# ── Settings ──────────────────────────────────────────────────────────────────

def test_settings_loads_from_env():
    """Settings must initialise from the env vars set in CI."""
    from app.config import Settings

    s = Settings(groq_api_key="test-key")
    assert s.groq_api_key == "test-key"
    assert s.vector_store_backend == "simple"
    assert s.chunk_size == 512
    assert s.chunk_overlap == 64
    assert s.retrieval_top_k == 8
    assert s.rerank_top_k == 4


def test_settings_defaults():
    """Check that all expected fields have sensible defaults."""
    from app.config import Settings

    s = Settings(groq_api_key="x")
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_device == "cpu"
    assert s.reranker_backend == ""
    assert s.is_hf_spaces is False


# ── Module imports ─────────────────────────────────────────────────────────────

def test_import_schemas():
    from app.schemas.api import ChatRequest, ChatResponse
    from app.schemas.agent import AgentResponse
    from app.schemas.search import SearchResult
    from app.schemas.course import Course


def test_import_ingestion():
    from app.ingestion.chunker import SemanticChunker
    from app.ingestion.parsers import (
        MarkdownStructuredParser,
        JSONFlatParser,
        CompositeParser,
    )


def test_import_retrieval():
    from app.retrieval.base import BaseRetriever
    from app.retrieval.reranker import NoOpReranker, RerankerFactory
    from app.retrieval.hybrid_retriever import HybridRetriever


def test_import_repositories():
    from app.repositories.base import IKnowledgeRepository


def test_import_prompts():
    from app.prompts import SYSTEM_PROMPT
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0


# ── FastAPI health endpoint (no lifespan) ─────────────────────────────────────

def test_health_endpoint():
    """
    /health must return 200 immediately — it must NOT depend on the RAG pipeline.
    We skip the lifespan so no embedding model or index is loaded.
    """
    from app.main import app

    # TestClient with use_lifespan=False skips startup (no heavy DI graph)
    with TestClient(app, raise_server_exceptions=True) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "app" in body
