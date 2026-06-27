"""
tests/test_retrieval.py — Retrieval Layer Unit Tests

Tests BM25Retriever, HybridRetriever (RRF fusion), Reranker variants,
and RetrievalService — all without a real vector store or embedding model.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llama_index.core.schema import TextNode, NodeWithScore


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_node(node_id: str, text: str = "content", metadata: dict | None = None) -> TextNode:
    return TextNode(
        id_=node_id,
        text=text,
        metadata=metadata or {"doc_type": "markdown"},
    )


def make_node_with_score(node_id: str, score: float = 0.9, text: str = "content") -> NodeWithScore:
    return NodeWithScore(node=make_node(node_id, text), score=score)


def make_settings(**overrides):
    from app.config import Settings
    defaults = dict(
        groq_api_key="test-key",
        retrieval_top_k=8,
        bm25_top_k=8,
        rerank_top_k=4,
        reranker_backend="",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def run(coro):
    """Helper to run async code in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── BM25Retriever ──────────────────────────────────────────────────────────────

class TestBM25Retriever:

    def _nodes(self, n: int = 5):
        return [make_node(f"node-{i}", text=f"Document about topic {i}") for i in range(n)]

    def test_initialises_with_nodes(self):
        from app.retrieval.bm25_retriever import BM25Retriever

        nodes = self._nodes(5)
        retriever = BM25Retriever(nodes, make_settings())
        assert retriever._default_top_k == 8

    def test_returns_empty_when_no_nodes(self):
        from app.retrieval.bm25_retriever import BM25Retriever

        retriever = BM25Retriever([], make_settings())
        results = run(retriever.retrieve("python programming"))
        assert isinstance(results, list)

    def test_retrieve_returns_list(self):
        from app.retrieval.bm25_retriever import BM25Retriever

        nodes = self._nodes(10)
        retriever = BM25Retriever(nodes, make_settings())
        results = run(retriever.retrieve("topic"))
        assert isinstance(results, list)

    def test_metadata_filter_applied(self):
        """Post-filter should remove nodes that don't match the filter."""
        from app.retrieval.bm25_retriever import BM25Retriever
        from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

        nodes = [
            make_node("a", metadata={"doc_type": "course"}),
            make_node("b", metadata={"doc_type": "markdown"}),
            make_node("c", metadata={"doc_type": "course"}),
        ]
        retriever = BM25Retriever(nodes, make_settings())

        # Test the internal filter logic directly
        scored = [
            NodeWithScore(node=n, score=0.5) for n in nodes
        ]
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="doc_type", value="course")]
        )
        filtered = retriever._apply_filters(scored, filters)

        assert all(n.node.metadata["doc_type"] == "course" for n in filtered)
        assert len(filtered) == 2

    def test_matches_filters_true(self):
        from app.retrieval.bm25_retriever import BM25Retriever
        from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

        retriever = BM25Retriever([], make_settings())
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="doc_type", value="course")]
        )
        assert retriever._matches_filters({"doc_type": "course"}, filters) is True

    def test_matches_filters_false(self):
        from app.retrieval.bm25_retriever import BM25Retriever
        from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

        retriever = BM25Retriever([], make_settings())
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="doc_type", value="course")]
        )
        assert retriever._matches_filters({"doc_type": "markdown"}, filters) is False


# ── HybridRetriever (RRF) ──────────────────────────────────────────────────────

class TestHybridRetriever:

    def _make_hybrid(self, dense_results, bm25_results):
        from app.retrieval.hybrid_retriever import HybridRetriever

        dense = MagicMock()
        dense.retrieve = AsyncMock(return_value=dense_results)

        bm25 = MagicMock()
        bm25.retrieve = AsyncMock(return_value=bm25_results)

        return HybridRetriever(dense, bm25)

    def test_rrf_combines_results(self):
        from app.retrieval.hybrid_retriever import HybridRetriever

        dense = [make_node_with_score("a", 0.9), make_node_with_score("b", 0.8)]
        bm25  = [make_node_with_score("b", 5.0), make_node_with_score("c", 3.0)]

        retriever = self._make_hybrid(dense, bm25)
        results = run(retriever.retrieve("query", top_k=4))

        ids = [r.node.node_id for r in results]
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids

    def test_rrf_boosts_results_present_in_both(self):
        """A node ranked high in both lists should beat one ranked high in only one."""
        dense = [make_node_with_score("shared", 0.95), make_node_with_score("only_dense", 0.85)]
        bm25  = [make_node_with_score("shared", 9.0),  make_node_with_score("only_bm25", 7.0)]

        retriever = self._make_hybrid(dense, bm25)
        results = run(retriever.retrieve("query", top_k=3))

        # "shared" should be #1
        assert results[0].node.node_id == "shared"

    def test_returns_top_k(self):
        dense = [make_node_with_score(f"d{i}") for i in range(10)]
        bm25  = [make_node_with_score(f"b{i}") for i in range(10)]

        retriever = self._make_hybrid(dense, bm25)
        results = run(retriever.retrieve("query", top_k=5))

        assert len(results) <= 5

    def test_handles_empty_dense(self):
        retriever = self._make_hybrid([], [make_node_with_score("x")])
        results = run(retriever.retrieve("query", top_k=4))
        assert len(results) >= 1

    def test_handles_empty_bm25(self):
        retriever = self._make_hybrid([make_node_with_score("y")], [])
        results = run(retriever.retrieve("query", top_k=4))
        assert len(results) >= 1

    def test_handles_both_empty(self):
        retriever = self._make_hybrid([], [])
        results = run(retriever.retrieve("query", top_k=4))
        assert results == []

    def test_rrf_scores_assigned(self):
        """Returned nodes must have a positive RRF score."""
        dense = [make_node_with_score("a", 0.9)]
        bm25  = [make_node_with_score("a", 3.0)]

        retriever = self._make_hybrid(dense, bm25)
        results = run(retriever.retrieve("query", top_k=2))

        assert all(r.score > 0 for r in results)

    def test_rrf_formula_correctness(self):
        """Manually verify the RRF score for a known rank."""
        from app.retrieval.hybrid_retriever import HybridRetriever, RRF_K

        dense = MagicMock()
        dense.retrieve = AsyncMock(return_value=[make_node_with_score("only", 1.0)])
        bm25 = MagicMock()
        bm25.retrieve = AsyncMock(return_value=[])

        retriever = HybridRetriever(dense, bm25)
        results = run(retriever.retrieve("query", top_k=1))

        # rank=0 (0-indexed), so score = 1/(60+0+1) = 1/61
        expected = 1.0 / (RRF_K + 1)
        assert abs(results[0].score - expected) < 1e-9


# ── Reranker ───────────────────────────────────────────────────────────────────

class TestReranker:

    def test_noop_reranker_returns_top_k(self):
        from app.retrieval.reranker import NoOpReranker

        reranker = NoOpReranker()
        nodes = [make_node_with_score(f"n{i}", score=float(i)) for i in range(8)]
        results = reranker.rerank("query", nodes, top_k=4)

        assert len(results) == 4

    def test_noop_reranker_preserves_order(self):
        from app.retrieval.reranker import NoOpReranker

        reranker = NoOpReranker()
        nodes = [make_node_with_score(f"n{i}", score=float(i)) for i in range(5)]
        results = reranker.rerank("query", nodes, top_k=5)

        assert [r.node.node_id for r in results] == [n.node.node_id for n in nodes]

    def test_noop_reranker_empty_input(self):
        from app.retrieval.reranker import NoOpReranker

        reranker = NoOpReranker()
        assert reranker.rerank("query", [], top_k=4) == []

    def test_factory_returns_noop_for_empty_backend(self):
        from app.retrieval.reranker import RerankerFactory, NoOpReranker

        settings = make_settings(reranker_backend="")
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, NoOpReranker)

    def test_factory_returns_noop_for_unknown_backend(self):
        from app.retrieval.reranker import RerankerFactory, NoOpReranker

        settings = make_settings(reranker_backend="nonexistent")
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, NoOpReranker)


# ── RetrievalService ───────────────────────────────────────────────────────────

class TestRetrievalService:

    def _make_service(self, candidates, reranked=None):
        from app.retrieval.retrieval_service import RetrievalService
        from app.retrieval.reranker import NoOpReranker

        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=candidates)

        reranker = MagicMock()
        reranker.rerank.return_value = reranked if reranked is not None else candidates[:4]

        return RetrievalService(hybrid, reranker, make_settings())

    def test_returns_reranked_results(self):
        candidates = [make_node_with_score(f"n{i}") for i in range(8)]
        top4 = candidates[:4]

        service = self._make_service(candidates, reranked=top4)
        results = run(service.retrieve("query"))

        assert results == top4

    def test_returns_empty_when_no_candidates(self):
        service = self._make_service([])
        results = run(service.retrieve("query"))
        assert results == []

    def test_passes_filters_to_retriever(self):
        from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
        from app.retrieval.retrieval_service import RetrievalService

        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[make_node_with_score("a")])

        reranker = MagicMock()
        reranker.rerank.return_value = [make_node_with_score("a")]

        service = RetrievalService(hybrid, reranker, make_settings())

        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="doc_type", value="course")]
        )
        run(service.retrieve("query", filters=filters))

        call_kwargs = hybrid.retrieve.call_args
        assert call_kwargs.kwargs.get("filters") == filters
