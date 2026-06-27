"""
retrieval/retrieval_service.py — Final Composed Retriever

RESPONSIBILITY: Compose HybridRetriever + Reranker into the final retrieval pipeline.

WHY A SEPARATE SERVICE CLASS?
    HybridRetriever knows how to FIND candidates.
    Reranker knows how to SCORE candidates more accurately.
    RetrievalService knows how to COMPOSE them.

    If you remove the reranker, only RetrievalService changes — not HybridRetriever.
    If you add a result deduplication step, it goes here.
    If you add query expansion (HyDE, step-back prompting), it goes here.

    This is the Decorator / Chain of Responsibility pattern:
        query → HybridRetriever → [raw candidates] → Reranker → [reranked] → caller

ALSO IMPLEMENTS: BaseRetriever
    So KnowledgeRepository receives a BaseRetriever (the interface).
    It calls .retrieve() and gets results — doesn't care about the internals.
    Swapping to a different retrieval strategy = swap the BaseRetriever impl.
"""

import logging

from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilters

from app.config import Settings
from app.retrieval.base import BaseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import BaseReranker

logger = logging.getLogger(__name__)


class RetrievalService(BaseRetriever):
    """
    Composes HybridRetriever + Reranker.

    Flow:
        1. HybridRetriever.retrieve(query, top_k=settings.retrieval_top_k)
           → 8 candidates (dense + BM25 fused via RRF)
        2. Reranker.rerank(query, candidates, top_k=settings.rerank_top_k)
           → 4 final results (cross-encoder scored)

    The two top_k values are different BY DESIGN:
    - retrieval_top_k=8: retrieve MORE candidates than you need
      (more recall → better chance the best docs are in the set)
    - rerank_top_k=4: return FEWER, highly precise results to the agent
      (the agent's context window is precious — don't fill it with mediocre chunks)
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: BaseReranker,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._retrieval_top_k = settings.retrieval_top_k
        self._rerank_top_k = settings.rerank_top_k

        logger.info(
            f"RetrievalService initialized. "
            f"retrieval_top_k={self._retrieval_top_k}, "
            f"rerank_top_k={self._rerank_top_k}, "
            f"reranker={type(reranker).__name__}"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[NodeWithScore]:
        """
        Full retrieval pipeline: HybridRetriever → Reranker.

        Args:
            query:   The user's question.
            top_k:   Override for rerank_top_k (optional).
            filters: Metadata filters for doc_type narrowing.

        Returns:
            list[NodeWithScore]: Reranked, precise results ready for the agent.
        """
        final_top_k = top_k or self._rerank_top_k

        # Step 1: Broad hybrid retrieval (Dense + BM25)
        logger.debug(f"RetrievalService: Phase 1 — Hybrid retrieval top_k={self._retrieval_top_k}")
        candidates = await self._retriever.retrieve(
            query=query,
            top_k=self._retrieval_top_k,
            filters=filters,
        )

        if not candidates:
            logger.warning("RetrievalService: No candidates found by retriever.")
            return []

        # Step 2: Rerank for precision
        logger.debug(
            f"RetrievalService: Phase 2 — Reranking {len(candidates)} candidates "
            f"→ top {final_top_k}"
        )
        reranked = self._reranker.rerank(
            query=query,
            nodes=candidates,
            top_k=final_top_k,
        )

        logger.debug(
            f"RetrievalService: Final {len(reranked)} results. "
            f"Top score: {reranked[0].score:.4f if reranked else 'N/A'}"
        )

        return reranked
