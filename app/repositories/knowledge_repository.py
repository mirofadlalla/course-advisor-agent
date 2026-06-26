"""
repositories/knowledge_repository.py — Knowledge Repository (Concrete)

RESPONSIBILITY: Implement IKnowledgeRepository using RetrievalService.

This class is the boundary between the retrieval infrastructure (LlamaIndex,
vector stores, rerankers) and the agent's tools (pure business logic).

WHAT IT DOES:
    1. Receives a query and optional filters from the tool
    2. Builds MetadataFilters from the dict (translates domain to LlamaIndex)
    3. Delegates to RetrievalService (which runs Hybrid + Reranker)
    4. Converts NodeWithScore → SearchResult (infrastructure → domain)
    5. Returns list[SearchResult] to the tool

WHY CONVERT NodeWithScore → SearchResult?
    NodeWithScore is a LlamaIndex type. SearchResult is OUR type.
    The tool never sees LlamaIndex types. This means:
    - Tools are portable: they'd work with Haystack, Weaviate, any backend
    - Tools are testable: inject MockKnowledgeRepository with fake SearchResults
    - Business logic (how to answer questions) is separate from
      infrastructure logic (how to retrieve documents)

    The conversion code lives HERE, in ONE place. Changing the retrieval
    infrastructure never requires touching tool code.
"""

import logging

from llama_index.core.vector_stores.types import (
    MetadataFilters,
    ExactMatchFilter,
    FilterOperator,
    FilterCondition,
)

from app.repositories.base import IKnowledgeRepository
from app.retrieval.base import BaseRetriever
from app.schemas.search import SearchResult

logger = logging.getLogger(__name__)


class KnowledgeRepository(IKnowledgeRepository):
    """
    Concrete knowledge repository backed by RetrievalService.

    Receives BaseRetriever (interface) via DI — doesn't know if it's
    a HybridRetriever, a DenseRetriever, or a MockRetriever.

    NOTE: لاحظ إن الـ Repository ميعرفش أى حاجة عن LlamaIndex أو Chroma.
    بيعرف بس عن RetrievalService (interface).
    ده هو الـ Dependency Inversion Principle.
    """

    def __init__(self, retrieval_service: BaseRetriever) -> None:
        self._service = retrieval_service
        logger.info(
            f"KnowledgeRepository initialized with "
            f"retriever={type(retrieval_service).__name__}"
        )

    async def search(
        self,
        query: str,
        top_k: int = 4,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Search the knowledge base.

        Translates the filters dict to LlamaIndex MetadataFilters,
        delegates to the retrieval service, and converts results to SearchResult.

        Args:
            query:   Natural language query.
            top_k:   Number of results to return.
            filters: Optional dict of metadata filters.
                     Example: {"doc_type": "course"}

        Returns:
            list[SearchResult]: Domain objects, decoupled from LlamaIndex.
        """
        logger.info(
            f"KnowledgeRepository.search: query='{query[:60]}...', "
            f"top_k={top_k}, filters={filters}"
        )

        # Build LlamaIndex MetadataFilters from the dict
        metadata_filters = self._build_filters(filters)

        # Delegate to the retrieval service (Hybrid + Reranker)
        nodes = await self._service.retrieve(
            query=query,
            top_k=top_k,
            filters=metadata_filters,
        )

        # Convert NodeWithScore → SearchResult (infrastructure → domain)
        results = [self._to_search_result(node) for node in nodes]

        logger.info(
            f"KnowledgeRepository.search: returned {len(results)} results."
        )
        return results

    def _build_filters(self, filters: dict | None) -> MetadataFilters | None:
        """Convert a simple dict to LlamaIndex MetadataFilters."""
        if not filters:
            return None

        filter_list = [
            ExactMatchFilter(key=key, value=value)
            for key, value in filters.items()
            if value is not None
        ]

        if not filter_list:
            return None

        return MetadataFilters(
            filters=filter_list,
            condition=FilterCondition.AND,
        )

    def _to_search_result(self, node_with_score) -> SearchResult:
        """
        Convert a LlamaIndex NodeWithScore to a SearchResult domain object.

        This is the anti-corruption layer between LlamaIndex and our domain.
        """
        node = node_with_score.node
        metadata = node.metadata or {}

        return SearchResult(
            text=node.get_content(),
            score=float(node_with_score.score or 0.0),
            source_file=metadata.get(
                "file_name",
                metadata.get("source_file", "unknown")
            ),
            doc_type=metadata.get("doc_type", ""),
            section_header=metadata.get("section_summary", metadata.get("header", "")),
            metadata={k: v for k, v in metadata.items()},
        )