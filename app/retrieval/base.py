"""
retrieval/base.py — Abstract Retriever Interface

RESPONSIBILITY: Define the contract every retriever must satisfy.

WHY THIS ABSTRACTION?
    The KnowledgeRepository depends on BaseRetriever.
    It doesn't care if it gets DenseRetriever, BM25Retriever, HybridRetriever,
    or RetrievalService. They all look the same from the outside.

    This enables:
    1. Testing: inject MockRetriever that returns predefined results
    2. Flexibility: swap retrieval strategies without touching the repository
    3. Composition: HybridRetriever wraps DenseRetriever + BM25Retriever,
       both of which implement BaseRetriever

METADATA FILTERING:
    The filters parameter enables narrowing retrieval to specific doc_types.
    Example: search_knowledge(query="...", doc_type="course") creates:
        MetadataFilters(filters=[ExactMatchFilter(key="doc_type", value="course")])
    The retriever passes this to LlamaIndex, which filters before/after scoring.

    This is how the agent can say:
    "Find me courses about Python" → filter doc_type="course"
    "What are Kayfa's refund policies?" → filter doc_type="markdown" + keyword
"""

from abc import ABC, abstractmethod

from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilters


class BaseRetriever(ABC):
    """
    Abstract retriever contract.

    All retrievers (dense, BM25, hybrid, service) implement this interface.
    KnowledgeRepository depends only on BaseRetriever — never on concrete types.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: MetadataFilters | None = None,
    ) -> list[NodeWithScore]:
        """
        Retrieve relevant nodes for a query.

        Args:
            query:   Natural language query string.
            top_k:   Maximum number of results to return.
            filters: Optional metadata filters (e.g., doc_type="course").

        Returns:
            list[NodeWithScore]: Ranked nodes with similarity scores.
                NodeWithScore.node.get_content() → chunk text
                NodeWithScore.node.metadata → source metadata
                NodeWithScore.score → relevance score [0.0, 1.0]
        """
        ...
