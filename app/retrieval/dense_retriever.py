"""
retrieval/dense_retriever.py — Dense (Semantic) Retriever

RESPONSIBILITY: Retrieve documents using vector similarity search.

HOW IT WORKS:
    1. The query is embedded: query_vector = embed_model.embed(query)
    2. Cosine similarity is computed between query_vector and all stored vectors
    3. Top-K most similar chunks are returned

WHY "DENSE"?
    The term refers to the vector representation.
    A dense vector has a non-zero value in (nearly) every dimension.
    BAAI/bge-m3 produces 1024-dimensional dense vectors.
    Every dimension contributes to similarity computation.

    Contrast with "sparse" (BM25): most dimensions are 0. Only terms
    present in the document have non-zero weights.

STRENGTHS OF DENSE RETRIEVAL:
    ✅ Captures semantic similarity: "learn coding" matches "programming courses"
    ✅ Cross-lingual: BGE-M3 maps Arabic + English to the same vector space
    ✅ Handles paraphrasing and synonyms naturally
    ✅ Context-aware: understands "it" refers to a previously mentioned course

WEAKNESSES (addressed by HybridRetriever + BM25):
    ❌ Struggles with exact keyword matching: "QRadar v7.5.0" may not match well
    ❌ Out-of-vocabulary terms in product names: "Splunk ES", "Burp Suite"
    ❌ Long-tail terminology in Arabic educational content

METADATA FILTERING:
    LlamaIndex VectorIndexRetriever supports MetadataFilters for pre-filtering.
    This reduces the search space BEFORE similarity computation.
    Example: filter doc_type="diploma" → only search diploma content.
"""

import asyncio
import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilters

from app.config import Settings
from app.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Semantic retriever using VectorStoreIndex.

    Wraps LlamaIndex's VectorIndexRetriever with async support.
    Receives the VectorStoreIndex (built by IndexBuilder) via DI.
    """

    def __init__(self, index: VectorStoreIndex, settings: Settings) -> None:
        self._index = index
        self._default_top_k = settings.retrieval_top_k
        logger.info(
            f"DenseRetriever initialized. "
            f"Default top_k={self._default_top_k}"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[NodeWithScore]:
        """
        Run semantic similarity search.

        LlamaIndex retrieval is synchronous (CPU + vector math).
        We wrap it in run_in_executor to keep the event loop free.
        """
        k = top_k or self._default_top_k

        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=k,
            filters=filters,
        )

        logger.debug(f"DenseRetriever: searching top_k={k}, query='{query[:80]}...'")

        loop = asyncio.get_event_loop()
        nodes = await loop.run_in_executor(
            None,
            lambda: retriever.retrieve(query),
        )

        logger.debug(f"DenseRetriever: returned {len(nodes)} results.")
        return nodes
