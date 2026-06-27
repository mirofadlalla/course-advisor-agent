"""
retrieval/bm25_retriever.py — BM25 (Sparse / Keyword) Retriever

RESPONSIBILITY: Retrieve documents using BM25 keyword scoring.

WHAT IS BM25?
    BM25 (Best Match 25) is a probabilistic retrieval function.
    It scores documents by how well they match a query based on:
    - Term Frequency (TF): How often does the query term appear in the document?
    - Inverse Document Frequency (IDF): How rare is the term across all documents?
    - Document length normalization: Longer documents aren't unfairly rewarded

    Formula: score(d, q) = Σ IDF(t) × [TF(t,d) × (k1+1)] / [TF(t,d) + k1 × (1-b + b × |d|/avgdl)]

    k1=1.5 (term saturation), b=0.75 (length normalization) — BM25 defaults.

WHY BM25 IN ADDITION TO DENSE?
    Dense retrieval can MISS exact terminology:
    - "QRadar" → the embedding model may not distinguish QRadar from other SIEMs
    - "3,500 EGP" → exact prices need exact matching, not semantic similarity
    - "IAO accreditation" → acronyms and certifications are keyword-heavy
    - Arabic product names → embeddings may not capture exact Arabic terms well

    BM25 excels at: exact keywords, product names, acronyms, codes, prices.
    Dense excels at: semantic meaning, paraphrasing, cross-lingual queries.

    Together: higher recall. You don't miss results that one approach would miss.

IMPLEMENTATION:
    LlamaIndex BM25Retriever uses rank_bm25 under the hood.
    It's built from the same nodes as the VectorStoreIndex.
    This ensures both retrievers search the same corpus.

METADATA FILTERING:
    BM25 doesn't natively support metadata filtering like vector DBs do.
    We implement post-filtering: retrieve top_k × 2, then filter by metadata.
    This is slightly less efficient but correct.
"""

import asyncio
import logging

from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilters

from app.config import Settings
from app.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """
    Keyword retriever using BM25 scoring.

    Built from the chunked nodes extracted after index construction.
    Requires: pip install llama-index-retrievers-bm25 rank-bm25
    """

    def __init__(self, nodes: list[BaseNode], settings: Settings) -> None:
        self._nodes = nodes
        self._default_top_k = settings.bm25_top_k
        self._retriever = self._build_retriever(nodes)
        logger.info(
            f"BM25Retriever initialized with {len(nodes)} nodes. "
            f"Default top_k={self._default_top_k}"
        )

    def _build_retriever(self, nodes: list[BaseNode]):
        """Build BM25 index from nodes. Runs at startup (once)."""
        if not nodes:
            logger.warning("BM25Retriever: no nodes provided, retriever disabled.")
            return None

        try:
            from llama_index.retrievers.bm25 import BM25Retriever as LlamaBM25

            return LlamaBM25.from_defaults(
                nodes=nodes,
                similarity_top_k=self._default_top_k,
                # stemmer=Stemmer.Stemmer("english"),  # optional: pip install pystemmer
            )
        except ImportError:
            logger.warning(
                "BM25Retriever: llama-index-retrievers-bm25 not installed. "
                "BM25 will return empty results. "
                "Install with: pip install llama-index-retrievers-bm25 rank-bm25"
            )
            return None

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[NodeWithScore]:
        """
        Run BM25 keyword search.

        Post-filters by metadata if filters are provided.
        """
        if self._retriever is None:
            logger.warning("BM25Retriever: retriever not available, returning []")
            return []

        k = top_k or self._default_top_k

        # Update similarity_top_k dynamically
        self._retriever.similarity_top_k = k

        logger.debug(f"BM25Retriever: searching top_k={k}, query='{query[:80]}...'")

        loop = asyncio.get_event_loop()
        nodes = await loop.run_in_executor(
            None,
            lambda: self._retriever.retrieve(query),
        )

        # Post-filter by metadata if filters specified
        if filters and nodes:
            nodes = self._apply_filters(nodes, filters)

        logger.debug(f"BM25Retriever: returned {len(nodes)} results.")
        return nodes

    def _apply_filters(
        self,
        nodes: list[NodeWithScore],
        filters: MetadataFilters,
    ) -> list[NodeWithScore]:
        """Apply metadata filters post-retrieval."""
        filtered = []
        for node_with_score in nodes:
            metadata = node_with_score.node.metadata
            if self._matches_filters(metadata, filters):
                filtered.append(node_with_score)
        return filtered

    def _matches_filters(self, metadata: dict, filters: MetadataFilters) -> bool:
        """Check if a node's metadata satisfies all filter conditions."""
        for f in filters.filters:
            node_value = metadata.get(f.key)
            if node_value != f.value:
                return False
        return True
