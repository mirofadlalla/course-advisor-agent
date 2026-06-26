"""
retrieval/hybrid_retriever.py — Hybrid Retriever (Dense + BM25) with RRF Fusion

RESPONSIBILITY: Combine Dense and BM25 results using Reciprocal Rank Fusion.

THE HYBRID SEARCH PROBLEM:
    Dense retriever returns nodes scored [0.0 → 1.0] (cosine similarity).
    BM25 retriever returns nodes scored [0.0 → ∞] (TF-IDF weighted).

    You cannot simply add these scores. They live in different ranges,
    and their distributions have different shapes.

    Option A: Normalize scores to [0,1] and average.
        Problem: Normalization is data-dependent. A cosine score of 0.8
        might be "excellent" for sparse content but "mediocre" for
        high-similarity semantic content. The normalization introduces bias.

    Option B: Reciprocal Rank Fusion (RRF) — CHOSEN APPROACH.
        Use only the RANK of each result, not the raw score.
        RRF formula: score(d) = Σᵢ 1 / (k + rankᵢ(d))
        where k=60 is a smoothing constant.

WHY RRF?
    - Rank is stable: document #1 is #1 regardless of score distribution
    - No assumptions about score distributions
    - Empirically proven in IR research (Cormack et al., 2009)
    - Used in production by Elasticsearch, Vespa, Qdrant hybrid search
    - Simple to implement, impossible to get wrong

RRF EXAMPLE:
    Dense results: [courseA(rank=1), courseB(rank=2), courseC(rank=3)]
    BM25 results:  [courseC(rank=1), courseA(rank=2), courseD(rank=3)]

    k=60:
    courseA: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
    courseC: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
    courseB: 1/(60+2) + 0         = 0.01613
    courseD: 0         + 1/(60+3) = 0.01587

    Final order: [courseA, courseC, courseB, courseD]
    courseA wins because it ranks high in BOTH retrievers.

FUTURE: Replace RRF with a learned fusion model (e.g., trained on click data).
    The interface (BaseRetriever) doesn't change — only the internals.
"""

import asyncio
import logging
from collections import defaultdict

from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilters

from app.config import Settings
from app.retrieval.base import BaseRetriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)

# RRF smoothing constant. k=60 is the standard recommendation from the
# original paper (Cormack, Clarke, Buettcher, SIGIR 2009).
# Higher k → reduces the dominance of top-ranked documents.
RRF_K = 60


class HybridRetriever(BaseRetriever):
    """
    Fuses Dense and BM25 results using Reciprocal Rank Fusion.

    Both retrievers run CONCURRENTLY (asyncio.gather) for minimal latency.
    Results are merged using RRF and returned in fused rank order.

    Receives both retrievers via DI — doesn't know how they work internally.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = RRF_K,
    ) -> None:
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self._rrf_k = rrf_k
        logger.info(
            f"HybridRetriever initialized. "
            f"Dense + BM25 with RRF k={rrf_k}"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 8,
        filters: MetadataFilters | None = None,
    ) -> list[NodeWithScore]:
        """
        Run both retrievers concurrently, fuse with RRF.

        Step 1: Retrieve from Dense and BM25 simultaneously
        Step 2: Apply RRF fusion
        Step 3: Return top_k fused results

        Args:
            query:   Natural language query.
            top_k:   Final number of results to return after fusion.
            filters: Metadata filters (passed to both retrievers).

        Returns:
            list[NodeWithScore]: Fused results with RRF scores, descending.
        """
        logger.debug(
            f"HybridRetriever: running Dense + BM25 concurrently "
            f"for query='{query[:60]}...'"
        )

        # Run both retrievers concurrently — halves retrieval latency
        dense_results, bm25_results = await asyncio.gather(
            self._dense.retrieve(query, top_k=top_k, filters=filters),
            self._bm25.retrieve(query, top_k=top_k, filters=filters),
        )

        logger.debug(
            f"HybridRetriever: Dense={len(dense_results)} results, "
            f"BM25={len(bm25_results)} results."
        )

        # Apply RRF fusion
        fused = self._reciprocal_rank_fusion(
            [dense_results, bm25_results],
            top_k=top_k,
        )

        logger.debug(f"HybridRetriever: Fused to {len(fused)} results.")
        return fused

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[NodeWithScore]],
        top_k: int,
    ) -> list[NodeWithScore]:
        """
        Merge multiple ranked result lists using RRF.

        Args:
            result_lists: Results from each retriever (ranked, descending score).
            top_k:        Number of final results to return.

        Returns:
            list[NodeWithScore]: Merged results with RRF scores, top_k items.
        """
        # Map: node_id → accumulated RRF score
        rrf_scores: dict[str, float] = defaultdict(float)
        # Map: node_id → NodeWithScore (to reconstruct output)
        node_map: dict[str, NodeWithScore] = {}

        for result_list in result_lists:
            for rank, node_with_score in enumerate(result_list):
                node_id = node_with_score.node.node_id

                # RRF contribution: 1 / (k + rank)
                # rank is 0-indexed; add 1 to make it 1-indexed
                rrf_contribution = 1.0 / (self._rrf_k + rank + 1)
                rrf_scores[node_id] += rrf_contribution

                # Store the node (last write wins; both retrievers return same content)
                if node_id not in node_map:
                    node_map[node_id] = node_with_score

        # Sort by RRF score (descending) and take top_k
        sorted_ids = sorted(rrf_scores, key=lambda nid: rrf_scores[nid], reverse=True)
        top_ids = sorted_ids[:top_k]

        # Reconstruct NodeWithScore objects with RRF scores
        results = []
        for node_id in top_ids:
            node_with_score = node_map[node_id]
            # Replace the original score with the RRF score for transparency
            results.append(
                NodeWithScore(
                    node=node_with_score.node,
                    score=rrf_scores[node_id],
                )
            )

        return results
