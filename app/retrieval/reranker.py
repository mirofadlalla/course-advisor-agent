"""
retrieval/reranker.py — Reranker Components

RESPONSIBILITY: Rerank retrieved candidates using a cross-encoder model.

THE RETRIEVAL RECALL-PRECISION TRADEOFF:
    Phase 1 (Retrieval): Fetch top-K candidates quickly.
        - Dense + BM25 hybrid → 8 candidates
        - Speed-optimized: bi-encoders embed query and doc separately
        - Approximate: may include false positives

    Phase 2 (Reranking): Score each (query, candidate) pair precisely.
        - Cross-encoder evaluates query and doc TOGETHER (full attention)
        - Much more accurate than bi-encoder scoring
        - But O(K) LLM calls → too slow for retrieval at scale
        - Applied only to the small candidate set (K=8)

    Result: High recall from retrieval + High precision from reranking.

CROSS-ENCODER vs BI-ENCODER:
    Bi-encoder (retrieval):
        embed(query) → q_vec    [independent]
        embed(doc)   → d_vec    [independent]
        score = cosine(q_vec, d_vec)
        Speed: O(1) lookup after indexing
        Accuracy: Misses joint features (e.g., "this" referring to prior context)

    Cross-encoder (reranking):
        score = model([query, doc])  [joint attention — reads both simultaneously]
        Speed: O(K) inference
        Accuracy: Much higher — can reason about relationships between query and doc

NO-OP PATTERN:
    NoOpReranker is the "Null Object Pattern":
    Instead of checking `if reranker: reranker.rerank(...)`, every code path
    calls `reranker.rerank(...)`. NoOpReranker returns the input unchanged.
    This eliminates conditional logic throughout the codebase.

FACTORY PATTERN:
    RerankerFactory reads settings.reranker_backend and returns the right type.
    The RetrievalService doesn't know which reranker it got.
"""

import logging
from abc import ABC, abstractmethod

from llama_index.core.schema import NodeWithScore

from app.config import Settings

logger = logging.getLogger(__name__)


# ─── Abstract Base ──────────────────────────────────────────────────────────


class BaseReranker(ABC):
    """
    Abstract contract for all rerankers.
    All rerankers receive (query, candidates, top_k) and return top_k reranked nodes.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        nodes: list[NodeWithScore],
        top_k: int,
    ) -> list[NodeWithScore]:
        """
        Rerank nodes by relevance to the query.

        Args:
            query:  The user's original query.
            nodes:  Candidate nodes from the retriever (e.g., 8 candidates).
            top_k:  Number of reranked results to return (e.g., 4).

        Returns:
            list[NodeWithScore]: top_k nodes sorted by reranker score (descending).
        """
        ...


# ─── No-Op (Null Object Pattern) ────────────────────────────────────────────


class NoOpReranker(BaseReranker):
    """
    Pass-through reranker. Returns top_k nodes without reranking.

    Used when:
    - settings.reranker_backend = "" (disabled)
    - Reranker model fails to load (graceful degradation)

    WHY NOT JUST SKIP THE RERANKER?
        Because then every caller would need an if/else check:
            if reranker:
                results = reranker.rerank(...)
            else:
                results = candidates[:top_k]

        NoOpReranker eliminates this. Callers always call .rerank().
        This is the Null Object Pattern.
    """

    def rerank(
        self,
        query: str,
        nodes: list[NodeWithScore],
        top_k: int,
    ) -> list[NodeWithScore]:
        return nodes[:top_k]


# ─── BGE Reranker (Local, No API Key) ───────────────────────────────────────


class BGEReranker(BaseReranker):
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    WHY BGE RERANKER?
        - Same family as BGE-M3 embeddings → consistent multilingual support
        - Runs locally: no API key, no network calls, no data privacy concerns
        - Apache 2.0 license: production-safe
        - bge-reranker-v2-m3: supports Arabic + English (critical for our corpus)

    INSTALLATION:
        pip install sentence-transformers  (already in requirements for BGE-M3)

    PERFORMANCE:
        CPU: ~100ms per (query, doc) pair × 8 candidates = ~800ms reranking
        GPU: ~10ms per pair = ~80ms total
        Acceptable for a chat application (human typing >> 800ms)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"BGEReranker: Loading cross-encoder '{model_name}'...")
            self._model = CrossEncoder(
                model_name,
                max_length=512,
            )
            logger.info("BGEReranker: Model loaded.")
        except ImportError as e:
            logger.error(
                f"BGEReranker: sentence-transformers not installed ({e}). "
                "Falling back to NoOpReranker."
            )
            self._model = None

    def rerank(
        self,
        query: str,
        nodes: list[NodeWithScore],
        top_k: int,
    ) -> list[NodeWithScore]:
        if not nodes or self._model is None:
            return nodes[:top_k]

        texts = [node.node.get_content() for node in nodes]
        pairs = [[query, text] for text in texts]

        # Cross-encoder scores each (query, doc) pair jointly
        scores = self._model.predict(pairs)

        # Zip scores with nodes and sort
        scored = sorted(
            zip(scores, nodes),
            key=lambda x: x[0],
            reverse=True,
        )

        reranked = [
            NodeWithScore(node=node.node, score=float(score))
            for score, node in scored[:top_k]
        ]

        logger.debug(
            f"BGEReranker: {len(nodes)} candidates → {len(reranked)} reranked results."
        )
        return reranked


# ─── Cohere Reranker (API-based) ────────────────────────────────────────────


class CohereReranker(BaseReranker):
    """
    Cross-encoder reranker using Cohere Rerank API.

    WHY COHERE?
        - State-of-the-art reranking quality (often better than local models)
        - No GPU needed: all compute on Cohere servers
        - Supports multilingual (Arabic + English)

    REQUIREMENTS:
        pip install cohere
        COHERE_API_KEY=your-key in .env

    COST:
        ~$0.001 per API call (1 call = 1 user query, reranks N docs)
        For a small sales agent: negligible cost.
    """

    def __init__(self, api_key: str, model: str = "rerank-multilingual-v3.0") -> None:
        try:
            import cohere
            self._client = cohere.Client(api_key)
            self._model = model
            logger.info(f"CohereReranker: Ready with model='{model}'")
        except ImportError:
            logger.error(
                "CohereReranker: cohere package not installed. "
                "Install with: pip install cohere. Falling back to pass-through."
            )
            self._client = None

    def rerank(
        self,
        query: str,
        nodes: list[NodeWithScore],
        top_k: int,
    ) -> list[NodeWithScore]:
        if not nodes or self._client is None:
            return nodes[:top_k]

        texts = [node.node.get_content() for node in nodes]

        response = self._client.rerank(
            query=query,
            documents=texts,
            top_n=top_k,
            model=self._model,
        )

        reranked = []
        for result in response.results:
            original_node = nodes[result.index]
            reranked.append(
                NodeWithScore(
                    node=original_node.node,
                    score=result.relevance_score,
                )
            )

        return reranked


# ─── Factory ────────────────────────────────────────────────────────────────


class RerankerFactory:
    """
    Creates the appropriate reranker based on Settings.

    Falls back to NoOpReranker on any error.
    """

    @staticmethod
    def create(settings: Settings) -> BaseReranker:
        backend = settings.reranker_backend.lower().strip()

        if backend == "bge":
            logger.info("RerankerFactory: Using BGEReranker (local cross-encoder).")
            reranker = BGEReranker()
            # If BGE failed to load (no sentence-transformers), fall back
            if reranker._model is None:
                logger.warning("RerankerFactory: BGEReranker load failed. Using NoOp.")
                return NoOpReranker()
            return reranker

        elif backend == "cohere":
            if not settings.cohere_api_key:
                logger.warning(
                    "RerankerFactory: reranker_backend=cohere but COHERE_API_KEY not set. "
                    "Using NoOpReranker."
                )
                return NoOpReranker()
            logger.info("RerankerFactory: Using CohereReranker.")
            return CohereReranker(api_key=settings.cohere_api_key)

        else:
            if backend:
                logger.warning(
                    f"RerankerFactory: Unknown reranker backend '{backend}'. "
                    "Using NoOpReranker."
                )
            else:
                logger.info("RerankerFactory: Reranker disabled. Using NoOpReranker.")
            return NoOpReranker()
