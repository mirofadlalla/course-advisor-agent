"""
ingestion/pipeline.py — Ingestion Pipeline Orchestrator

RESPONSIBILITY: Coordinate the build-or-load decision.
This is the ONLY class that knows the full ingestion flow.
It answers the question: "Do we need to build a new index, or load the existing one?"

WHY A SEPARATE PIPELINE CLASS?
    Without it, main.py lifespan would need to know about StorageManager,
    IndexBuilder, and their coordination logic. That's business logic in
    infrastructure code — a violation of SRP and the wrong abstraction level.

    The pipeline encapsulates the decision:
    "IF storage exists THEN load ELSE build AND persist"

    main.py just calls: index = await pipeline.run()
    It doesn't need to know what happened.

ALSO EXPOSES: the chunked nodes used during the last build.
    The HybridRetriever needs these nodes to build its BM25 index.
    Storing them here avoids re-running the pipeline just to get nodes.
"""

import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode

from app.config import Settings
from app.ingestion.index_bootstrap import bootstrap_index_from_bundle
from app.ingestion.index_builder import IndexBuilder
from app.ingestion.storage_manager import StorageManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the build-or-load ingestion flow.

    Logic:
        1. Check if a persisted index exists (StorageManager.exists())
        2a. YES → Load from disk (fast path, ~5 seconds)
        2b. NO  → Build from scratch (slow path, 2-5 min on CPU)
                → Persist to disk (so next startup takes the fast path)

    The `run()` method returns a (VectorStoreIndex, list[BaseNode]) tuple.
    The list[BaseNode] is needed by BM25Retriever.

    On load path: nodes are extracted from the loaded index's docstore.
    On build path: nodes come from IndexBuilder during construction.
    """

    def __init__(
        self,
        index_builder: IndexBuilder,
        storage_manager: StorageManager,
        settings: Settings | None = None,
    ) -> None:
        self._builder = index_builder
        self._storage = storage_manager
        self._settings = settings
        self._nodes: list[BaseNode] = []

    async def run(self) -> tuple[VectorStoreIndex, list[BaseNode]]:
        """
        Run the ingestion pipeline.

        Returns:
            tuple[VectorStoreIndex, list[BaseNode]]:
                - The ready-to-query vector index
                - All chunked nodes (for BM25 index construction)
        """
        if self._settings is not None:
            bootstrap_index_from_bundle(self._settings)

        if self._storage.exists():
            return await self._load_path()
        else:
            return await self._build_path()

    async def _load_path(self) -> tuple[VectorStoreIndex, list[BaseNode]]:
        """Fast path: load persisted index."""
        logger.info("IngestionPipeline: Loading existing index from storage...")
        index = self._storage.load()

        # Extract nodes from docstore for BM25 retriever
        nodes = list(index.docstore.docs.values())
        logger.info(f"IngestionPipeline: Loaded index with {len(nodes)} nodes.")

        return index, nodes

    async def _build_path(self) -> tuple[VectorStoreIndex, list[BaseNode]]:
        """Slow path: build index from scratch, then persist."""
        logger.info("IngestionPipeline: No existing index found. Building from scratch...")
        logger.info("IngestionPipeline: This may take 2-5 minutes on CPU (embedding computation).")

        index = await self._builder.build_async()

        # Extract chunked nodes before persisting (for BM25)
        nodes = list(index.docstore.docs.values())

        logger.info(f"IngestionPipeline: Built index with {len(nodes)} nodes. Persisting...")
        self._storage.persist(index)

        logger.info("IngestionPipeline: Index built and persisted. Future startups will be fast.")
        return index, nodes

    async def force_rebuild(self) -> tuple[VectorStoreIndex, list[BaseNode]]:
        """
        Force a full rebuild even if a persisted index exists.

        Use this when the knowledge base is updated (new documents added).
        Called via admin endpoint: DELETE /admin/index
        """
        logger.info("IngestionPipeline: Force rebuild requested. Deleting old index...")
        self._storage.delete()
        return await self._build_path()
