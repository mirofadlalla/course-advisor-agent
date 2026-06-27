"""
ingestion/storage_manager.py — Index Storage Manager

RESPONSIBILITY: Persist and load VectorStoreIndex to/from disk.
That's it. Nothing else.

WHY SEPARATE FROM IndexBuilder?
    Single Responsibility Principle:
    - IndexBuilder: "Build an index from documents" (computation)
    - StorageManager: "Save and restore an index" (I/O)

    They change for different reasons:
    - IndexBuilder changes when you change the pipeline (new loader, new parser)
    - StorageManager changes when you change storage strategy (S3, Redis, disk format)

    Mixing them creates a class that changes for two reasons = violation of SRP.

THE BUILD-OR-LOAD PATTERN:
    First run:  No persisted index → build (expensive: 2-5 min)
    Subsequent: Index exists → load (fast: ~5 seconds)

    This is critical for HuggingFace Spaces:
    - Container restart → load from persistent /data volume
    - New deployment with changed data → delete storage → rebuild

    Without this, every Spaces restart re-embeds all documents.
    That's 2-5 minutes of downtime + GPU/CPU costs.

STORAGE FORMAT:
    LlamaIndex stores indexes as:
    - docstore.json      — all document nodes
    - index_store.json   — index metadata
    - vector_store.json  — vector data (for SimpleVectorStore)
    - graph_store.json   — graph data (unused here)

    For external vector stores (Chroma, Qdrant), vectors are stored in the DB.
    Only the index metadata is stored on disk.
"""

import logging
from pathlib import Path

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages persistence and loading of VectorStoreIndex.

    Receives:
        storage_path (str): Directory path for index storage.
        embed_model (BaseEmbedding): Required for loading — LlamaIndex needs
            the embed model to reconstruct the index for querying.

    Both are injected — StorageManager doesn't know HOW to create them.
    """

    def __init__(self, storage_path: str, embed_model: BaseEmbedding) -> None:
        self._storage_path = Path(storage_path)
        self._embed_model = embed_model

    def exists(self) -> bool:
        """
        Check if a persisted index exists at the configured path.

        Checks for docstore.json — the canonical indicator that a complete
        LlamaIndex storage exists. If only vector_store.json exists, the
        index is incomplete and should be rebuilt.
        """
        docstore_path = self._storage_path / "docstore.json"
        exists = docstore_path.exists()

        if exists:
            logger.info(f"StorageManager: Found persisted index at {self._storage_path}")
        else:
            logger.info(f"StorageManager: No persisted index at {self._storage_path}")

        return exists

    def persist(self, index: VectorStoreIndex) -> None:
        """
        Persist the index to disk.

        Creates the storage directory if it doesn't exist.
        Overwrites existing files — always saves the latest version.

        Args:
            index: The VectorStoreIndex to persist.
        """
        self._storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"StorageManager: Persisting index to {self._storage_path}...")

        index.storage_context.persist(persist_dir=str(self._storage_path))

        logger.info("StorageManager: Index persisted successfully.")

    def load(self) -> VectorStoreIndex:
        """
        Load a persisted index from disk.

        Raises:
            FileNotFoundError: If no persisted index exists.
            RuntimeError: If the index cannot be loaded (corrupted data).

        Returns:
            VectorStoreIndex: Ready-to-query index.
        """
        if not self.exists():
            raise FileNotFoundError(
                f"StorageManager: No index found at {self._storage_path}. "
                "Call persist() first or run the ingestion pipeline."
            )

        logger.info(f"StorageManager: Loading index from {self._storage_path}...")

        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(self._storage_path))
            index = load_index_from_storage(
                storage_context,
                embed_model=self._embed_model,
            )
            logger.info("StorageManager: Index loaded successfully.")
            return index

        except Exception as e:
            raise RuntimeError(
                f"StorageManager: Failed to load index from {self._storage_path}. "
                f"The storage may be corrupted. Delete the directory and rebuild. "
                f"Error: {e}"
            ) from e

    def delete(self) -> None:
        """
        Delete the persisted index. Useful for forcing a full rebuild.
        Called via admin endpoint or CLI when knowledge base is updated.
        """
        import shutil

        if self._storage_path.exists():
            shutil.rmtree(self._storage_path)
            logger.info(f"StorageManager: Deleted index at {self._storage_path}")
        else:
            logger.warning(f"StorageManager: Nothing to delete at {self._storage_path}")
