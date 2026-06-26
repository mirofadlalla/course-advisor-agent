"""
vectorstores/chroma_store.py — ChromaDB Adapter

RESPONSIBILITY: Wrap ChromaDB as a LlamaIndex vector store.

ChromaDB is a production-grade ANN (Approximate Nearest Neighbor) vector DB:
- Persistent: survives process restarts without re-embedding
- Fast: HNSW index for sub-millisecond similarity search at 100k+ vectors
- Filterable: metadata filtering with AND/OR operators
- Self-hosted: run locally or in Docker

WHEN TO USE CHROMA OVER SIMPLE:
    - Knowledge base > 50k chunks
    - Need sub-millisecond latency at scale
    - Need to share the vector store across multiple API server instances
    - Need advanced metadata filtering (not just doc_type, but complex queries)

INSTALLATION:
    pip install chromadb llama-index-vector-stores-chroma

CONFIGURATION (via Settings / .env):
    VECTOR_STORE_BACKEND=chroma
    CHROMA_HOST=localhost       # for client-server mode
    CHROMA_PORT=8000
    CHROMA_COLLECTION=kayfa_knowledge

NOTE: If chromadb is not installed, VectorStoreFactory catches the ImportError
and falls back to SimpleVectorStoreAdapter. No crash, just a warning log.
"""

import logging
from dataclasses import dataclass, field

from llama_index.core import StorageContext

from app.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


@dataclass
class ChromaSettings:
    """Configuration for Chroma connection."""
    collection_name: str = "kayfa_knowledge"
    # None = in-process ephemeral (dev). Set host for client-server mode.
    host: str | None = None
    port: int = 8000
    persist_directory: str = "./storage/chroma"


class ChromaVectorStoreAdapter(BaseVectorStore):
    """
    Adapter for ChromaDB via LlamaIndex.

    Supports two modes:
    1. In-process (PersistentClient): Chroma runs in the same Python process.
       Data is persisted to disk at persist_directory. Simple setup.
    2. Client-server (HttpClient): Connect to a running Chroma server.
       Required for multi-process deployments.
    """

    def __init__(self, chroma_settings: ChromaSettings | None = None) -> None:
        # Defer import — chromadb may not be installed
        try:
            import chromadb
            from llama_index.vector_stores.chroma import ChromaVectorStore
        except ImportError as e:
            raise ImportError(
                "chromadb is not installed. "
                "Install with: pip install chromadb llama-index-vector-stores-chroma"
            ) from e

        cfg = chroma_settings or ChromaSettings()

        if cfg.host:
            logger.info(f"ChromaVectorStoreAdapter: Connecting to Chroma at {cfg.host}:{cfg.port}")
            client = chromadb.HttpClient(host=cfg.host, port=cfg.port)
        else:
            logger.info(f"ChromaVectorStoreAdapter: Using PersistentClient at {cfg.persist_directory}")
            client = chromadb.PersistentClient(path=cfg.persist_directory)

        collection = client.get_or_create_collection(
            name=cfg.collection_name,
            # cosine similarity is standard for text embeddings
            metadata={"hnsw:space": "cosine"},
        )

        self._store = ChromaVectorStore(chroma_collection=collection)
        logger.info(f"ChromaVectorStoreAdapter: Ready. Collection='{cfg.collection_name}'")

    def get_llama_vector_store(self):
        return self._store

    def get_storage_context(self) -> StorageContext:
        return StorageContext.from_defaults(vector_store=self._store)
