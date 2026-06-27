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
    CHROMA_HOST=localhost            # leave empty for in-process mode
    CHROMA_PORT=8000
    CHROMA_COLLECTION=kayfa_knowledge
    CHROMA_PERSIST_DIR=./storage/chroma

NOTE: If chromadb is not installed, VectorStoreFactory catches the ImportError
and falls back to SimpleVectorStoreAdapter. No crash, just a warning log.
"""

import logging

from llama_index.core import StorageContext

from app.config import Settings
from app.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStoreAdapter(BaseVectorStore):
    """
    Adapter for ChromaDB via LlamaIndex.

    Receives Settings (DI) — all configuration comes from the environment.
    No hardcoded paths, hosts, or collection names.

    Supports two modes:
    1. In-process (PersistentClient): Chroma runs in the same Python process.
       Data is persisted to disk at settings.chroma_persist_dir. Simple setup.
       Set by leaving settings.chroma_host empty ("").
    2. Client-server (HttpClient): Connect to a running Chroma server.
       Required for multi-process deployments.
       Set settings.chroma_host="localhost" (or any hostname).
    """

    def __init__(self, settings: Settings) -> None:
        # Defer import — chromadb may not be installed
        try:
            import chromadb
            from llama_index.vector_stores.chroma import ChromaVectorStore
        except ImportError as e:
            raise ImportError(
                "chromadb is not installed. "
                "Install with: pip install chromadb llama-index-vector-stores-chroma"
            ) from e

        if settings.chroma_host:
            logger.info(
                f"ChromaVectorStoreAdapter: Connecting to Chroma at "
                f"{settings.chroma_host}:{settings.chroma_port}"
            )
            client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
        else:
            logger.info(
                f"ChromaVectorStoreAdapter: Using PersistentClient at "
                f"'{settings.chroma_persist_dir}'"
            )
            client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

        collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            # cosine similarity is standard for text embeddings
            metadata={"hnsw:space": "cosine"},
        )

        self._store = ChromaVectorStore(chroma_collection=collection)
        logger.info(f"ChromaVectorStoreAdapter: Ready. Collection='{settings.chroma_collection}'")

    def get_llama_vector_store(self):
        return self._store

    def get_storage_context(self) -> StorageContext:
        return StorageContext.from_defaults(vector_store=self._store)
