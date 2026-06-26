"""
vectorstores/qdrant_store.py — Qdrant Adapter

RESPONSIBILITY: Wrap Qdrant as a LlamaIndex vector store.

Qdrant is a high-performance ANN vector search engine:
- Written in Rust (extremely fast, low memory)
- Rich metadata filtering with typed payload schema
- Supports sparse+dense hybrid search natively
- Cloud-hosted option (Qdrant Cloud) or self-hosted via Docker

WHEN TO USE QDRANT OVER CHROMA:
    - Need best-in-class performance at scale (millions of vectors)
    - Need typed payload schemas with complex filters
    - Need native binary quantization for memory efficiency
    - Want cloud-managed vector DB (Qdrant Cloud)

INSTALLATION:
    pip install qdrant-client llama-index-vector-stores-qdrant

CONFIGURATION (via Settings / .env):
    VECTOR_STORE_BACKEND=qdrant
    QDRANT_URL=http://localhost:6333   # or Qdrant Cloud URL
    QDRANT_API_KEY=your-api-key        # for Qdrant Cloud; leave empty otherwise
    QDRANT_COLLECTION=kayfa_knowledge
"""

import logging

from llama_index.core import StorageContext

from app.config import Settings
from app.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)

# BAAI/bge-m3 produces 1024-dimensional embeddings.
# Must match the embedding model configured in Settings.
_BGE_M3_DIM = 1024


class QdrantVectorStoreAdapter(BaseVectorStore):
    """
    Adapter for Qdrant via LlamaIndex.

    Receives Settings (DI) — all configuration comes from the environment.
    Supports both local (HTTP) and Qdrant Cloud (HTTP + API key) instances.
    """

    def __init__(self, settings: Settings) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            from llama_index.vector_stores.qdrant import QdrantVectorStore
        except ImportError as e:
            raise ImportError(
                "qdrant-client is not installed. "
                "Install with: pip install qdrant-client llama-index-vector-stores-qdrant"
            ) from e

        api_key = settings.qdrant_api_key or None   # empty string → None
        logger.info(f"QdrantVectorStoreAdapter: Connecting to {settings.qdrant_url}")

        client = QdrantClient(url=settings.qdrant_url, api_key=api_key)

        # Create collection if it doesn't exist
        existing = {c.name for c in client.get_collections().collections}
        if settings.qdrant_collection not in existing:
            logger.info(
                f"QdrantVectorStoreAdapter: Creating collection "
                f"'{settings.qdrant_collection}' (dim={_BGE_M3_DIM})"
            )
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=_BGE_M3_DIM,
                    distance=Distance.COSINE,
                ),
            )

        self._store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
        )
        logger.info(
            f"QdrantVectorStoreAdapter: Ready. "
            f"Collection='{settings.qdrant_collection}'"
        )

    def get_llama_vector_store(self):
        return self._store

    def get_storage_context(self) -> StorageContext:
        return StorageContext.from_defaults(vector_store=self._store)
