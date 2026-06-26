"""
vectorstores/qdrant_store.py — Qdrant Adapter

RESPONSIBILITY: Wrap Qdrant as a LlamaIndex vector store.

Qdrant is a high-performance ANN vector search engine:
- Written in Rust (extremely fast, low memory)
- Rich metadata filtering with typed payload schema
- Supports sparse+dense hybrid search natively (future: use Qdrant's built-in hybrid)
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
    QDRANT_API_KEY=your-api-key        # for Qdrant Cloud
    QDRANT_COLLECTION=kayfa_knowledge
"""

import logging

from llama_index.core import StorageContext

from app.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStoreAdapter(BaseVectorStore):
    """
    Adapter for Qdrant via LlamaIndex.

    Supports both local (in-memory) and remote (HTTP) Qdrant instances.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "kayfa_knowledge",
        embedding_dim: int = 1024,  # BAAI/bge-m3 dimension
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            from llama_index.vector_stores.qdrant import QdrantVectorStore
        except ImportError as e:
            raise ImportError(
                "qdrant-client is not installed. "
                "Install with: pip install qdrant-client llama-index-vector-stores-qdrant"
            ) from e

        logger.info(f"QdrantVectorStoreAdapter: Connecting to {url}")

        client = QdrantClient(url=url, api_key=api_key)

        # Create collection if it doesn't exist
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            logger.info(f"QdrantVectorStoreAdapter: Creating collection '{collection_name}'")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

        self._store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
        )
        logger.info(f"QdrantVectorStoreAdapter: Ready. Collection='{collection_name}'")

    def get_llama_vector_store(self):
        return self._store

    def get_storage_context(self) -> StorageContext:
        return StorageContext.from_defaults(vector_store=self._store)
