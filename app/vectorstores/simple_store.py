"""
vectorstores/simple_store.py — SimpleVectorStore Adapter

RESPONSIBILITY: Wrap LlamaIndex's built-in SimpleVectorStore.

SimpleVectorStore is an in-memory vector store that:
- Requires zero external dependencies
- Stores vectors in a Python dict
- Persists to disk as vector_store.json
- Supports exact cosine similarity search

WHY IS THIS THE DEFAULT AND FALLBACK?
    1. HuggingFace Spaces free tier: no persistent external DB service.
       Chroma or Qdrant would require a separate running service.
       SimpleVectorStore lives IN the Python process — no external connection.

    2. Development: no Docker, no setup, just run.

    3. Scale: our knowledge base is ~500-1000 chunks.
       SimpleVectorStore handles up to ~50k vectors with acceptable latency.
       Exact brute-force search over 1000 vectors: < 1ms.
       ANN index (Chroma/Qdrant) adds overhead that's not justified at this scale.

    4. Production upgrade path: when you need scale (100k+ vectors),
       swap the adapter. Zero business logic changes.

LIMITATIONS:
    - In-memory: all vectors must fit in RAM (~2MB for 1000 × 1024-dim float32)
    - No horizontal scaling: can't share across multiple processes/containers
    - For this project: none of these limitations apply
"""

import logging

from llama_index.core import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore

from app.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class SimpleVectorStoreAdapter(BaseVectorStore):
    """
    Adapter for LlamaIndex's built-in SimpleVectorStore.

    Creates one SimpleVectorStore instance and wraps it.
    The same instance is returned by both get methods to ensure consistency.
    """

    def __init__(self) -> None:
        logger.info("SimpleVectorStoreAdapter: Initializing in-memory vector store.")
        self._store = SimpleVectorStore()

    def get_llama_vector_store(self) -> SimpleVectorStore:
        return self._store

    def get_storage_context(self) -> StorageContext:
        return StorageContext.from_defaults(vector_store=self._store)
