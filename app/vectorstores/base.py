"""
vectorstores/base.py — Abstract Vector Store Interface

RESPONSIBILITY: Define the contract every vector store adapter must satisfy.
The retriever and index builder depend ONLY on this interface.
They NEVER import chromadb, qdrant_client, or any concrete implementation.

WHY THIS ABSTRACTION?
    Without it:
        from chromadb import Client  # in retriever.py or index_builder.py
        chroma_client = Client()
        collection = chroma_client.get_collection("kayfa")

    Now retriever.py is COUPLED to Chroma. To switch to Qdrant:
    → Edit retriever.py → Edit index_builder.py → Edit pipeline.py
    → Re-test everything.

    With BaseVectorStore:
    → Change ONE line in Settings: vector_store_backend = "qdrant"
    → VectorStoreFactory creates QdrantVectorStoreAdapter
    → Everything else is unchanged (they use the interface, not the impl)

    This is the Dependency Inversion Principle:
    "High-level modules (IndexBuilder, Retriever) should not depend on
     low-level modules (Chroma, Qdrant). Both should depend on abstractions."

WHY TWO METHODS?
    get_llama_vector_store():
        IndexBuilder needs the LlamaIndex vector store object to build
        a VectorStoreIndex. Different DBs have different LlamaIndex adapters.

    get_storage_context():
        StorageContext wraps both the vector store and docstore.
        Needed for index construction and persistence.
"""

from abc import ABC, abstractmethod

from llama_index.core import StorageContext
from llama_index.core.vector_stores.types import BasePydanticVectorStore


class BaseVectorStore(ABC):
    """
    Abstract interface for all vector store adapters.

    Every concrete adapter (Simple, Chroma, Qdrant, Weaviate) must implement
    these two methods. No other methods are part of the public contract.

    The retrieval layer (DenseRetriever) accesses the vector store through
    the VectorStoreIndex — it doesn't call these methods directly.
    These methods are used only during index construction (IndexBuilder).
    """

    @abstractmethod
    def get_llama_vector_store(self) -> BasePydanticVectorStore:
        """
        Return the LlamaIndex-compatible vector store object.

        Used by IndexBuilder to construct a VectorStoreIndex:
            VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(
                vector_store=adapter.get_llama_vector_store()
            ))

        Returns:
            BasePydanticVectorStore: LlamaIndex vector store object.
        """
        ...

    @abstractmethod
    def get_storage_context(self) -> StorageContext:
        """
        Return a StorageContext configured for this vector store.

        StorageContext bundles: vector_store, docstore, index_store.
        Used during VectorStoreIndex construction and persistence.

        Returns:
            StorageContext: Ready-to-use storage context.
        """
        ...
