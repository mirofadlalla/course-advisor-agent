"""
vectorstores/vector_store_factory.py — Vector Store Factory with Fallback

RESPONSIBILITY: Create the appropriate vector store adapter based on Settings.
Implements automatic fallback to SimpleVectorStore when external DBs
are unavailable (key for HuggingFace Spaces deployment).

THE FALLBACK ARCHITECTURE:
    Problem: You deploy to HuggingFace Spaces. You set VECTOR_STORE_BACKEND=chroma.
    But chromadb is not in requirements.txt, or the Chroma server isn't running.
    Without a fallback, the app crashes on startup.

    Solution: VectorStoreFactory catches ImportError and ConnectionError,
    logs a warning, and returns SimpleVectorStoreAdapter.
    The app starts successfully. The user gets working answers.
    You fix the issue without emergency hotfixes.

FALLBACK DECISION TREE:
    ┌─────────────────────────────────────────────┐
    │ is_hf_spaces = True?                         │
    │ OR vector_store_backend = "simple"?          │
    └──────────────────────┬──────────────────────┘
                           │ YES → SimpleVectorStoreAdapter
                           │
                           │ NO
                           ▼
    ┌─────────────────────────────────────────────┐
    │ Try to create Chroma/Qdrant adapter          │
    │ (requires package + running service)         │
    └──────────────────────┬──────────────────────┘
                           │ SUCCESS → Chroma/QdrantAdapter
                           │
                           │ ImportError / ConnectionError
                           ▼
                    SimpleVectorStoreAdapter
                    (with warning log)

ALL ADAPTERS RECEIVE Settings:
    ChromaVectorStoreAdapter(settings) — reads chroma_host, chroma_port, etc.
    QdrantVectorStoreAdapter(settings) — reads qdrant_url, qdrant_api_key, etc.
    No hardcoded values anywhere in adapters.
"""

import logging

from app.config import Settings
from app.vectorstores.base import BaseVectorStore
from app.vectorstores.simple_store import SimpleVectorStoreAdapter

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """
    Creates vector store adapters with automatic fallback.

    Usage:
        vector_store = VectorStoreFactory.create(settings)
        # Returns an adapter implementing BaseVectorStore.
        # Caller never knows if it got Chroma, Qdrant, or Simple.
    """

    @staticmethod
    def create(settings: Settings) -> BaseVectorStore:
        """
        Create the appropriate vector store adapter.

        Args:
            settings: Application settings with vector_store_backend config.

        Returns:
            BaseVectorStore: A configured vector store adapter.
        """
        # Fast path: explicitly configured for simple or HuggingFace Spaces
        if settings.is_hf_spaces:
            logger.info(
                "VectorStoreFactory: HuggingFace Spaces detected (IS_HF_SPACES=true). "
                "Using SimpleVectorStore (no external DB required)."
            )
            return SimpleVectorStoreAdapter()

        backend = settings.vector_store_backend.lower().strip()

        if backend == "simple":
            logger.info("VectorStoreFactory: Using SimpleVectorStore (configured).")
            return SimpleVectorStoreAdapter()

        elif backend == "chroma":
            return VectorStoreFactory._create_chroma(settings)

        elif backend == "qdrant":
            return VectorStoreFactory._create_qdrant(settings)

        else:
            logger.warning(
                f"VectorStoreFactory: Unknown backend '{backend}'. "
                "Falling back to SimpleVectorStore."
            )
            return SimpleVectorStoreAdapter()

    @staticmethod
    def _create_chroma(settings: Settings) -> BaseVectorStore:
        """Attempt to create Chroma adapter, fallback to Simple on any failure."""
        try:
            from app.vectorstores.chroma_store import ChromaVectorStoreAdapter
            logger.info("VectorStoreFactory: Creating ChromaVectorStoreAdapter...")
            return ChromaVectorStoreAdapter(settings)
        except ImportError as e:
            logger.warning(
                f"VectorStoreFactory: chromadb not available ({e}). "
                "Falling back to SimpleVectorStore. "
                "Install with: pip install chromadb llama-index-vector-stores-chroma"
            )
            return SimpleVectorStoreAdapter()
        except Exception as e:
            logger.warning(
                f"VectorStoreFactory: Failed to connect to Chroma ({e}). "
                "Falling back to SimpleVectorStore."
            )
            return SimpleVectorStoreAdapter()

    @staticmethod
    def _create_qdrant(settings: Settings) -> BaseVectorStore:
        """Attempt to create Qdrant adapter, fallback to Simple on any failure."""
        try:
            from app.vectorstores.qdrant_store import QdrantVectorStoreAdapter
            logger.info("VectorStoreFactory: Creating QdrantVectorStoreAdapter...")
            return QdrantVectorStoreAdapter(settings)
        except ImportError as e:
            logger.warning(
                f"VectorStoreFactory: qdrant-client not available ({e}). "
                "Falling back to SimpleVectorStore. "
                "Install with: pip install qdrant-client llama-index-vector-stores-qdrant"
            )
            return SimpleVectorStoreAdapter()
        except Exception as e:
            logger.warning(
                f"VectorStoreFactory: Failed to connect to Qdrant ({e}). "
                "Falling back to SimpleVectorStore."
            )
            return SimpleVectorStoreAdapter()
