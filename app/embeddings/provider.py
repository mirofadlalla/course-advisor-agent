"""
embeddings/provider.py — Embedding Model Provider

RESPONSIBILITY: Create and cache the embedding model instance.
Provides a factory method that returns a configured HuggingFaceEmbedding.

WHY A PROVIDER CLASS (not a module-level singleton)?

    The "global singleton" anti-pattern:

        # embeddings.py (BAD)
        embedding_model = HuggingFaceEmbedding("BAAI/bge-m3")
        # ^ This runs at IMPORT TIME. Every time any file does:
        # from app.embeddings import embedding_model
        # Python downloads 400MB and loads the model into memory.
        # This breaks: tests, type checking, cold imports, CLI tools.

    The Provider Pattern:

        # embeddings/provider.py (GOOD)
        class EmbeddingProvider:
            @classmethod
            def get(cls, settings) -> HuggingFaceEmbedding:
                ...
        # Nothing happens at import time. The model loads only when
        # EmbeddingProvider.get(settings) is explicitly called.
        # Called exactly ONCE in lifespan() → injected everywhere else.

WHY CACHE THE INSTANCE (_instance)?
    Creating HuggingFaceEmbedding("BAAI/bge-m3") is expensive:
    - Downloads ~570MB model files on first call
    - Loads tokenizer and model into memory (~2GB RAM)
    - Compiles CUDA kernels if GPU available

    We want exactly ONE instance for the entire application lifetime.
    The _instance class variable provides this without a global.

    Unlike a module-level singleton, this is:
    - Lazy: deferred until first use
    - Testable: can be reset between tests with EmbeddingProvider._instance = None
    - Explicit: callers know they're getting a shared instance

ABOUT BAAI/bge-m3:
    - Multilingual: handles Arabic + English in the same corpus seamlessly
    - 1024-dimensional embeddings (high expressiveness)
    - MTEB benchmark top performer for retrieval tasks
    - Apache 2.0 license (production-safe)
    - Configurable via settings.embedding_model — swap to any HuggingFace model
"""

import logging

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """
    Factory + Cache for the embedding model.

    Usage:
        embed_model = EmbeddingProvider.get(settings)

    The same instance is returned on every call after the first.
    Thread-safe for read operations (model inference is stateless).
    """

    _instance: HuggingFaceEmbedding | None = None

    @classmethod
    def get(cls, settings: Settings) -> HuggingFaceEmbedding:
        """
        Return the embedding model, creating it if not yet initialized.

        Args:
            settings: Application settings (model name, device).

        Returns:
            HuggingFaceEmbedding: Ready-to-use embedding model.
        """
        if cls._instance is None:
            logger.info(
                f"EmbeddingProvider: Loading model '{settings.embedding_model}' "
                f"on device='{settings.embedding_device}'..."
            )
            logger.info(
                "EmbeddingProvider: First load may take several minutes "
                "to download model files (~570MB for BAAI/bge-m3)."
            )

            cls._instance = HuggingFaceEmbedding(
                model_name=settings.embedding_model,
                device=settings.embedding_device,
                # embed_batch_size: process 32 chunks at once for efficiency
                embed_batch_size=32,
                # trust_remote_code needed for some newer HuggingFace models
                trust_remote_code=False,
            )

            embed_dim = getattr(cls._instance, "embed_dim", None)
            if embed_dim is not None:
                logger.info(f"EmbeddingProvider: Model loaded. Embedding dimension: {embed_dim}")
            else:
                logger.info("EmbeddingProvider: Model loaded.")

        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Reset the cached instance. Useful for testing.

        Example:
            def test_different_model():
                EmbeddingProvider.reset()
                settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                embed = EmbeddingProvider.get(settings)
        """
        cls._instance = None
        logger.info("EmbeddingProvider: Instance reset.")
