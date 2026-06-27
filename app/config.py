"""
config.py — Application Settings (Single Source of Truth)

WHY:
    Every value that varies between environments (local, HuggingFace Spaces,
    CI/CD, production) lives here. Business logic NEVER contains hardcoded
    paths, model names, or thresholds. They read from Settings.

    This follows the 12-Factor App principle: config in the environment.

HOW:
    pydantic-settings reads from environment variables first, then .env file.
    All components receive a Settings instance via dependency injection.
    No component calls Settings() on its own — it receives it.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ─── Application ──────────────────────────────────────────────────────────
    app_name: str = "Course Advisor Agent"
    debug: bool = True

    # ─── LLM ──────────────────────────────────────────────────────────────────
    groq_api_key: str
    model_name: str = "groq:llama-3.3-70b-versatile"

    # ─── Embedding ────────────────────────────────────────────────────────────
    # BAAI/bge-m3: multilingual, 1024-dim, excellent for Arabic+English corpus.
    # Configurable so you can swap to OpenAI or Cohere embeddings via env var.
    embedding_model: str = "BAAI/bge-m3"
    # Use "cuda" on GPU machines, "cpu" for local dev & HuggingFace free tier.
    embedding_device: str = "cpu"

    # ─── Vector Store Backend ─────────────────────────────────────────────────
    # Options: "simple" | "chroma" | "qdrant"
    # "simple" = LlamaIndex in-memory SimpleVectorStore (no external DB needed)
    # VectorStoreFactory will auto-fallback to "simple" if the chosen backend
    # is unavailable (e.g., chromadb not installed on HuggingFace Spaces).
    vector_store_backend: str = "simple"

    # ─── Chroma Settings ──────────────────────────────────────────────────────
    # Only used when vector_store_backend="chroma".
    # chroma_host: set to a hostname for client-server mode (e.g. "localhost").
    #              Leave empty ("") for in-process PersistentClient mode.
    chroma_host: str = ""
    chroma_port: int = 8000
    chroma_collection: str = "kayfa_knowledge"
    chroma_persist_dir: str = "./storage/chroma"

    # ─── Qdrant Settings ──────────────────────────────────────────────────────
    # Only used when vector_store_backend="qdrant".
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "kayfa_knowledge"

    # ─── Storage ──────────────────────────────────────────────────────────────
    # Where the persisted index lives.
    # Local:               ./storage/index
    # HuggingFace Spaces:  /data/index  (persistent volume)
    index_storage_path: str = "storage/index"
    # Root directory for all knowledge base files (MD + JSON)
    data_dir: str = "data"
    # Where metrics are persisted
    metrics_storage_path: str = "storage/metrics.json"

    # ─── Chunking ─────────────────────────────────────────────────────────────
    # chunk_size=512: enough context for meaningful semantic content without
    # diluting the embedding signal. BGE-M3 supports 8192 tokens but shorter
    # chunks = sharper embeddings = better retrieval precision.
    chunk_size: int = 512
    # chunk_overlap=64: ~12.5% overlap prevents losing context at boundaries.
    # Sentences split across chunk edges remain retrievable from either chunk.
    chunk_overlap: int = 64

    # ─── Retrieval ────────────────────────────────────────────────────────────
    # retrieval_top_k: candidates fetched from vector store (before reranking).
    # Fetch more than you need so the reranker has good candidates to work with.
    retrieval_top_k: int = 8
    # bm25_top_k: keyword (sparse) retrieval candidates.
    bm25_top_k: int = 8
    # rerank_top_k: final results returned to the agent after reranking.
    rerank_top_k: int = 4

    # ─── Reranker ─────────────────────────────────────────────────────────────
    # Options: "bge" (local, free) | "cohere" (API) | "" (disabled / NoOp)
    reranker_backend: str = ""
    cohere_api_key: str = ""

    # ─── Deployment ───────────────────────────────────────────────────────────
    # Set HF_SPACES=true in HuggingFace Space secrets.
    # When true, VectorStoreFactory ALWAYS uses SimpleVectorStore regardless
    # of vector_store_backend — no external DB connections attempted.
    is_hf_spaces: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


# Module-level instance: used only in places that CANNOT receive DI
# (e.g., agent.py create_agent factory, which runs before lifespan).
# All injected components should receive settings as a constructor argument.
settings = Settings()
