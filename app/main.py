"""
main.py — FastAPI Application & Full Lifespan Wiring

RESPONSIBILITY: Application entry-point and dependency graph construction.
This is the Composition Root — the ONE place where all objects are created
and wired together. No other file calls constructors for services or repos.

LIFESPAN FLOW (Chapter 9 of the RAG Pipeline plan):
    ┌─ Startup ────────────────────────────────────────────────────────────┐
    │  1.  Settings                  (already available as module-level)   │
    │  2.  EmbeddingProvider.get()   (lazy-loaded, cached singleton)       │
    │  3.  VectorStoreFactory.create() (Simple / Chroma / Qdrant + fallback│
    │  4.  Loaders (Markdown + JSON) composed via CompositeLoader          │
    │  5.  Parser + Chunker                                                 │
    │  6.  IndexBuilder              (receives all the above via DI)       │
    │  7.  StorageManager            (persist / load index)                │
    │  8.  IngestionPipeline.run()   → VectorStoreIndex + nodes            │
    │       • Fast path: loads persisted index from disk  (~5s)            │
    │       • Slow path: embeds all docs, persists        (~2-5 min CPU)   │
    │  9.  DenseRetriever + BM25Retriever → HybridRetriever (RRF fusion)  │
    │  10. RerankerFactory.create()  (BGE / Cohere / NoOp)                 │
    │  11. RetrievalService          (Hybrid + Reranker composed)          │
    │  12. Repositories              (Course, Roadmap, Knowledge)          │
    │  13. AgentDependencies         (typed container for all of the above)│
    │  14. ChatService               (creates Agent, registers tools)       │
    └──────────────────────────────────────────────────────────────────────┘
    app.state stores: agent_dependencies, chat_service
    Every request reads from app.state — zero re-creation per request.

    ┌─ Shutdown ───────────────────────────────────────────────────────────┐
    │  Yield resumes — clean-up hooks go here when needed.                 │
    └──────────────────────────────────────────────────────────────────────┘

DESIGN DECISIONS:
    • NO global singletons except `settings` (needed by create_agent before DI).
    • ALL other objects flow through the lifespan context and app.state.
    • The /chat endpoint is SYNCHRONOUS (run_sync). Upgrading to async streaming
      only requires changing chat_service.chat() — endpoint stays the same.
    • /health returns 200 immediately — no dependency on the RAG pipeline.
      Load balancers and HuggingFace health checks use this endpoint.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.config import settings
from app.dependencies import AgentDependencies
from app.embeddings.provider import EmbeddingProvider
from app.ingestion.chunker import SemanticChunker
from app.ingestion.index_builder import IndexBuilder
from app.ingestion.loaders import CompositeLoader, JSONLoader, MarkdownLoader
from app.ingestion.parsers import CompositeParser
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage_manager import StorageManager
from app.repositories.course_repository import CourseRepository
from app.repositories.knowledge_repository import KnowledgeRepository
from app.repositories.roadmap_repository import RoadmapRepository
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import RerankerFactory
from app.retrieval.retrieval_service import RetrievalService
from app.schemas.api import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.vectorstores.vector_store_factory import VectorStoreFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Full application startup and shutdown lifecycle.

    Constructs the entire dependency graph in the correct order.
    Each step receives its dependencies as constructor arguments (DI).
    """

    logger.info("=" * 60)
    logger.info("Course Advisor Agent — Starting up")
    logger.info("=" * 60)

    # ── Step 1: Embedding model ────────────────────────────────────────
    # Lazy-loaded and cached. The 400MB BAAI/bge-m3 model is downloaded
    # and cached on first call. Subsequent startups reuse the OS cache.
    logger.info(f"Step 1/9: Loading embedding model '{settings.embedding_model}'...")
    try:
        embed_model = EmbeddingProvider.get(settings)
        logger.info("Step 1/9: Embedding model ready.")
    except OSError as exc:
        # Model files unavailable (no network / no cache).
        # The /health endpoint must still respond 200; all other endpoints
        # will fail gracefully because embed_model is None.
        logger.warning(f"Step 1/9: Embedding model unavailable: {exc}. RAG disabled.")
        embed_model = None  # type: ignore[assignment]

    # ── Steps 2-9: Full RAG pipeline (skipped when embed_model is unavailable) ──
    if embed_model is not None:
        # ── Step 2: Vector store ───────────────────────────────────────────
        # Factory handles Simple / Chroma / Qdrant selection + auto-fallback.
        logger.info(
            f"Step 2/9: Creating vector store (backend='{settings.vector_store_backend}')..."
        )
        vector_store = VectorStoreFactory.create(settings)
        logger.info(f"Step 2/9: Vector store ready: {type(vector_store).__name__}")

        # ── Step 3: Ingestion components ───────────────────────────────────
        logger.info("Step 3/9: Configuring ingestion pipeline components...")
        loader = CompositeLoader(
            [
                MarkdownLoader(settings),
                JSONLoader(settings),
            ]
        )
        parser = CompositeParser()
        chunker = SemanticChunker(settings)

        # ── Step 4: Index builder ──────────────────────────────────────────
        # Receives all components via DI — doesn't create them internally.
        index_builder = IndexBuilder(
            loader=loader,
            parser=parser,
            chunker=chunker,
            embed_model=embed_model,
            vector_store=vector_store.get_llama_vector_store(),
        )

        # ── Step 5: Storage manager ────────────────────────────────────────
        storage_manager = StorageManager(
            storage_path=settings.index_storage_path,
            embed_model=embed_model,
        )

        # ── Step 6: Run ingestion pipeline ────────────────────────────────
        # build-or-load: fast if persisted index exists, slow if building fresh.
        logger.info("Step 4/9: Running ingestion pipeline (build or load)...")
        pipeline = IngestionPipeline(index_builder, storage_manager)
        index, nodes = await pipeline.run()
        logger.info(f"Step 4/9: Index ready with {len(nodes)} nodes.")

        # ── Step 7: Retrieval layer ────────────────────────────────────────
        logger.info("Step 5/9: Building retrieval layer (Dense + BM25 + Hybrid)...")
        dense_retriever = DenseRetriever(index, settings)
        bm25_retriever = BM25Retriever(nodes, settings)
        hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)
        reranker = RerankerFactory.create(settings)
        retrieval_service = RetrievalService(hybrid_retriever, reranker, settings)
        logger.info(f"Step 5/9: Retrieval ready. Reranker: {type(reranker).__name__}")

        # ── Step 8: Repositories ───────────────────────────────────────────
        logger.info("Step 6/9: Initializing repositories...")
        course_repo = CourseRepository()
        roadmap_repo = RoadmapRepository()
        knowledge_repo = KnowledgeRepository(retrieval_service=retrieval_service)
        logger.info("Step 6/9: Repositories ready.")

        # ── Step 9: Wire dependencies and services ─────────────────────────
        logger.info("Step 7/9: Wiring AgentDependencies...")
        deps = AgentDependencies(
            course_repository=course_repo,
            roadmap_repository=roadmap_repo,
            knowledge_repository=knowledge_repo,
            settings=settings,
        )

        logger.info("Step 8/9: Creating ChatService and registering agent tools...")
        chat_service = ChatService()

        app.state.agent_dependencies = deps
        app.state.chat_service = chat_service
    else:
        logger.warning("RAG pipeline skipped — embedding model unavailable.")
        app.state.agent_dependencies = None
        app.state.chat_service = None

    logger.info("=" * 60)
    logger.info("Course Advisor Agent — Ready to serve requests")
    logger.info("=" * 60)

    yield  # ← Application serves requests here

    # ── Shutdown ───────────────────────────────────────────────────────
    logger.info("Course Advisor Agent — Shutting down.")


# ── FastAPI application ────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """
    Health check endpoint.
    Returns 200 immediately — does not depend on the RAG pipeline.
    Used by load balancers and HuggingFace Spaces health checks.
    """
    return {"status": "ok", "app": settings.app_name}


@app.post("/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest, request: Request):
    """
    Main chat endpoint.
    Delegates entirely to ChatService — the endpoint knows nothing about
    the agent, tools, or retrieval pipeline.
    """
    return request.app.state.chat_service.chat(
        question=chat_request.message,
        deps=request.app.state.agent_dependencies,
    )
