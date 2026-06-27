"""
main.py — FastAPI Application & Full Lifespan Wiring

RESPONSIBILITY: Application entry-point and dependency graph construction.
This is the Composition Root — the ONE place where all objects are created
and wired together. No other file calls constructors for services or repos.

LIFESPAN FLOW:
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
    app.state stores: agent_dependencies, chat_service, pipeline,
                      storage_manager, index_node_count, last_build_time
    Every request reads from app.state — zero re-creation per request.

    ┌─ Shutdown ───────────────────────────────────────────────────────────┐
    │  Yield resumes — clean-up hooks go here when needed.                 │
    └──────────────────────────────────────────────────────────────────────┘

DESIGN DECISIONS:
    • NO global singletons except `settings` (needed by create_agent before DI).
    • ALL other objects flow through the lifespan context and app.state.
    • The /chat endpoint measures total wall-clock latency and writes to
      MetricsStore after every request (success or failure).
    • /health returns 200 immediately — no dependency on the RAG pipeline.
      Load balancers and HuggingFace health checks use this endpoint.
    • Static files served at /static; / redirects to the chat UI.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path


import json

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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
from app.monitoring.store import metrics_store
from app.monitoring.router import router as monitoring_router
from app.api.ingestion_router import router as ingestion_router

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
    logger.info(f"Step 1/9: Loading embedding model '{settings.embedding_model}'...")
    try:
        embed_model = EmbeddingProvider.get(settings)
        logger.info("Step 1/9: Embedding model ready.")
    except OSError as exc:
        logger.warning(f"Step 1/9: Embedding model unavailable: {exc}. RAG disabled.")
        embed_model = None  # type: ignore[assignment]

    # ── Steps 2-9: Full RAG pipeline (skipped when embed_model is unavailable) ──
    if embed_model is not None:
        # ── Step 2: Vector store ───────────────────────────────────────────
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
        logger.info("Step 4/9: Running ingestion pipeline (build or load)...")
        pipeline = IngestionPipeline(index_builder, storage_manager, settings=settings)
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
        app.state.pipeline = pipeline
        app.state.storage_manager = storage_manager
        app.state.index_node_count = len(nodes)
        app.state.last_build_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    else:
        logger.warning("RAG pipeline skipped — embedding model unavailable.")
        app.state.agent_dependencies = None
        app.state.chat_service = None
        app.state.pipeline = None
        app.state.storage_manager = None
        app.state.index_node_count = 0
        app.state.last_build_time = None

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

# ── Static files ───────────────────────────────────────────────────────────
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── Include routers ────────────────────────────────────────────────────────
app.include_router(monitoring_router)
app.include_router(ingestion_router)


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Redirect / to the Chat UI."""
    index_path = _static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return RedirectResponse(url="/static/index.html")


@app.get("/chat-ui", include_in_schema=False)
def chat_ui():
    return FileResponse(str(_static_dir / "index.html"))


@app.get("/ingest-ui", include_in_schema=False)
def ingest_ui():
    return FileResponse(str(_static_dir / "ingest.html"))


@app.get("/monitor-ui", include_in_schema=False)
def monitor_ui():
    return FileResponse(str(_static_dir / "monitor.html"))


@app.get("/health")
def health():
    """
    Health check endpoint.
    Returns 200 immediately — does not depend on the RAG pipeline.
    Used by load balancers and HuggingFace Spaces health checks.
    """
    return {
        "status": "ok",
        "app": settings.app_name,
        "rag_enabled": True,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest, request: Request):
    """
    Main chat endpoint.
    Delegates entirely to ChatService, instruments timing, writes to MetricsStore.
    """
    chat_service = request.app.state.chat_service
    agent_deps = request.app.state.agent_dependencies

    if chat_service is None or agent_deps is None:
        # Record the error
        metrics_store.record_request(
            question=chat_request.message,
            total_latency_ms=0,
            agent_process_ms=0,
            tokens_in=0,
            tokens_out=0,
            success=False,
            error_msg="RAG pipeline not initialized",
            model=settings.model_name,
        )
        return ChatResponse(
            response="⚠️ The RAG pipeline is not initialized. The system is starting up or the embedding model is unavailable.",
            latency_ms=0,
        )

    t_start = time.perf_counter()
    try:
        result = chat_service.chat(
            question=chat_request.message,
            deps=agent_deps,
        )
        total_latency_ms = (time.perf_counter() - t_start) * 1000

        request_id = metrics_store.record_request(
            question=chat_request.message,
            total_latency_ms=total_latency_ms,
            agent_process_ms=result.get("agent_process_ms", 0),
            tokens_in=result.get("tokens_in", 0),
            tokens_out=result.get("tokens_out", 0),
            success=True,
            model=settings.model_name,
        )

        return ChatResponse(
            response=result["response"],
            latency_ms=round(total_latency_ms, 2),
            agent_process_ms=result.get("agent_process_ms", 0),
            tokens_in=result.get("tokens_in", 0),
            tokens_out=result.get("tokens_out", 0),
            cost_usd=result.get("cost_usd", 0.0),
            request_id=request_id,
        )

    except Exception as exc:
        total_latency_ms = (time.perf_counter() - t_start) * 1000
        metrics_store.record_request(
            question=chat_request.message,
            total_latency_ms=total_latency_ms,
            agent_process_ms=0,
            tokens_in=0,
            tokens_out=0,
            success=False,
            error_msg=str(exc),
            model=settings.model_name,
        )
        logger.error(f"Chat endpoint error: {exc}")
        return ChatResponse(
            response=f"⚠️ An error occurred: {exc}",
            latency_ms=round(total_latency_ms, 2),
        )


# ── Streaming SSE endpoint ─────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(chat_request: ChatRequest, request: Request):
    """
    Streaming SSE version of /chat.

    Uses PydanticAI run_stream() for native LLM token streaming. Application
    status events are emitted during tool execution; tokens stream as soon as
    the model begins generating the final answer.

    SSE event format:
        data: {"type": "status", "text": "Searching knowledge base..."}
        data: {"type": "token",  "text": "<delta>"}
        data: {"type": "done",   "latency_ms": 123, "tokens_in": 50, ...}
        data: {"type": "error",  "message": "..."}
    """
    chat_service: ChatService | None = request.app.state.chat_service
    agent_deps = request.app.state.agent_dependencies

    async def event_generator():
        if chat_service is None or agent_deps is None:
            yield f'data: {json.dumps({"type": "error", "message": "RAG pipeline not initialized"})}\n\n'
            return

        t_start = time.perf_counter()
        recorded = False

        try:
            async for event in chat_service.astream(
                question=chat_request.message,
                deps=agent_deps,
            ):
                yield f"data: {json.dumps(event)}\n\n"

                if event.get("type") == "done":
                    tokens_in = event.get("tokens_in", 0)
                    tokens_out = event.get("tokens_out", 0)
                    agent_process_ms = event.get("agent_process_ms", 0.0)
                    total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
                    metrics_store.record_request(
                        question=chat_request.message,
                        total_latency_ms=total_latency_ms,
                        agent_process_ms=agent_process_ms,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        success=True,
                        model=settings.model_name,
                    )
                    recorded = True
                elif event.get("type") == "error":
                    total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
                    metrics_store.record_request(
                        question=chat_request.message,
                        total_latency_ms=total_latency_ms,
                        agent_process_ms=event.get("agent_process_ms", 0),
                        tokens_in=0,
                        tokens_out=0,
                        success=False,
                        error_msg=event.get("message", "Unknown error"),
                        model=settings.model_name,
                    )
                    recorded = True

        except Exception as exc:
            logger.error(f"Streaming chat error: {exc}")
            yield f'data: {json.dumps({"type": "error", "message": "Model generation failed."})}\n\n'
            if not recorded:
                total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
                metrics_store.record_request(
                    question=chat_request.message,
                    total_latency_ms=total_latency_ms,
                    agent_process_ms=0,
                    tokens_in=0,
                    tokens_out=0,
                    success=False,
                    error_msg=str(exc),
                    model=settings.model_name,
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
