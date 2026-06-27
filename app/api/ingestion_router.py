"""
api/ingestion_router.py — RAG File Ingestion API

Endpoints:
    POST   /ingest/upload   — upload .md / .json / .txt files into the data dir
    POST   /ingest/rebuild  — force a full index rebuild (async)
    GET    /ingest/status   — current index status (node count, last build time)
    DELETE /ingest/index    — delete the persisted index

DESIGN:
    The router reads app.state for the IngestionPipeline and StorageManager
    that were wired during lifespan startup. It does NOT create new instances.
    This guarantees the same objects used for startup are used for rebuilds.

SAFETY:
    - Only .md, .json, .txt extensions are accepted.
    - File size is capped at 50 MB.
    - Rebuild is idempotent — calling it while a rebuild is running returns
      a 409 Conflict so the UI can show "already rebuilding".
"""

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from app.monitoring.store import metrics_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Allowed file extensions and their target subdirectory
_ALLOWED_EXT = {
    ".md": "text",
    ".txt": "text",
    ".json": "json",
}

_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

# Simple flag to prevent concurrent rebuilds
_rebuild_lock = asyncio.Lock()
_rebuild_in_progress: bool = False
_last_rebuild_info: dict = {}


@router.get("/status")
def get_ingest_status(request: Request):
    """
    Returns current index status:
    - whether the index exists on disk
    - number of nodes in the in-memory index
    - last rebuild time
    - whether a rebuild is currently in progress
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    storage_manager = getattr(request.app.state, "storage_manager", None)
    index_node_count = getattr(request.app.state, "index_node_count", 0)
    last_build_time = getattr(request.app.state, "last_build_time", None)

    index_exists_on_disk = False
    if storage_manager is not None:
        try:
            index_exists_on_disk = storage_manager.exists()
        except Exception:
            pass

    return {
        "index_exists": index_exists_on_disk,
        "node_count": index_node_count,
        "last_build_time": last_build_time,
        "rebuild_in_progress": _rebuild_in_progress,
        "last_rebuild_info": _last_rebuild_info,
        "rag_enabled": pipeline is not None,
    }


@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Upload a document file (.md, .txt, .json) into the knowledge base data directory.

    Files are saved immediately; the index is NOT rebuilt automatically.
    Call POST /ingest/rebuild after uploading to update the index.
    """
    # Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {list(_ALLOWED_EXT)}",
        )

    # Read content
    content = await file.read()
    if len(content) > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). Max: {_MAX_FILE_SIZE_BYTES} bytes.",
        )

    # Determine target directory (relative to CWD, matching config defaults)
    subdir = _ALLOWED_EXT[suffix]
    target_dir = Path("data") / subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_name = Path(file.filename or "upload").name
    target_path = target_dir / safe_name

    with open(target_path, "wb") as f:
        f.write(content)

    logger.info(f"Uploaded file: {target_path} ({len(content)} bytes)")

    # Record to monitoring
    metrics_store.record_ingest_event(
        action="upload",
        filename=safe_name,
        success=True,
    )

    return {
        "status": "ok",
        "filename": safe_name,
        "size_bytes": len(content),
        "saved_to": str(target_path),
        "message": "File uploaded. Call POST /ingest/rebuild to update the index.",
    }


@router.post("/rebuild")
async def rebuild_index(request: Request):
    """
    Force a full index rebuild from all files in the data directory.

    This is an async operation. The endpoint triggers the rebuild in a
    background task and returns immediately with a status message.
    Returns 409 if a rebuild is already in progress.
    """
    global _rebuild_in_progress, _last_rebuild_info

    pipeline = getattr(request.app.state, "pipeline", None)

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized (embedding model may be unavailable).",
        )

    if _rebuild_in_progress:
        raise HTTPException(
            status_code=409,
            detail="A rebuild is already in progress. Please wait.",
        )

    # Launch rebuild as a background task
    _rebuild_in_progress = True

    async def _do_rebuild():
        global _rebuild_in_progress, _last_rebuild_info
        start = time.perf_counter()
        try:
            logger.info("Ingestion rebuild started (background task)...")
            index, nodes = await pipeline.force_rebuild()

            duration_ms = (time.perf_counter() - start) * 1000
            node_count = len(nodes)

            # Update app state
            request.app.state.index_node_count = node_count
            request.app.state.last_build_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            _last_rebuild_info = {
                "status": "completed",
                "duration_ms": round(duration_ms, 2),
                "node_count": node_count,
                "finished_at": request.app.state.last_build_time,
            }

            metrics_store.record_ingest_event(
                action="rebuild",
                duration_ms=duration_ms,
                node_count_after=node_count,
                success=True,
            )
            logger.info(f"Rebuild completed: {node_count} nodes in {duration_ms:.0f}ms")

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            _last_rebuild_info = {
                "status": "failed",
                "duration_ms": round(duration_ms, 2),
                "error": str(exc),
            }
            metrics_store.record_ingest_event(
                action="rebuild",
                duration_ms=duration_ms,
                success=False,
                error_msg=str(exc),
            )
            logger.error(f"Rebuild failed: {exc}")
        finally:
            _rebuild_in_progress = False

    asyncio.create_task(_do_rebuild())

    return {
        "status": "started",
        "message": "Index rebuild started in background. Poll GET /ingest/status for progress.",
    }


@router.delete("/index")
def delete_index(request: Request):
    """
    Delete the persisted index from disk. The index will be rebuilt on
    next application startup or when POST /ingest/rebuild is called.
    """
    storage_manager = getattr(request.app.state, "storage_manager", None)
    if storage_manager is None:
        raise HTTPException(status_code=503, detail="Storage manager not available.")

    try:
        storage_manager.delete()
        request.app.state.index_node_count = 0
        request.app.state.last_build_time = None

        metrics_store.record_ingest_event(action="delete", success=True)
        logger.info("Index deleted via DELETE /ingest/index")

        return {"status": "ok", "message": "Index deleted. Call POST /ingest/rebuild to rebuild."}

    except Exception as exc:
        metrics_store.record_ingest_event(action="delete", success=False, error_msg=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
