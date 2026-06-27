"""
index_bootstrap.py — Bundled index detection and bootstrap for deployment.

On HuggingFace Spaces the runtime index lives on the persistent volume
(INDEX_STORAGE_PATH=/data/index). A pre-built index can be shipped in the
repo at data/index so the first cold start loads embeddings instead of
re-indexing all documents (2–5 min on CPU).

Flow at startup:
    1. Validate the runtime index path (e.g. /data/index).
    2. If invalid/missing, validate the bundled path (data/index in the repo).
    3. If bundled is valid, copy it to the runtime path.
    4. IngestionPipeline then takes the normal load path.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from app.config import Settings

logger = logging.getLogger(__name__)

_REQUIRED_FILES = ("docstore.json", "index_store.json")
_VECTOR_STORE_SUFFIX = "__vector_store.json"


def bundled_index_path(settings: Settings) -> Path:
    """Path to the index shipped with the repo (default: {data_dir}/index)."""
    if settings.bundled_index_path:
        return Path(settings.bundled_index_path)
    return Path(settings.data_dir) / "index"


def runtime_index_path(settings: Settings) -> Path:
    return Path(settings.index_storage_path)


def _vector_store_files(index_path: Path) -> list[Path]:
    """Return vector-store JSON files for a LlamaIndex SimpleVectorStore persist dir."""
    files: list[Path] = []
    legacy = index_path / "vector_store.json"
    if legacy.is_file():
        files.append(legacy)
    files.extend(sorted(index_path.glob(f"*{_VECTOR_STORE_SUFFIX}")))
    return files


def _docstore_has_nodes(docstore_path: Path) -> bool:
    """Return True when docstore.json contains at least one stored node/document."""
    data = json.loads(docstore_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return False

    for collection in data.values():
        if not isinstance(collection, dict):
            continue
        for key, value in collection.items():
            if key.startswith("__") or not value:
                continue
            if isinstance(value, dict) and value:
                return True
    return False


def _vector_store_has_embeddings(vector_path: Path) -> bool:
    """Return True when a vector store file contains at least one embedding."""
    data = json.loads(vector_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return False
    embeddings = data.get("embedding_dict", {})
    return isinstance(embeddings, dict) and len(embeddings) > 0


def validate_index_structure(index_path: Path) -> tuple[bool, str]:
    """
    Check that a directory looks like a complete LlamaIndex persisted index.

    Returns:
        (is_valid, reason) — reason explains failure or "ok" on success.
    """
    if not index_path.is_dir():
        return False, "not a directory"

    missing = [name for name in _REQUIRED_FILES if not (index_path / name).is_file()]
    if missing:
        return False, f"missing required file(s): {', '.join(missing)}"

    vector_files = _vector_store_files(index_path)
    if not vector_files:
        return False, "missing vector store file (*__vector_store.json or vector_store.json)"

    try:
        if not _docstore_has_nodes(index_path / "docstore.json"):
            return False, "docstore.json has no nodes"
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid docstore.json: {exc}"

    embedded = False
    for vector_file in vector_files:
        try:
            if _vector_store_has_embeddings(vector_file):
                embedded = True
                break
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"invalid {vector_file.name}: {exc}"

    if not embedded:
        return False, "vector store has no embeddings"

    return True, "ok"


def copy_index_tree(source: Path, target: Path) -> None:
    """Copy a validated index directory to the runtime storage path."""
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        dest = target / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def bootstrap_index_from_bundle(settings: Settings) -> bool:
    """
    Ensure the runtime index path is populated from a bundled index when possible.

    Returns:
        True if a bundled index was copied to the runtime path, else False.
    """
    runtime = runtime_index_path(settings)
    bundled = bundled_index_path(settings)

    if runtime.resolve() == bundled.resolve():
        valid, reason = validate_index_structure(runtime)
        if valid:
            logger.info(
                "IndexBootstrap: using bundled index at %s (%s)",
                runtime,
                reason,
            )
        else:
            logger.info(
                "IndexBootstrap: bundled index at %s is not valid (%s); will rebuild",
                runtime,
                reason,
            )
        return False

    runtime_valid, runtime_reason = (
        validate_index_structure(runtime) if runtime.exists() else (False, "missing")
    )
    if runtime_valid:
        logger.info(
            "IndexBootstrap: runtime index ready at %s (%s)",
            runtime,
            runtime_reason,
        )
        return False

    if runtime.exists():
        logger.warning(
            "IndexBootstrap: runtime index at %s is incomplete (%s)",
            runtime,
            runtime_reason,
        )

    bundled_valid, bundled_reason = validate_index_structure(bundled)
    if not bundled_valid:
        logger.info(
            "IndexBootstrap: no usable bundled index at %s (%s); will build from documents",
            bundled,
            bundled_reason,
        )
        return False

    logger.info(
        "IndexBootstrap: copying bundled index from %s → %s",
        bundled,
        runtime,
    )
    if runtime.exists():
        shutil.rmtree(runtime)
    copy_index_tree(bundled, runtime)
    logger.info("IndexBootstrap: bundled index copied successfully.")
    return True
