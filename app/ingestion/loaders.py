"""
ingestion/loaders.py — Document Loaders

RESPONSIBILITY: Load raw files from disk and return LlamaIndex Document objects.
Nothing else. No parsing, no chunking, no embedding.

WHY ABSTRACT BASE CLASS?
    The Open/Closed Principle: open for extension, closed for modification.
    Adding a PDFLoader tomorrow means creating one new class that implements
    BaseDocumentLoader — zero changes to IndexBuilder, Pipeline, or any caller.
    The interface is the contract; implementations are interchangeable.

WHY ASYNC?
    File I/O is naturally async (the OS does the actual reading).
    Today it's local files. Tomorrow it could be S3, GCS, or HTTP endpoints.
    Async ensures the event loop isn't blocked during file reads, and that
    multiple loaders (MD + JSON) can run concurrently via asyncio.gather().

WHY CompositeLoader?
    Composite Pattern: treat a group of loaders identically to a single loader.
    The IndexBuilder receives ONE loader — it never knows if it's loading from
    one source or five. This is transparent composition.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from app.config import Settings

logger = logging.getLogger(__name__)


class BaseDocumentLoader(ABC):
    """
    Abstract contract for all document loaders.
    Every loader must implement a single async method: load().
    """

    @abstractmethod
    async def load(self) -> list[Document]:
        """
        Load documents from the data source.

        Returns:
            list[Document]: LlamaIndex Document objects ready for parsing.
        """
        ...


class MarkdownLoader(BaseDocumentLoader):
    """
    Loads all .md files from data/text/ directory.

    Uses LlamaIndex SimpleDirectoryReader which automatically populates
    document metadata with: file_name, file_path, file_type, file_size.
    We extend this metadata with doc_type="markdown" for filtering later.
    """

    def __init__(self, settings: Settings) -> None:
        self._data_dir = Path(settings.data_dir) / "text"

    async def load(self) -> list[Document]:
        """
        Run SimpleDirectoryReader in a thread pool to avoid blocking the
        event loop (SimpleDirectoryReader is synchronous).
        """
        logger.info(f"Loading Markdown files from: {self._data_dir}")

        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(
            None,  # default thread pool executor
            self._load_sync,
        )

        # Inject doc_type into metadata for metadata filtering in retrieval
        for doc in documents:
            doc.metadata["doc_type"] = "markdown"
            doc.metadata.setdefault("source", str(self._data_dir))

        logger.info(f"Loaded {len(documents)} Markdown documents.")
        return documents

    def _load_sync(self) -> list[Document]:
        reader = SimpleDirectoryReader(
            input_dir=str(self._data_dir),
            required_exts=[".md"],
            recursive=False,
        )
        return reader.load_data()


class JSONLoader(BaseDocumentLoader):
    """
    Loads JSON knowledge base files (courses + roadmaps) and converts each
    record into a LlamaIndex Document.

    WHY CONVERT JSON TO DOCUMENTS?
        If we only use CourseRepository (exact/fuzzy search), the agent can only
        answer "what is course X?". It cannot answer "what courses are good for
        beginners interested in cybersecurity?" — that requires semantic search
        over the content.

        By converting each JSON record to a Document:
        1. The text content is embedded for semantic search
        2. All fields (track, level, duration) are preserved as metadata
           enabling metadata filtering: filter by track="soc", level="beginner"
        3. One unified retrieval path handles all knowledge types

    STRATEGY:
        Each JSON record → one Document where:
        - text = human-readable prose representation of all fields
        - metadata = all original fields for filtering + the record id
        - doc_type = "course" or "roadmap" for metadata filtering
    """

    # Map of filename → doc_type label for metadata
    _FILE_TYPE_MAP = {
        "kayfa_courses.json": "course",
        "kayfa_roadmaps.json": "roadmap",
    }

    def __init__(self, settings: Settings) -> None:
        self._data_dir = Path(settings.data_dir) / "json"

    async def load(self) -> list[Document]:
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(None, self._load_sync)
        logger.info(f"Loaded {len(documents)} JSON documents.")
        return documents

    def _load_sync(self) -> list[Document]:
        documents: list[Document] = []

        for json_file in self._data_dir.glob("*.json"):
            doc_type = self._FILE_TYPE_MAP.get(json_file.name, "json")
            logger.info(f"Loading JSON file: {json_file.name} as doc_type={doc_type}")

            with open(json_file, encoding="utf-8") as f:
                records = json.load(f)

            for record in records:
                doc = self._record_to_document(record, doc_type, json_file.name)
                documents.append(doc)

        return documents

    def _record_to_document(
        self,
        record: dict,
        doc_type: str,
        source_file: str,
    ) -> Document:
        """
        Convert a single JSON record into a Document.

        Text representation: human-readable so the embedding captures semantics.
        Metadata: all original fields for metadata filtering.
        """
        text_parts: list[str] = []

        # Build prose representation field-by-field
        for key, value in record.items():
            if value is None or value == "" or value == []:
                continue
            if isinstance(value, list):
                formatted = ", ".join(str(v) for v in value)
            else:
                formatted = str(value)
            text_parts.append(f"{key.replace('_', ' ').title()}: {formatted}")

        text = "\n".join(text_parts)

        # Metadata: preserve everything for filtering + traceability
        metadata = {k: v for k, v in record.items() if v is not None}
        metadata["doc_type"] = doc_type
        metadata["source_file"] = source_file
        metadata["record_id"] = record.get("id", "unknown")

        return Document(text=text, metadata=metadata)


class CompositeLoader(BaseDocumentLoader):
    """
    Composite Pattern: runs multiple loaders concurrently and merges results.

    WHY COMPOSITE?
        The IndexBuilder accepts ONE loader. By wrapping multiple loaders in
        CompositeLoader, we hide the complexity. IndexBuilder calls load() once
        and receives documents from ALL sources.

        This is the same pattern as a file system (a folder "is" a file for
        iteration purposes) — a CompositeLoader "is" a BaseDocumentLoader.

    WHY asyncio.gather()?
        All loaders run concurrently. If MarkdownLoader reads 12 files and
        JSONLoader reads 2 files, they do it at the same time — halving I/O time.
    """

    def __init__(self, loaders: list[BaseDocumentLoader]) -> None:
        if not loaders:
            raise ValueError("CompositeLoader requires at least one loader.")
        self._loaders = loaders

    async def load(self) -> list[Document]:
        logger.info(f"CompositeLoader: running {len(self._loaders)} loaders concurrently.")

        # Concurrent execution of all loaders
        results = await asyncio.gather(*[loader.load() for loader in self._loaders])

        # Flatten the list of lists
        documents: list[Document] = []
        for batch in results:
            documents.extend(batch)

        logger.info(f"CompositeLoader: total {len(documents)} documents loaded.")
        return documents
