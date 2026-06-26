"""
ingestion/__init__.py

Exports the public API of the ingestion package.
Other modules import from here, not from individual submodules.
This decouples callers from internal file structure.
"""

from app.ingestion.loaders import (
    BaseDocumentLoader,
    MarkdownLoader,
    JSONLoader,
    CompositeLoader,
)
from app.ingestion.parsers import (
    BaseDocumentParser,
    MarkdownStructuredParser,
    JSONFlatParser,
)
from app.ingestion.chunker import SemanticChunker
from app.ingestion.index_builder import IndexBuilder
from app.ingestion.storage_manager import StorageManager
from app.ingestion.pipeline import IngestionPipeline

__all__ = [
    "BaseDocumentLoader",
    "MarkdownLoader",
    "JSONLoader",
    "CompositeLoader",
    "BaseDocumentParser",
    "MarkdownStructuredParser",
    "JSONFlatParser",
    "SemanticChunker",
    "IndexBuilder",
    "StorageManager",
    "IngestionPipeline",
]
