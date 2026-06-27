"""
ingestion/__init__.py

Exports the public API of the ingestion package.
Other modules import from here, not from individual submodules.
This decouples callers from internal file structure.
"""

from app.ingestion.chunker import SemanticChunker
from app.ingestion.index_builder import IndexBuilder
from app.ingestion.loaders import (
    BaseDocumentLoader,
    CompositeLoader,
    JSONLoader,
    MarkdownLoader,
)
from app.ingestion.parsers import (
    BaseDocumentParser,
    JSONFlatParser,
    MarkdownStructuredParser,
)
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage_manager import StorageManager

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
