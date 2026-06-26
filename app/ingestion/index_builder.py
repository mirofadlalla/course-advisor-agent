"""
ingestion/index_builder.py — Index Builder

RESPONSIBILITY: Build a VectorStoreIndex from raw documents.
One job: load → parse → chunk → embed → index. Return the index.
Does NOT persist. Does NOT load from disk. That's StorageManager's job.

WHY DEPENDENCY INJECTION IN THE CONSTRUCTOR?
    Consider the alternative: IndexBuilder creates its own dependencies.

        class IndexBuilder:
            def __init__(self):
                self.loader = MarkdownLoader(settings)    # hidden dependency
                self.embed = HuggingFaceEmbedding(...)    # downloads 400MB here!
                self.parser = MarkdownStructuredParser()  # couples to one parser

    Problems:
    1. Testing: you MUST download the real embedding model to test indexing logic.
       With DI, you inject a MockEmbedding that returns random vectors instantly.
    2. Flexibility: want to test with only JSON? Inject JSONLoader. No code change.
    3. Hidden state: where does the embedding model come from? You can't tell.
       With DI, the constructor signature is the documentation.

    With DI:
        builder = IndexBuilder(
            loader=CompositeLoader([MarkdownLoader(s), JSONLoader(s)]),
            parser=CompositeParser(),
            chunker=SemanticChunker(s),
            embed_model=EmbeddingProvider.get(s),
            vector_store=VectorStoreFactory.create(s),
        )
    Everything is explicit. Everything is testable.

WHY ASYNC build_async()?
    loader.load() is async (file I/O, potentially network I/O).
    VectorStoreIndex construction with embeddings is CPU-heavy but can be
    offloaded to a thread pool for future scalability.
    Establishing the async pattern now costs nothing and prevents refactoring later.
"""

import asyncio
import logging

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from app.ingestion.loaders import BaseDocumentLoader
from app.ingestion.parsers import BaseDocumentParser
from app.ingestion.chunker import SemanticChunker

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Orchestrates the document → node → index pipeline.

    Receives all dependencies via constructor injection.
    Never creates its own loaders, parsers, embeddings, or vector stores.

    Dependencies (all injected):
        loader:       Knows HOW to load documents (MD, JSON, PDF, etc.)
        parser:       Knows HOW to structure documents into nodes
        chunker:      Knows HOW to size-normalize nodes
        embed_model:  Knows HOW to embed text into vectors
        vector_store: Knows WHERE to store vectors (Simple, Chroma, Qdrant)
    """

    def __init__(
        self,
        loader: BaseDocumentLoader,
        parser: BaseDocumentParser,
        chunker: SemanticChunker,
        embed_model: BaseEmbedding,
        vector_store: BasePydanticVectorStore,
    ) -> None:
        self._loader = loader
        self._parser = parser
        self._chunker = chunker
        self._embed_model = embed_model
        self._vector_store = vector_store

    async def build_async(self) -> VectorStoreIndex:
        """
        Full async ingestion pipeline:

        Step 1: Load — async, concurrent (via CompositeLoader.gather)
        Step 2: Parse — sync, CPU (structure analysis)
        Step 3: Chunk — sync, CPU (size normalization)
        Step 4: Index — sync, CPU+GPU (embedding computation)
                        this is the most expensive step

        Returns:
            VectorStoreIndex: Ready-to-query index with all documents embedded.
        """
        # Step 1: Load documents (async)
        logger.info("IndexBuilder: Step 1/4 — Loading documents...")
        documents = await self._loader.load()
        logger.info(f"IndexBuilder: Loaded {len(documents)} documents.")

        if not documents:
            raise ValueError(
                "IndexBuilder: No documents loaded. "
                "Check that data/ directory contains .md and .json files."
            )

        # Step 2: Parse documents into structured nodes (sync → thread pool)
        logger.info("IndexBuilder: Step 2/4 — Parsing documents into nodes...")
        loop = asyncio.get_event_loop()
        nodes = await loop.run_in_executor(None, self._parser.parse, documents)
        logger.info(f"IndexBuilder: Produced {len(nodes)} nodes from parsing.")

        # Step 3: Chunk nodes to uniform size (sync → thread pool)
        logger.info("IndexBuilder: Step 3/4 — Chunking nodes...")
        chunked_nodes = await loop.run_in_executor(None, self._chunker.chunk, nodes)
        logger.info(f"IndexBuilder: Produced {len(chunked_nodes)} chunks.")

        # Step 4: Build VectorStoreIndex (embedding computation)
        # This calls embed_model.get_text_embedding() for each chunk.
        # With BAAI/bge-m3 and ~500 chunks, expect ~2-5 minutes on CPU.
        logger.info(
            f"IndexBuilder: Step 4/4 — Building vector index "
            f"({len(chunked_nodes)} chunks × embed_model={self._embed_model.model_name})..."
        )
        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )
        index = await loop.run_in_executor(
            None,
            lambda: VectorStoreIndex(
                chunked_nodes,
                embed_model=self._embed_model,
                storage_context=storage_context,
                show_progress=True,
            ),
        )

        logger.info("IndexBuilder: VectorStoreIndex built successfully.")
        return index

    def get_last_nodes(self) -> list:
        """
        Returns the last set of chunked nodes for BM25 retriever initialization.
        Call this after build_async() to get nodes for BM25Retriever.
        """
        # Stored during build for BM25 access
        return getattr(self, "_last_chunked_nodes", [])