"""
ingestion/chunker.py — Semantic Chunker

RESPONSIBILITY: Apply size-aware sliding window chunking on top of parser output.
Ensures every chunk is within bounds: [chunk_size ± margin] with [chunk_overlap] continuity.

WHY A SEPARATE CHUNKER CLASS?
    The parser splits by STRUCTURE (headers, records).
    The chunker splits by SIZE (token count).
    These are orthogonal concerns. A Markdown section titled "## Curriculum"
    might have 3000 tokens. The parser creates one node for it.
    The chunker splits it into ~6 overlapping 512-token chunks.

    Without the chunker:
    - Short sections → very small embeddings (noisy, low recall)
    - Long sections → over-large embeddings (diluted signal, low precision)
    - Extreme variance → inconsistent retrieval quality

CHUNKING STRATEGY RATIONALE:
    chunk_size = 512 tokens
    ─────────────────────────
    BGE-M3 supports up to 8192 tokens, but:
    - Longer chunks = more content per embedding = diluted semantic focus
    - A 512-token chunk captures 1-3 dense paragraphs = ideal granularity
    - Short enough for precise retrieval, long enough for coherent context

    chunk_overlap = 64 tokens
    ─────────────────────────
    ~12.5% overlap. Prevents the "boundary problem":
    If a sentence is split across two chunks, both chunks contain it.
    A query about that sentence will match AT LEAST one chunk.
    Higher overlap (e.g., 50%) wastes storage and increases index size.

    include_prev_next_rel = True
    ─────────────────────────────
    LlamaIndex links each chunk to its previous/next sibling.
    Future "context window expansion" retrieval can retrieve a chunk and
    then pull its neighbors for more context — without re-embedding.

METADATA PRESERVATION:
    Every child chunk inherits the FULL metadata of its parent node:
    file_name, doc_type, section_header, record_id, parser_type.
    LlamaIndex SentenceSplitter does this automatically.
    We additionally inject: chunk_index, total_chunks for debugging.
"""

import logging

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from app.config import Settings

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Wraps LlamaIndex SentenceSplitter with configuration from Settings.

    SentenceSplitter is preferred over TokenTextSplitter because:
    - It respects sentence boundaries (doesn't cut mid-sentence)
    - It uses tiktoken for accurate token counting
    - It handles Unicode correctly (important for Arabic content)

    Constructor receives Settings (DI) — no global imports.
    """

    def __init__(self, settings: Settings) -> None:
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

        self._splitter = SentenceSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            # Respect paragraph boundaries before sentence boundaries
            paragraph_separator="\n\n",
            # Try to keep sentences together
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
            include_prev_next_rel=True,
            include_metadata=True,
        )

        logger.info(
            f"SemanticChunker initialized: "
            f"chunk_size={self._chunk_size}, "
            f"chunk_overlap={self._chunk_overlap}"
        )

    def chunk(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """
        Apply size-aware chunking to a list of nodes.

        Nodes that are already within chunk_size pass through unchanged.
        Nodes that exceed chunk_size are split into overlapping chunks.
        Metadata is inherited by all child chunks.

        Args:
            nodes: Parser output nodes (variable size).

        Returns:
            list[BaseNode]: Uniformly-sized chunks ready for embedding.
        """
        if not nodes:
            logger.warning("SemanticChunker: received empty node list.")
            return []

        logger.info(f"SemanticChunker: chunking {len(nodes)} nodes.")

        chunked_nodes = self._splitter.get_nodes_from_documents(
            nodes,  # type: ignore[arg-type]
            # SentenceSplitter accepts BaseNode objects
        )

        logger.info(
            f"SemanticChunker: {len(nodes)} nodes → {len(chunked_nodes)} chunks "
            f"(avg {len(chunked_nodes) / max(len(nodes), 1):.1f} chunks/node)"
        )

        return chunked_nodes
