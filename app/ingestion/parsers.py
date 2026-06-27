"""
ingestion/parsers.py — Document Parsers

RESPONSIBILITY: Transform raw Document objects into structured BaseNode objects.
Parsers understand document *structure* (Markdown headers, JSON records).
They do NOT determine chunk size — that's the Chunker's job.

WHY SEPARATE PARSER FROM CHUNKER?
    Two different concerns:
    - Parser: "Where are the logical section boundaries?" (structure-aware)
    - Chunker: "Are the resulting pieces the right size?" (size-aware)

    MarkdownNodeParser splits on headers (H1/H2/H3). A section can be 50 words
    or 2000 words. The Chunker then ensures uniform size with overlap.
    Mixing them would create a class with two reasons to change.

WHY PRESERVE METADATA THROUGH PARSING?
    Every node must know where it came from.
    "Which file? Which section? What doc_type?"
    This metadata flows all the way to the SearchResult the agent receives.
    Without it, you can't explain WHY the agent answered what it answered.
"""

import logging
from abc import ABC, abstractmethod

from llama_index.core.node_parser import MarkdownNodeParser, SimpleNodeParser
from llama_index.core.schema import BaseNode, Document

logger = logging.getLogger(__name__)


class BaseDocumentParser(ABC):
    """
    Abstract contract for all document parsers.
    Receives a list of Documents, returns a list of BaseNodes.
    """

    @abstractmethod
    def parse(self, documents: list[Document]) -> list[BaseNode]:
        """
        Parse documents into nodes.

        Args:
            documents: Raw LlamaIndex Document objects from a loader.

        Returns:
            list[BaseNode]: Structured nodes ready for chunking.
        """
        ...


class MarkdownStructuredParser(BaseDocumentParser):
    """
    Parses Markdown documents using LlamaIndex MarkdownNodeParser.

    WHAT IT DOES:
        Splits documents on H1 (# ), H2 (## ), H3 (### ) headers.
        Each section becomes its own node.
        The section header is preserved in node metadata as 'header'.

    WHY HEADER-BASED SPLITTING FOR MARKDOWN?
        Our knowledge base is human-authored Markdown:
        - kayfa_policies_and_faqs.md has sections: ## Payment, ## Refunds, ## FAQ
        - kayfa_soc_diploma.md has sections: ## Curriculum, ## Instructors

        Header-based splitting gives the retriever semantically coherent chunks.
        "What is the refund policy?" retrieves the ## Refunds section, not a
        random 512-token window that might straddle two unrelated sections.

    METADATA INHERITANCE:
        All parent document metadata is inherited by child nodes automatically
        via LlamaIndex's node_parser architecture.
        We add: section_depth (H1/H2/H3), parser_type for traceability.
    """

    def __init__(self) -> None:
        self._parser = MarkdownNodeParser()

    def parse(self, documents: list[Document]) -> list[BaseNode]:
        markdown_docs = [doc for doc in documents if doc.metadata.get("doc_type") == "markdown"]

        if not markdown_docs:
            logger.warning("MarkdownStructuredParser: no markdown documents found.")
            return []

        logger.info(f"Parsing {len(markdown_docs)} Markdown documents into nodes.")
        nodes = self._parser.get_nodes_from_documents(markdown_docs)

        # Tag each node with parser identity for observability
        for node in nodes:
            node.metadata["parser_type"] = "markdown_structured"

        logger.info(f"MarkdownStructuredParser produced {len(nodes)} nodes.")
        return nodes


class JSONFlatParser(BaseDocumentParser):
    """
    Passes JSON documents through without structural splitting.

    WHY NO SPLITTING FOR JSON?
        Each JSON record was already converted to ONE Document in JSONLoader.
        A course record is ~200-400 tokens — well within chunk_size=512.
        Splitting it would break the semantic unit (a course is one thing).

        If a record exceeds chunk_size, the Chunker will handle it.
        The parser's job here is only to convert Documents → Nodes,
        which is a trivial 1:1 mapping with metadata enrichment.

    METADATA ENRICHMENT:
        We ensure doc_type (course/roadmap) is present on every node.
        This enables metadata filtering: "only search courses" or "only roadmaps".
    """

    def __init__(self) -> None:
        # SimpleNodeParser: 1 Document → 1 Node (no splitting)
        self._parser = SimpleNodeParser.from_defaults(
            chunk_size=None,  # Do not split — let Chunker handle it
            chunk_overlap=0,
        )

    def parse(self, documents: list[Document]) -> list[BaseNode]:
        json_docs = [
            doc for doc in documents if doc.metadata.get("doc_type") in ("course", "roadmap")
        ]

        if not json_docs:
            logger.warning("JSONFlatParser: no JSON documents found.")
            return []

        logger.info(f"Parsing {len(json_docs)} JSON documents into nodes.")
        nodes = self._parser.get_nodes_from_documents(json_docs)

        for node in nodes:
            node.metadata["parser_type"] = "json_flat"

        logger.info(f"JSONFlatParser produced {len(nodes)} nodes.")
        return nodes


class CompositeParser(BaseDocumentParser):
    """
    Runs multiple parsers and merges their output.

    Routes documents to the appropriate parser based on doc_type:
    - markdown docs → MarkdownStructuredParser
    - course/roadmap docs → JSONFlatParser

    WHY NOT ONE PARSER FOR EVERYTHING?
        Different document structures require different splitting strategies.
        A single parser that handles all types would violate SRP.
        The Composite pattern lets each parser specialize while the caller
        sees one unified interface.
    """

    def __init__(
        self,
        parsers: list[BaseDocumentParser] | None = None,
    ) -> None:
        # Default: use both parsers
        self._parsers = parsers or [
            MarkdownStructuredParser(),
            JSONFlatParser(),
        ]

    def parse(self, documents: list[Document]) -> list[BaseNode]:
        logger.info(f"CompositeParser: running {len(self._parsers)} parsers.")
        all_nodes: list[BaseNode] = []

        for parser in self._parsers:
            nodes = parser.parse(documents)
            all_nodes.extend(nodes)

        logger.info(f"CompositeParser: total {len(all_nodes)} nodes produced.")
        return all_nodes
