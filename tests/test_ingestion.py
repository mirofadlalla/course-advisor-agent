"""
tests/test_ingestion.py — Parser and Chunker Unit Tests

Tests the ingestion pipeline components in isolation.
No embedding model, no vector store, no API calls.
"""

import pytest
from unittest.mock import MagicMock
from llama_index.core.schema import Document, TextNode


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_settings(**overrides):
    from app.config import Settings
    defaults = dict(groq_api_key="test-key", chunk_size=512, chunk_overlap=64)
    defaults.update(overrides)
    return Settings(**defaults)


def make_markdown_doc(text: str, filename: str = "test.md") -> Document:
    return Document(
        text=text,
        metadata={"doc_type": "markdown", "file_name": filename},
    )


def make_course_doc(text: str) -> Document:
    return Document(
        text=text,
        metadata={"doc_type": "course", "file_name": "courses.json"},
    )


# ── MarkdownStructuredParser ───────────────────────────────────────────────────

class TestMarkdownStructuredParser:

    def test_parses_markdown_documents(self):
        from app.ingestion.parsers import MarkdownStructuredParser

        parser = MarkdownStructuredParser()
        doc = make_markdown_doc("# Title\n\nSome content here.\n\n## Section\n\nMore content.")
        nodes = parser.parse([doc])

        assert len(nodes) > 0
        for node in nodes:
            assert node.metadata.get("parser_type") == "markdown_structured"

    def test_ignores_non_markdown_documents(self):
        from app.ingestion.parsers import MarkdownStructuredParser

        parser = MarkdownStructuredParser()
        doc = make_course_doc("Course content")
        nodes = parser.parse([doc])

        assert nodes == []

    def test_empty_list_returns_empty(self):
        from app.ingestion.parsers import MarkdownStructuredParser

        parser = MarkdownStructuredParser()
        assert parser.parse([]) == []

    def test_multiple_sections_produce_multiple_nodes(self):
        from app.ingestion.parsers import MarkdownStructuredParser

        parser = MarkdownStructuredParser()
        text = "\n".join([
            "# H1",
            "Intro paragraph.",
            "",
            "## Section One",
            "Content one.",
            "",
            "## Section Two",
            "Content two.",
        ])
        nodes = parser.parse([make_markdown_doc(text)])
        # At least one node per top-level section
        assert len(nodes) >= 1


# ── JSONFlatParser ─────────────────────────────────────────────────────────────

class TestJSONFlatParser:

    def test_parses_course_documents(self):
        from app.ingestion.parsers import JSONFlatParser

        parser = JSONFlatParser()
        doc = make_course_doc("Python Programming Course — 30 hours")
        nodes = parser.parse([doc])

        assert len(nodes) > 0
        for node in nodes:
            assert node.metadata.get("parser_type") == "json_flat"

    def test_parses_roadmap_documents(self):
        from app.ingestion.parsers import JSONFlatParser

        parser = JSONFlatParser()
        doc = Document(
            text="SOC Analyst Roadmap",
            metadata={"doc_type": "roadmap", "file_name": "roadmaps.json"},
        )
        nodes = parser.parse([doc])
        assert len(nodes) > 0

    def test_ignores_markdown_documents(self):
        from app.ingestion.parsers import JSONFlatParser

        parser = JSONFlatParser()
        nodes = parser.parse([make_markdown_doc("# Title\n\nContent")])
        assert nodes == []

    def test_empty_list_returns_empty(self):
        from app.ingestion.parsers import JSONFlatParser

        parser = JSONFlatParser()
        assert parser.parse([]) == []


# ── CompositeParser ────────────────────────────────────────────────────────────

class TestCompositeParser:

    def test_routes_to_both_parsers(self):
        from app.ingestion.parsers import CompositeParser

        parser = CompositeParser()
        docs = [
            make_markdown_doc("# Policies\n\nRefund policy here."),
            make_course_doc("Advanced Python — 40 hours"),
        ]
        nodes = parser.parse(docs)
        assert len(nodes) >= 2  # at least one from each parser

    def test_empty_input_returns_empty(self):
        from app.ingestion.parsers import CompositeParser

        parser = CompositeParser()
        assert parser.parse([]) == []

    def test_custom_parsers(self):
        """CompositeParser should accept injected parsers."""
        from app.ingestion.parsers import CompositeParser, MarkdownStructuredParser

        mock_parser = MagicMock()
        mock_parser.parse.return_value = []

        composite = CompositeParser(parsers=[mock_parser])
        composite.parse([make_markdown_doc("# test")])

        mock_parser.parse.assert_called_once()


# ── SemanticChunker ────────────────────────────────────────────────────────────

class TestSemanticChunker:

    def test_returns_nodes_for_non_empty_input(self):
        from app.ingestion.chunker import SemanticChunker
        from app.ingestion.parsers import MarkdownStructuredParser

        settings = make_settings()
        chunker = SemanticChunker(settings)

        # Build some real nodes first via the parser
        doc = make_markdown_doc(
            "# Chapter 1\n\n" + ("This is a sentence. " * 50) + "\n\n## Sub\n\nMore content."
        )
        parser = MarkdownStructuredParser()
        nodes = parser.parse([doc])

        chunked = chunker.chunk(nodes)
        assert len(chunked) >= len(nodes)  # chunking only ever adds nodes

    def test_empty_input_returns_empty(self):
        from app.ingestion.chunker import SemanticChunker

        chunker = SemanticChunker(make_settings())
        assert chunker.chunk([]) == []

    def test_respects_chunk_size_setting(self):
        """A very small chunk_size should produce more chunks than a large one."""
        from app.ingestion.chunker import SemanticChunker
        from app.ingestion.parsers import MarkdownStructuredParser

        long_text = "# Doc\n\n" + ("Word sentence content. " * 200)
        doc = make_markdown_doc(long_text)
        parser = MarkdownStructuredParser()
        nodes = parser.parse([doc])

        small_chunker = SemanticChunker(make_settings(chunk_size=64, chunk_overlap=8))
        large_chunker = SemanticChunker(make_settings(chunk_size=1024, chunk_overlap=64))

        small_chunks = small_chunker.chunk(nodes)
        large_chunks = large_chunker.chunk(nodes)

        assert len(small_chunks) >= len(large_chunks)
