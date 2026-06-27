"""
tests/test_kb_coverage.py — Knowledge base completeness checks (Week 3 Req 1).

Ensures every Kayfa source file under data/ exists on disk and is loaded
by the ingestion loaders. Prevents accidental deletion of markdown sources
while relying on a stale bundled index.
"""

import asyncio
import json
from pathlib import Path

import pytest

# Authoritative list of Kayfa knowledge-base source files (data/ directory).
EXPECTED_JSON_FILES = (
    "kayfa_courses.json",
    "kayfa_roadmaps.json",
)

EXPECTED_MARKDOWN_FILES = (
    "Kayfa_Fullstack_Diploma.md",
    "Kayfa_PenTest_Diploma.md",
    "kayfa_ai_diploma.md",
    "kayfa_company_overview.md",
    "kayfa_data_science_diploma.md",
    "kayfa_free_educational_content.md",
    "kayfa_instructor_network.md",
    "kayfa_paid_educational_tracks.md",
    "kayfa_paid_individual_courses.md",
    "kayfa_policies_and_faqs.md",
    "kayfa_privacy_policy.md",
    "kayfa_soc_diploma.md",
)


def _data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _make_settings():
    from app.config import Settings

    return Settings(groq_api_key="test-key", data_dir=str(_data_root()))


def _run(coro):
    return asyncio.run(coro)


class TestKnowledgeBaseFilesOnDisk:
    def test_json_sources_exist(self):
        json_dir = _data_root() / "json"
        missing = [name for name in EXPECTED_JSON_FILES if not (json_dir / name).is_file()]
        assert missing == [], f"Missing JSON knowledge files: {missing}"

    def test_markdown_sources_exist(self):
        text_dir = _data_root() / "text"
        missing = [name for name in EXPECTED_MARKDOWN_FILES if not (text_dir / name).is_file()]
        assert missing == [], f"Missing markdown knowledge files: {missing}"

    def test_json_files_are_non_empty_arrays(self):
        json_dir = _data_root() / "json"
        for name in EXPECTED_JSON_FILES:
            data = json.loads((json_dir / name).read_text(encoding="utf-8"))
            assert isinstance(data, list) and len(data) > 0, f"{name} must be a non-empty list"

    def test_markdown_files_have_content(self):
        text_dir = _data_root() / "text"
        for name in EXPECTED_MARKDOWN_FILES:
            text = (text_dir / name).read_text(encoding="utf-8").strip()
            assert len(text) > 50, f"{name} is empty or too short"


class TestKnowledgeBaseLoaderCoverage:
    def test_json_loader_covers_all_json_files(self):
        from app.ingestion.loaders import JSONLoader

        loader = JSONLoader(_make_settings())
        documents = _run(loader.load())

        loaded_files = {doc.metadata.get("source_file") for doc in documents}
        assert loaded_files == set(EXPECTED_JSON_FILES)

        doc_types = {doc.metadata.get("doc_type") for doc in documents}
        assert doc_types == {"course", "roadmap"}

        assert len(documents) > 0

    def test_markdown_loader_covers_all_markdown_files(self):
        from app.ingestion.loaders import MarkdownLoader

        loader = MarkdownLoader(_make_settings())
        documents = _run(loader.load())

        loaded_names = {doc.metadata.get("file_name") for doc in documents}
        assert loaded_names == set(EXPECTED_MARKDOWN_FILES)
        assert all(doc.metadata.get("doc_type") == "markdown" for doc in documents)
        assert len(documents) == len(EXPECTED_MARKDOWN_FILES)

    def test_composite_loader_merges_json_and_markdown(self):
        from app.ingestion.loaders import CompositeLoader, JSONLoader, MarkdownLoader

        loader = CompositeLoader([MarkdownLoader(_make_settings()), JSONLoader(_make_settings())])
        documents = _run(loader.load())

        md_names = {
            doc.metadata.get("file_name")
            for doc in documents
            if doc.metadata.get("doc_type") == "markdown"
        }
        json_names = {
            doc.metadata.get("source_file")
            for doc in documents
            if doc.metadata.get("doc_type") in ("course", "roadmap")
        }

        assert md_names == set(EXPECTED_MARKDOWN_FILES)
        assert json_names == set(EXPECTED_JSON_FILES)
        assert len(documents) == len(EXPECTED_MARKDOWN_FILES) + sum(
            len(json.loads((_data_root() / "json" / name).read_text(encoding="utf-8")))
            for name in EXPECTED_JSON_FILES
        )
