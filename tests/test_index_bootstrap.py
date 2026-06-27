"""Tests for bundled index validation and bootstrap."""

import json
import shutil
from pathlib import Path

import pytest

from app.config import Settings
from app.ingestion.index_bootstrap import (
    bootstrap_index_from_bundle,
    bundled_index_path,
    copy_index_tree,
    validate_index_structure,
)


def _write_minimal_index(index_dir: Path, *, with_embeddings: bool = True) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    docstore = {
        "docstore/data": {
            "node-1": {
                "__data__": {"id_": "node-1", "text": "Hello world", "metadata": {}},
            }
        }
    }
    (index_dir / "docstore.json").write_text(json.dumps(docstore), encoding="utf-8")
    (index_dir / "index_store.json").write_text(
        json.dumps({"index_store/data": {"index-1": {"__data__": {}}}}),
        encoding="utf-8",
    )
    embeddings = {"node-1": [0.1, 0.2, 0.3]} if with_embeddings else {}
    (index_dir / "default__vector_store.json").write_text(
        json.dumps(
            {
                "embedding_dict": embeddings,
                "text_id_to_ref_doc_id": {},
                "metadata_dict": {},
            }
        ),
        encoding="utf-8",
    )
    (index_dir / "graph_store.json").write_text('{"graph_dict": {}}', encoding="utf-8")


@pytest.fixture
def tmp_dirs(tmp_path):
    runtime = tmp_path / "runtime_index"
    bundled = tmp_path / "data" / "index"
    return runtime, bundled


class TestValidateIndexStructure:
    def test_valid_minimal_index(self, tmp_dirs):
        _, bundled = tmp_dirs
        _write_minimal_index(bundled)
        valid, reason = validate_index_structure(bundled)
        assert valid is True
        assert reason == "ok"

    def test_rejects_missing_docstore(self, tmp_dirs):
        _, bundled = tmp_dirs
        _write_minimal_index(bundled)
        (bundled / "docstore.json").unlink()
        valid, reason = validate_index_structure(bundled)
        assert valid is False
        assert "docstore.json" in reason

    def test_rejects_empty_embeddings(self, tmp_dirs):
        _, bundled = tmp_dirs
        _write_minimal_index(bundled, with_embeddings=False)
        valid, reason = validate_index_structure(bundled)
        assert valid is False
        assert "embeddings" in reason

    def test_rejects_incomplete_data_index_like_repo(self, tmp_path):
        """Current repo data/index only has graph + empty image vector store."""
        index_dir = tmp_path / "partial"
        index_dir.mkdir()
        (index_dir / "graph_store.json").write_text('{"graph_dict": {}}', encoding="utf-8")
        (index_dir / "image__vector_store.json").write_text(
            json.dumps(
                {
                    "embedding_dict": {},
                    "text_id_to_ref_doc_id": {},
                    "metadata_dict": {},
                }
            ),
            encoding="utf-8",
        )
        valid, reason = validate_index_structure(index_dir)
        assert valid is False


class TestBootstrapIndexFromBundle:
    def test_copies_bundled_to_runtime_when_runtime_missing(self, tmp_dirs):
        runtime, bundled = tmp_dirs
        _write_minimal_index(bundled)

        settings = Settings(
            groq_api_key="test-key",
            index_storage_path=str(runtime),
            data_dir=str(bundled.parent),
        )

        copied = bootstrap_index_from_bundle(settings)
        assert copied is True
        assert validate_index_structure(runtime)[0] is True
        assert (runtime / "docstore.json").exists()

    def test_skips_copy_when_runtime_already_valid(self, tmp_dirs):
        runtime, bundled = tmp_dirs
        _write_minimal_index(runtime)
        _write_minimal_index(bundled)

        settings = Settings(
            groq_api_key="test-key",
            index_storage_path=str(runtime),
            data_dir=str(bundled.parent),
        )
        copied = bootstrap_index_from_bundle(settings)
        assert copied is False

    def test_bundled_path_default(self):
        settings = Settings(groq_api_key="test-key", data_dir="data")
        assert bundled_index_path(settings) == Path("data/index")


class TestCopyIndexTree:
    def test_copies_all_files(self, tmp_dirs):
        runtime, bundled = tmp_dirs
        _write_minimal_index(bundled)
        copy_index_tree(bundled, runtime)
        assert sorted(p.name for p in runtime.iterdir()) == sorted(
            p.name for p in bundled.iterdir()
        )

    def test_overwrites_existing_runtime(self, tmp_dirs):
        runtime, bundled = tmp_dirs
        _write_minimal_index(bundled)
        runtime.mkdir()
        (runtime / "stale.txt").write_text("old", encoding="utf-8")
        shutil.rmtree(runtime)
        copy_index_tree(bundled, runtime)
        assert (runtime / "docstore.json").exists()
