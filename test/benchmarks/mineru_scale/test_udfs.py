"""Dependency-free tests for MinerU scale UDF helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rayorch.benchmarks.mineru_scale.udfs import (
    MinerUScalePdfMetadata,
    _local_committed_document,
    is_hdfs_uri,
    pdf_stem,
)


def test_pdf_stems_cover_local_and_hdfs_paths():
    assert is_hdfs_uri("hdfs://namenode/data/paper.pdf")
    assert not is_hdfs_uri("/data/paper.pdf")
    assert pdf_stem("hdfs://namenode/data/paper.pdf") == "paper"
    assert pdf_stem("/data/paper.pdf") == "paper"
    assert MinerUScalePdfMetadata().run(
        ["hdfs://namenode/data/a.pdf", "/data/b.pdf"]
    ) == ["a", "b"]


def test_pdf_stem_rejects_unsafe_names():
    with pytest.raises(ValueError, match="unsafe"):
        pdf_stem("hdfs://namenode/data/.pdf")


def test_local_commit_requires_complete_valid_artifacts(tmp_path: Path):
    root = tmp_path / "paper"
    markdown = root / "vlm" / "paper.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text("hello", encoding="utf-8")
    (root / "vlm" / "layout.json").write_text(
        json.dumps({"pdf_info": []}),
        encoding="utf-8",
    )
    (root / "_SUCCESS").write_text(
        json.dumps(
            {
                "pdf": "paper",
                "status": "completed",
                "pages": 2,
                "chars": 5,
            }
        ),
        encoding="utf-8",
    )

    assert _local_committed_document(root, "paper", "vlm") == {
        "chars": 5,
        "pages": 2,
        "status": "completed",
    }
    (root / ".rayorch-incomplete").write_text("partial", encoding="utf-8")
    assert _local_committed_document(root, "paper", "vlm") is None
