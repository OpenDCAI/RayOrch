"""Tests for compact Docling output inspection reports."""

from __future__ import annotations

import pytest

from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_output_diff import (
    build_parser,
    compare_document_artifacts,
)


def _doc(pdf: str, markdown: str, *, tables: int = 1) -> dict:
    return {
        "pdf": pdf,
        "markdown": markdown,
        "pages": 1,
        "texts": 2,
        "tables": tables,
        "pictures": 0,
    }


def test_output_diff_ranks_low_similarity_and_bounds_evidence() -> None:
    report = compare_document_artifacts(
        (_doc("a", "same"), _doc("b", "alpha beta\nsecond")),
        (_doc("a", "same"), _doc("b", "alpha gamma\nsecond", tables=2)),
        low_limit=1,
    )

    assert report["document_count"] == 2
    assert report["markdown_exact_count"] == 1
    assert report["structure_exact_count"] == 1
    assert report["token_jaccard_min"] == 0.5
    lowest = report["lowest_similarity"][0]
    assert lowest["pdf"] == "b"
    assert lowest["token_delta"] == {
        "removed": [("beta", 1)],
        "added": [("gamma", 1)],
    }
    assert lowest["diff_excerpt"]


def test_output_diff_rejects_document_order_drift() -> None:
    with pytest.raises(ValueError, match="order mismatch"):
        compare_document_artifacts(
            (_doc("a", "text"),),
            (_doc("b", "text"),),
        )


def test_output_diff_truncates_pathological_long_markdown_lines() -> None:
    report = compare_document_artifacts(
        (_doc("a", "x" * 2000),),
        (_doc("a", "y" * 2000),),
    )

    assert max(
        map(len, report["lowest_similarity"][0]["diff_excerpt"])
    ) <= 600


def test_output_diff_cli_accepts_prefix_limit() -> None:
    args = build_parser().parse_args(
        ["baseline.gz", "candidate.gz", "--limit", "48"]
    )

    assert args.limit == 48
