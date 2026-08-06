"""Compact semantic diff for Docling ``documents.json.gz`` artifacts."""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter
from statistics import median
from typing import Any, Iterable, Mapping

from .core_compare import load_documents


def _tokens(markdown: str) -> list[str]:
    return markdown.lower().split()


def _token_jaccard(left: str, right: str) -> float:
    lhs = set(_tokens(left))
    rhs = set(_tokens(right))
    union = lhs | rhs
    return len(lhs & rhs) / len(union) if union else 1.0


def _top_token_delta(
    left: str,
    right: str,
    *,
    limit: int = 12,
) -> dict[str, list[tuple[str, int]]]:
    lhs = Counter(_tokens(left))
    rhs = Counter(_tokens(right))
    return {
        "removed": (lhs - rhs).most_common(limit),
        "added": (rhs - lhs).most_common(limit),
    }


def _diff_excerpt(
    left: str,
    right: str,
    *,
    limit: int = 40,
    line_chars: int = 600,
) -> list[str]:
    lines = difflib.unified_diff(
        left.splitlines(),
        right.splitlines(),
        fromfile="baseline",
        tofile="candidate",
        lineterm="",
        n=2,
    )
    return [
        line
        if len(line) <= line_chars
        else line[: line_chars - 1] + "…"
        for line in next_lines(lines, limit)
    ]


def next_lines(lines: Iterable[str], limit: int) -> Iterable[str]:
    """Yield at most ``limit`` diff lines without materializing a full diff."""

    for index, line in enumerate(lines):
        if index >= limit:
            return
        yield line


def compare_document_artifacts(
    baseline: Iterable[Mapping[str, Any]],
    candidate: Iterable[Mapping[str, Any]],
    *,
    low_limit: int = 20,
) -> dict[str, Any]:
    """Rank low-similarity documents and include bounded inspectable evidence."""

    left = tuple(baseline)
    right = tuple(candidate)
    if len(left) != len(right):
        raise ValueError("documents artifacts have different lengths")
    if low_limit <= 0:
        raise ValueError("low_limit must be positive")

    rows = []
    structure_keys = ("pages", "texts", "tables", "pictures")
    for index, (before, after) in enumerate(zip(left, right)):
        before_pdf = before.get("pdf")
        after_pdf = after.get("pdf")
        if before_pdf != after_pdf:
            raise ValueError(
                f"document order mismatch at {index}: {before_pdf!r} != {after_pdf!r}"
            )
        before_markdown = str(before.get("markdown", ""))
        after_markdown = str(after.get("markdown", ""))
        similarity = _token_jaccard(before_markdown, after_markdown)
        rows.append(
            {
                "index": index,
                "pdf": before_pdf,
                "token_jaccard": round(similarity, 8),
                "structure_exact": all(
                    before.get(key) == after.get(key) for key in structure_keys
                ),
                "structure_before": {
                    key: before.get(key) for key in structure_keys
                },
                "structure_after": {
                    key: after.get(key) for key in structure_keys
                },
                "markdown_chars_before": len(before_markdown),
                "markdown_chars_after": len(after_markdown),
                "token_delta": _top_token_delta(
                    before_markdown,
                    after_markdown,
                ),
                "diff_excerpt": _diff_excerpt(
                    before_markdown,
                    after_markdown,
                ),
            }
        )
    ranked = sorted(rows, key=lambda row: (row["token_jaccard"], row["index"]))
    similarities = [row["token_jaccard"] for row in rows]
    return {
        "document_count": len(rows),
        "token_jaccard_min": min(similarities, default=1.0),
        "token_jaccard_median": median(similarities) if similarities else 1.0,
        "markdown_exact_count": sum(
            row["token_jaccard"] == 1.0
            and row["markdown_chars_before"] == row["markdown_chars_after"]
            and not row["diff_excerpt"]
            for row in rows
        ),
        "structure_exact_count": sum(row["structure_exact"] for row in rows),
        "below_0_99_count": sum(value < 0.99 for value in similarities),
        "below_0_95_count": sum(value < 0.95 for value in similarities),
        "lowest_similarity": ranked[:low_limit],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="compare only the first N documents from each artifact",
    )
    parser.add_argument("--low-limit", type=int, default=20)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    baseline = load_documents(args.baseline)
    candidate = load_documents(args.candidate)
    if args.limit:
        baseline = baseline[: args.limit]
        candidate = candidate[: args.limit]
    report = compare_document_artifacts(
        baseline,
        candidate,
        low_limit=args.low_limit,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
