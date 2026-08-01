"""比较两个 MinerU output roots 下的 Markdown 文档集合。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _markdowns(root: str) -> dict[str, str]:
    """按 PDF stem 收集 `<stem>/<method>/<stem>.md` 或任意递归 Markdown。"""

    values = {}
    for path in Path(root).rglob("*.md"):
        values[path.stem] = path.read_text(encoding="utf-8", errors="replace")
    return values


def _tokens(text: str) -> set[str]:
    """规范化为小写 whitespace token set。"""

    return set(text.lower().split())


def _jaccard(left: str, right: str) -> float:
    """计算两个 Markdown token sets 的 Jaccard。"""

    lhs = _tokens(left)
    rhs = _tokens(right)
    union = lhs | rhs
    return len(lhs & rhs) / len(union) if union else 1.0


def compare(left_root: str, right_root: str) -> dict[str, Any]:
    """比较文档集合、缺失项和逐文档 token Jaccard。"""

    left = _markdowns(left_root)
    right = _markdowns(right_root)
    common = sorted(left.keys() & right.keys())
    scores = [_jaccard(left[name], right[name]) for name in common]
    return {
        "left_documents": len(left),
        "right_documents": len(right),
        "matched_documents": len(common),
        "missing_from_left": sorted(right.keys() - left.keys()),
        "missing_from_right": sorted(left.keys() - right.keys()),
        "jaccard_min": min(scores) if scores else None,
        "jaccard_mean": statistics.mean(scores) if scores else None,
        "jaccard_median": statistics.median(scores) if scores else None,
        "jaccard_ge_0_95": sum(score >= 0.95 for score in scores),
        "jaccard_ge_0_98": sum(score >= 0.98 for score in scores),
    }


def build_parser() -> argparse.ArgumentParser:
    """构造 correctness CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行比较并打印/写入 JSON。"""

    args = build_parser().parse_args(argv)
    text = json.dumps(
        compare(args.left, args.right),
        ensure_ascii=False,
        indent=2,
    )
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
