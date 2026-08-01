"""Docling 整文档 native baseline。"""

from __future__ import annotations

import time
from typing import Any

from .workload import create_converter


def run_native(
    paths: list[str],
    *,
    device: str = "cpu",
    num_threads: int = 4,
) -> tuple[tuple[dict[str, Any], ...], float]:
    """使用一个 persistent DocumentConverter 顺序转换整 PDF。"""

    converter = create_converter(
        input_format="pdf",
        device=device,
        num_threads=num_threads,
    )
    started = time.perf_counter()
    outputs = []
    for path in paths:
        markdown = converter.convert(path).document.export_to_markdown()
        outputs.append(
            {
                "markdown": markdown,
                "markdown_chars": len(markdown),
            }
        )
    return tuple(outputs), time.perf_counter() - started
