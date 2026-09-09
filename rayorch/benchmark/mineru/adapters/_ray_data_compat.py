"""Small row conversion helpers shared by the Ray Data adapter."""

from __future__ import annotations

from typing import Any


def _rows_from_batch(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """把 Ray Data 的 numpy batch 转为普通 row 字典列表。"""

    if not batch:
        return []
    size = len(next(iter(batch.values())))
    return [
        {name: values[index] for name, values in batch.items()}
        for index in range(size)
    ]


def _page_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """把 Ray Data 的 columnar page row 恢复为 MinerU 原始 page record。"""

    from PIL import Image  # pyright: ignore[reportMissingImports]

    image_rgb = row["image_rgb"]
    if not hasattr(image_rgb, "shape"):
        raise ValueError("image_rgb must be a numpy-compatible tensor")
    return {
        "pdf_path": str(row["pdf_path"]),
        "page_id": int(row["page_id"]),
        "img_pil": Image.fromarray(image_rgb, mode="RGB"),
        "scale": float(row["scale"]),
        "page_width": int(row["page_width"]),
        "page_height": int(row["page_height"]),
        "pdf_len": int(row["pdf_len"]),
    }


__all__ = ["_page_from_row", "_rows_from_batch"]
