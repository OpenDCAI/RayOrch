"""Importable slow-test operators over real MinerU image objects.

The operators intentionally model MinerU-shaped work without importing vLLM:
records carry bounded ``PIL.Image`` payloads while every ``run`` method remains
value-pure and independent of framework identity/lineage.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from PIL import Image

from rayorch.runtime import BadRecordError


DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get(
        "RAYORCH_MINERU_ARTIFACT_ROOT",
        "/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/"
        "workspace/Flash-mineru/outputs_regression_baseline_verify",
    )
)

_PREFERRED_DOCS = (
    "1838_reformer_the_efficient_transfo",
    "2410.19313v1",
    "2411.10741v1",
)


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _first_text(block: dict[str, Any]) -> str:
    for item in _walk_dicts(block):
        content = item.get("content") or item.get("html")
        if content:
            return str(content)
    return ""


def _image_paths(block: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for item in _walk_dicts(block):
        path = item.get("image_path")
        if path and path not in paths:
            paths.append(str(path))
    return paths


def _bounded_image(path: Path, max_side: int = 192) -> tuple[Image.Image, int]:
    with Image.open(path) as source:
        original_pixels = source.width * source.height
        image = source.convert("RGB")
        image.thumbnail((max_side, max_side))
        return image.copy(), original_pixels


def load_artifact_docs(
    root: str | Path = DEFAULT_ARTIFACT_ROOT,
    *,
    max_docs: int = 3,
    max_pages_per_doc: int = 3,
) -> list[dict[str, Any]]:
    """Build a small deterministic object corpus from prior MinerU outputs."""
    root = Path(root)
    docs: list[dict[str, Any]] = []
    candidates = [
        root / stem / "vlm"
        for stem in _PREFERRED_DOCS
        if (root / stem / "vlm" / "layout.json").is_file()
    ]
    for vlm_dir in candidates[:max_docs]:
        payload = json.loads((vlm_dir / "layout.json").read_text())
        pages: list[dict[str, Any]] = []
        for page_index, page_info in enumerate(payload.get("pdf_info", [])):
            blocks = list(page_info.get("para_blocks", []))
            image_paths: list[str] = []
            for block in blocks:
                for path in _image_paths(block):
                    if path not in image_paths:
                        image_paths.append(path)
            existing = [
                vlm_dir / "images" / path
                for path in image_paths
                if (vlm_dir / "images" / path).is_file()
            ]
            if not existing:
                continue

            page_key = f"{vlm_dir.parent.name}#p{page_index}"
            records: list[dict[str, Any]] = []
            # Two visual rows plus two evidence rows create a genuine 2x2 M:N
            # key-join on at least one page without fabricating image payloads.
            for visual_index, path in enumerate(existing[:2]):
                image, original_pixels = _bounded_image(path)
                records.append(
                    {
                        "doc": vlm_dir.parent.name,
                        "page": page_index,
                        "page_key": page_key,
                        "block": visual_index,
                        "role": "visual",
                        "kind": "image",
                        "bbox": [0, 0, image.width, image.height],
                        "content": path.name,
                        "image": image,
                        # Preserve the real pre-thumbnail pixel estimate while
                        # keeping the object payload itself bounded.
                        "work": max(1, original_pixels),
                    }
                )

            texts = [
                (block, _first_text(block))
                for block in blocks
                if _first_text(block)
            ][:2]
            for evidence_index, (block, text) in enumerate(texts):
                image, _ = _bounded_image(existing[0])
                records.append(
                    {
                        "doc": vlm_dir.parent.name,
                        "page": page_index,
                        "page_key": page_key,
                        "block": len(records) + evidence_index,
                        "role": "evidence",
                        "kind": str(block.get("type", "text")),
                        "bbox": list(block.get("bbox", [0, 0, 1, 1])),
                        "content": text[:256],
                        "image": image,
                        "work": max(1, len(text) * 256),
                    }
                )
            pages.append(
                {
                    "doc": vlm_dir.parent.name,
                    "page": page_index,
                    "page_key": page_key,
                    "blocks": records,
                }
            )
            if len(pages) >= max_pages_per_doc:
                break

        if pages:
            if not docs:
                # A real document with an empty logical page exercises zero-child
                # Expand without introducing a synthetic image payload.
                pages.append(
                    {
                        "doc": vlm_dir.parent.name,
                        "page": len(pages),
                        "page_key": f"{vlm_dir.parent.name}#empty",
                        "blocks": [],
                    }
                )
            docs.append({"name": vlm_dir.parent.name, "pages": pages})
    return docs


class DocsToPages:
    def run(self, docs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        return [list(doc["pages"]) for doc in docs]


class PagesToBlocks:
    """Multi-output Expand with one shared parent/ordinal relation."""

    def run(
        self,
        pages: list[dict[str, Any]],
    ) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
        blocks = [list(page["blocks"]) for page in pages]
        metadata = [
            [
                {
                    "doc": block["doc"],
                    "page": block["page"],
                    "page_key": block["page_key"],
                    "block": block["block"],
                    "role": block["role"],
                    "kind": block["kind"],
                }
                for block in group
            ]
            for group in blocks
        ]
        return blocks, metadata


def _fingerprint(image: Image.Image) -> str:
    return hashlib.sha1(image.tobytes()).hexdigest()[:12]


class ImageFeature:
    def __init__(self, sleep_scale: float = 0.0) -> None:
        self.sleep_scale = float(sleep_scale)

    def run(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for block in blocks:
            if self.sleep_scale:
                time.sleep(self.sleep_scale * image_work(block))
            image = block["image"]
            out.append(
                {
                    "key": record_key(block),
                    "page_key": block["page_key"],
                    "fingerprint": _fingerprint(image),
                    "pixels": image.width * image.height,
                }
            )
        return out


class PowerLawImageFeature(ImageFeature):
    """CPU sleep proxy for nonlinear vision/OCR cost on measured artifacts."""

    def __init__(self, sleep_scale: float = 0.0) -> None:
        super().__init__(0.0)
        self.power_sleep_scale = float(sleep_scale)

    def run(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for block in blocks:
            if self.power_sleep_scale:
                time.sleep(self.power_sleep_scale * power_image_work(block))
        return super().run(blocks)


class LayoutFeature:
    def run(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "key": record_key(block),
                "page_key": block["page_key"],
                "bbox": tuple(block["bbox"]),
                "kind": block["kind"],
                "content_len": len(block["content"]),
            }
            for block in blocks
        ]


class MergeFeatures:
    def run(
        self,
        images: list[dict[str, Any]],
        layouts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        out = []
        for image, layout in zip(images, layouts):
            if image["key"] != layout["key"]:
                raise ValueError("diamond fan-in paired different logical blocks")
            out.append({**image, **layout})
        return out


class KeepRole:
    def __init__(self, role: str) -> None:
        self.role = role

    def run(self, blocks: list[dict[str, Any]], *unused: Any) -> list[bool]:
        return [block["role"] == self.role for block in blocks]


class LinkVisualEvidence:
    def run(self, by_role: dict[str, dict[str, Any]]) -> dict[str, Any]:
        visual = by_role["visual"]
        evidence = by_role["evidence"]
        return {
            "page_key": visual["page_key"],
            "visual_key": record_key(visual),
            "evidence_key": record_key(evidence),
            "fingerprint": _fingerprint(visual["image"]),
            "content": evidence["content"],
        }


class AssembleDoc:
    def run(
        self,
        docs: list[dict[str, Any]],
        groups: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "doc": doc["name"],
                "keys": [item.get("key", item.get("visual_key")) for item in group],
                "count": len(group),
            }
            for doc, group in zip(docs, groups)
        ]


class TransientImageFeature(ImageFeature):
    def __init__(self, targets: str | tuple[str, ...], sleep_scale: float = 0.0) -> None:
        super().__init__(sleep_scale)
        self.targets = {targets} if isinstance(targets, str) else set(targets)
        self.seen: set[str] = set()

    def run(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for index, block in enumerate(blocks):
            key = record_key(block)
            if key in self.targets and key not in self.seen:
                self.seen.add(key)
                raise BadRecordError("transient image decode", index=index, retryable=True)
        return super().run(blocks)


class PermanentImageFeature(ImageFeature):
    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target

    def run(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for index, block in enumerate(blocks):
            if record_key(block) == self.target:
                raise BadRecordError("permanent image poison", index=index)
        return super().run(blocks)


class OpaqueImageFeature(ImageFeature):
    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target

    def run(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if any(record_key(block) == self.target for block in blocks):
            raise RuntimeError("opaque image shard failure")
        return super().run(blocks)


class AlwaysOpaqueImageFeature:
    def run(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise RuntimeError("dense opaque image shard failure")


def record_key(block: dict[str, Any]) -> str:
    return f"{block['doc']}#p{block['page']}#b{block['block']}#{block['role']}"


def image_work(block: Any) -> float:
    if not isinstance(block, dict) or "image" not in block:
        return 1.0
    # Normalize to a small sleep multiplier while retaining the real long tail.
    return max(1.0, float(block["work"]) / 200_000.0)


def power_image_work(block: Any) -> float:
    """Nonlinear cost proxy: pixels/tokens induce a power-law service tail."""
    return image_work(block) ** 3


__all__ = [
    "AssembleDoc",
    "AlwaysOpaqueImageFeature",
    "DEFAULT_ARTIFACT_ROOT",
    "DocsToPages",
    "ImageFeature",
    "KeepRole",
    "LayoutFeature",
    "LinkVisualEvidence",
    "MergeFeatures",
    "OpaqueImageFeature",
    "PagesToBlocks",
    "PermanentImageFeature",
    "PowerLawImageFeature",
    "TransientImageFeature",
    "image_work",
    "load_artifact_docs",
    "power_image_work",
    "record_key",
]
