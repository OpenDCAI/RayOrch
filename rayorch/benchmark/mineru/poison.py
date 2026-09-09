"""Shared deterministic poison-page contract for MinerU experiments."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_POISON_SEED = "rayorch-mineru-poison-v1"


def worker_resource_options(name: str | None) -> dict[str, Any]:
    """Pin benchmark actors to a mounted Taiji worker group when requested."""

    if not name:
        return {}
    return {"resources": {name: 0.001}}


def gpu_memory_peaks(samples: Any, count: int) -> tuple[int, ...]:
    """Return driver-visible GPU peaks without assuming the driver has GPUs."""

    return tuple(
        max(
            (
                sample.memory_used[index]
                for sample in samples
                if index < len(sample.memory_used)
            ),
            default=0,
        )
        for index in range(count)
    )


@dataclass(frozen=True, slots=True)
class PoisonPage:
    """One exact injected bad page and the parent size used for accounting."""

    pdf_index: int
    pdf_path: str
    page_id: int
    pdf_pages: int
    cause: str

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "pdf_path": os.path.abspath(self.pdf_path),
            "pdf_name": Path(self.pdf_path).name,
        }


@dataclass(frozen=True, slots=True)
class PoisonedPage:
    """Application-level bad-page value consumed by a tolerant assembler."""

    cause: str


def pdf_page_counts(pdfs: Sequence[str]) -> tuple[int, ...]:
    """Read exact source cardinalities outside the measured execution window."""

    from pypdf import PdfReader  # pyright: ignore[reportMissingImports]

    return tuple(len(PdfReader(path, strict=False).pages) for path in pdfs)


def load_pdf_manifest(
    manifest_path: str,
    *,
    limit: int | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Load a frozen JSON/JSONL corpus without reopening PDFs on the driver."""

    entries: list[tuple[str, int]] = []
    seen: set[str] = set()
    raw_text = Path(manifest_path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        values = [
            (line_number, json.loads(raw))
            for line_number, raw in enumerate(raw_text.splitlines(), start=1)
            if raw.strip()
        ]
    else:
        if isinstance(parsed, list):
            values = list(enumerate(parsed, start=1))
        else:
            values = [(1, parsed)]
    for line_number, value in values:
        if not isinstance(value, dict):
            raise ValueError(
                f"manifest line {line_number} must contain an object"
            )
        path = os.path.abspath(str(value["path"]))
        pages = int(value["pages"])
        if pages <= 0:
            raise ValueError(
                f"manifest line {line_number} has non-positive pages"
            )
        if path in seen:
            raise ValueError(
                f"manifest line {line_number} duplicates {path}"
            )
        seen.add(path)
        entries.append((path, pages))
        if limit is not None and limit > 0 and len(entries) >= limit:
            break
    if not entries:
        raise ValueError("PDF manifest is empty")
    return (
        tuple(path for path, _ in entries),
        tuple(pages for _, pages in entries),
    )


def select_poison_pages(
    pdfs: Sequence[str],
    page_counts: Sequence[int],
    *,
    count: int,
    page_id: int = 0,
    seed: str = DEFAULT_POISON_SEED,
    exact_indices: Sequence[int] = (),
) -> tuple[PoisonPage, ...]:
    """Select a nested, reproducible PDF subset and poison one exact page each."""

    if len(pdfs) != len(page_counts):
        raise ValueError("PDF paths and page counts must align")
    if count < 0 or count > len(pdfs):
        raise ValueError("poison count is outside the selected PDF range")
    if exact_indices and count:
        raise ValueError("exact poison indices and poison count are exclusive")
    if exact_indices:
        indices = tuple(exact_indices)
    else:
        ranked = sorted(
            range(len(pdfs)),
            key=lambda index: hashlib.sha256(
                f"{seed}\0{index}\0{Path(pdfs[index]).name}".encode()
            ).digest(),
        )
        indices = tuple(ranked[:count])
    if len(set(indices)) != len(indices):
        raise ValueError("poison PDF indices must be unique")

    manifest = []
    for index in sorted(indices):
        if not 0 <= index < len(pdfs):
            raise ValueError("poison PDF index is outside the selected range")
        pages = int(page_counts[index])
        if not 0 <= page_id < pages:
            raise ValueError(
                f"poison page {page_id} is outside PDF index {index} ({pages} pages)"
            )
        path = os.path.abspath(pdfs[index])
        manifest.append(
            PoisonPage(
                pdf_index=index,
                pdf_path=path,
                page_id=page_id,
                pdf_pages=pages,
                cause=f"poisoned PDF {Path(path).name} at page {page_id}",
            )
        )
    return tuple(manifest)


__all__ = [
    "DEFAULT_POISON_SEED",
    "PoisonPage",
    "PoisonedPage",
    "load_pdf_manifest",
    "pdf_page_counts",
    "select_poison_pages",
]
