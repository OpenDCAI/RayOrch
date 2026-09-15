"""Input identity and historical comparability checks for MinerU runs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path


MINERU_GOLDEN_MEASURED_S = 587.781
MINERU_GOLDEN_PDFS = 368
MINERU_GOLDEN_PAGES = 7_072


def load_pdf_paths(
    manifest_path: str,
    *,
    limit: int | None = None,
) -> tuple[str, ...]:
    """Load ordered PDF paths from a JSON array or JSONL manifest."""

    raw_text = Path(manifest_path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        entries = [
            json.loads(line)
            for line in raw_text.splitlines()
            if line.strip()
        ]
    else:
        entries = parsed if isinstance(parsed, list) else [parsed]

    paths: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"manifest entry {index} must contain a path")
        path = os.path.abspath(str(entry["path"]))
        if path in seen:
            raise ValueError(f"manifest entry {index} duplicates {path}")
        seen.add(path)
        paths.append(path)
        if limit is not None and limit > 0 and len(paths) >= limit:
            break
    if not paths:
        raise ValueError("PDF manifest is empty")
    return tuple(paths)


def validate_inputs(
    pdfs: tuple[str, ...],
    *,
    flash_repo: str,
    model: str,
    packaged_runtime: bool = False,
) -> None:
    """Fail before Ray startup when required workload inputs are absent."""

    missing_pdfs = [path for path in pdfs if not Path(path).is_file()]
    if missing_pdfs:
        preview = ", ".join(missing_pdfs[:3])
        suffix = "" if len(missing_pdfs) <= 3 else ", ..."
        raise FileNotFoundError(f"PDF inputs do not exist: {preview}{suffix}")
    if not packaged_runtime and not Path(flash_repo).is_dir():
        raise FileNotFoundError(f"Flash-MinerU repository does not exist: {flash_repo}")
    if not Path(model).exists():
        raise FileNotFoundError(f"MinerU model does not exist: {model}")
    stems = [Path(path).stem for path in pdfs]
    if len(stems) != len(set(stems)):
        raise ValueError("PDF filenames must have unique stems")


def corpus_digest(pdfs: Iterable[str]) -> str:
    """Hash ordered PDF names and content without depending on mount paths."""

    digest = hashlib.sha256()
    for path in pdfs:
        absolute = os.path.abspath(path)
        digest.update(Path(absolute).name.encode("utf-8"))
        digest.update(b"\0")
        content = hashlib.sha256()
        with open(absolute, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                content.update(chunk)
        digest.update(content.digest())
        digest.update(b"\0")
    return digest.hexdigest()


def golden_comparable(
    *,
    corpus_digest: str,
    expected_digest: str | None,
    pdf_count: int,
    page_count: int,
) -> bool:
    """Return whether a run is comparable with the recorded MinerU baseline."""

    if expected_digest is None:
        return False
    expected = expected_digest.lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError("--golden-corpus-sha256 must be a 64-character hex digest")
    return (
        corpus_digest == expected
        and pdf_count == MINERU_GOLDEN_PDFS
        and page_count == MINERU_GOLDEN_PAGES
    )


__all__ = [
    "MINERU_GOLDEN_MEASURED_S",
    "MINERU_GOLDEN_PAGES",
    "MINERU_GOLDEN_PDFS",
    "corpus_digest",
    "golden_comparable",
    "load_pdf_paths",
    "validate_inputs",
]
