"""Full-decode and content-identity audit for large video manifests.

Metadata probing is intentionally kept separate from this expensive gate.
Manifest construction rejects containers that cannot be opened; this module
then reads every byte, decodes every frame, and reports truncated streams or
duplicate contents before a paired framework benchmark starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ClipAudit:
    path: str
    bytes: int
    digest: str
    declared_frames: int
    decoded_frames: int
    error: str | None = None

    @property
    def short_decode(self) -> bool:
        return self.declared_frames > 0 and self.decoded_frames < self.declared_frames


def _decode_and_hash(path: str) -> ClipAudit:
    """Audit one clip in an isolated process with one OpenCV thread."""

    digest = hashlib.blake2b(digest_size=16)
    try:
        with open(path, "rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        size = Path(path).stat().st_size

        # These must be set before importing cv2 in a spawned worker.
        os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
        os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
        import cv2  # pyright: ignore[reportMissingImports]

        cv2.setNumThreads(1)
        capture = cv2.VideoCapture(path)
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded = 0
        try:
            if not capture.isOpened():
                raise ValueError("OpenCV cannot open video")
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                decoded += 1
        finally:
            capture.release()
        if decoded == 0:
            raise ValueError("OpenCV decoded zero frames")
        return ClipAudit(
            path=path,
            bytes=size,
            digest=digest.hexdigest(),
            declared_frames=declared,
            decoded_frames=decoded,
        )
    except Exception as exc:  # noqa: BLE001 - audit must preserve every failure
        return ClipAudit(
            path=path,
            bytes=Path(path).stat().st_size if Path(path).is_file() else 0,
            digest=digest.hexdigest(),
            declared_frames=0,
            decoded_frames=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _manifest_rows(path: str) -> tuple[dict[str, object], ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("video manifest must be a non-empty JSON list")
    rows = tuple(raw)
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("video manifest rows must be objects")
    return rows


def _declared_bytes(row: dict[str, object]) -> int | None:
    value = row.get("bytes")
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("video manifest bytes must be an integer or null")
    return value


def audit_manifest(
    manifest: str,
    *,
    workers: int,
    output: str | None = None,
) -> dict[str, object]:
    """Fully decode and hash all manifest clips, returning a bounded report."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    rows = _manifest_rows(manifest)
    identities = tuple(
        (str(row["dataset"]), str(row["split"]), str(row["source_id"]))
        for row in rows
    )
    paths = tuple(str(row["path"]) for row in rows)
    if workers == 1:
        audits = tuple(_decode_and_hash(path) for path in paths)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            audits = tuple(pool.map(_decode_and_hash, paths, chunksize=8))

    failures = [audit for audit in audits if audit.error is not None]
    short = [audit for audit in audits if audit.short_decode]
    byte_mismatches = []
    for row, audit in zip(rows, audits):
        declared_bytes = _declared_bytes(row)
        if declared_bytes is not None and declared_bytes != audit.bytes:
            byte_mismatches.append(
                {
                    "path": audit.path,
                    "manifest_bytes": declared_bytes,
                    "actual_bytes": audit.bytes,
                }
            )
    by_content: dict[tuple[int, str], list[str]] = defaultdict(list)
    for audit in audits:
        if audit.error is None:
            by_content[(audit.bytes, audit.digest)].append(audit.path)
    duplicate_groups = [
        paths for paths in by_content.values() if len(paths) > 1
    ]
    ordered_digest = hashlib.blake2b(digest_size=16)
    for identity, item in zip(identities, audits):
        ordered_digest.update("\0".join(identity).encode())
        ordered_digest.update(item.bytes.to_bytes(8, "big"))
        ordered_digest.update(item.digest.encode())

    report = {
        "schema_version": 1,
        "manifest": str(Path(manifest).resolve()),
        "workers": workers,
        "clips": len(audits),
        "bytes": sum(audit.bytes for audit in audits),
        "decoded_frames": sum(audit.decoded_frames for audit in audits),
        "ordered_content_digest": ordered_digest.hexdigest(),
        "source_identity_unique": len(set(identities)) == len(identities),
        "decode_errors": len(failures),
        "short_decodes": len(short),
        "byte_mismatches": len(byte_mismatches),
        "content_duplicate_groups": len(duplicate_groups),
        "content_duplicate_files": sum(len(group) for group in duplicate_groups),
        "passed": (
            len(set(identities)) == len(identities)
            and not failures
            and not short
            and not byte_mismatches
            and not duplicate_groups
        ),
        "failure_examples": [asdict(item) for item in failures[:100]],
        "short_decode_examples": [asdict(item) for item in short[:100]],
        "byte_mismatch_examples": byte_mismatches[:100],
        "content_duplicate_examples": duplicate_groups[:100],
    }
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=32)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_manifest(
        args.manifest,
        workers=args.workers,
        output=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["ClipAudit", "audit_manifest", "build_parser"]
