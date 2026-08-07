"""Reproducible ~50 GB Kinetics-400 source plan for video pipeline gates.

The plan uses CVDF's official Kinetics S3 snapshot: all 20 validation shards
plus the first 13 training shards.  The frozen Content-Length values were
verified on 2026-08-06.  Default CLI behavior is plan-only; downloading and
extracting require an explicit action.

Kinetics clips originate from third-party videos.  This helper is intended for
local research benchmarking and does not grant redistribution rights.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rayorch.experimental.multigrain_v3.benchmark.video.data import (
    VideoManifestEntry,
    write_video_manifest,
)
from rayorch.experimental.multigrain_v3.benchmark.video.manifest import (
    VIDEO_SUFFIXES,
    probe_candidates,
)


_BASE = "https://s3.amazonaws.com/kinetics/400"
_VAL_SIZES = (
    1546665596,
    1577976961,
    1564443902,
    1476180406,
    1602310844,
    1526745719,
    1466433224,
    1584117306,
    1617834308,
    1527227822,
    1475265722,
    1502494589,
    1491709203,
    1568992712,
    1470548062,
    1535436476,
    1505409596,
    1441614136,
    1510521854,
    1362588801,
)
_TRAIN_SIZES = (
    1631315939,
    1622090263,
    1540666914,
    1522090712,
    1570152216,
    1490910675,
    1500063543,
    1509946610,
    1597994571,
    1589172684,
    1514029457,
    1587186529,
    1577485945,
)


@dataclass(frozen=True, slots=True)
class ArchiveSpec:
    split: str
    index: int
    url: str
    expected_bytes: int

    @property
    def filename(self) -> str:
        return f"part_{self.index}.tar.gz"


def archive_plan() -> tuple[ArchiveSpec, ...]:
    """Return the stable 20-val + 13-train archive selection."""

    return tuple(
        ArchiveSpec(
            split,
            index,
            f"{_BASE}/{split}/part_{index}.tar.gz",
            size,
        )
        for split, sizes in (("val", _VAL_SIZES), ("train", _TRAIN_SIZES))
        for index, size in enumerate(sizes)
    )


def plan_summary(root: str) -> dict[str, object]:
    archives = archive_plan()
    total = sum(item.expected_bytes for item in archives)
    return {
        "schema_version": 1,
        "dataset": "kinetics-400-cvdf-snapshot",
        "purpose": "local-research-video-pipeline-gate",
        "redistribution": "not_granted",
        "selection": "all validation shards plus train shards 0..12",
        "archive_count": len(archives),
        "expected_archive_bytes": total,
        "expected_archive_gb_decimal": total / 10**9,
        "expected_archive_gib": total / 1024**3,
        "root": str(Path(root).resolve()),
        "archives": [asdict(item) for item in archives],
    }


def _archive_path(root: Path, spec: ArchiveSpec) -> Path:
    return root / "archives" / spec.split / spec.filename


def _download_one(root: Path, spec: ArchiveSpec) -> str:
    destination = _archive_path(root, spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size != spec.expected_bytes:
            raise ValueError(f"existing archive size mismatch: {destination}")
        return str(destination)

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > spec.expected_bytes:
        raise ValueError(f"partial archive exceeds expected size: {partial}")
    request = urllib.request.Request(spec.url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request) as response:
        if offset and response.status != 206:
            offset = 0
            partial.unlink(missing_ok=True)
        mode = "ab" if offset else "wb"
        with partial.open(mode) as stream:
            shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
    if partial.stat().st_size != spec.expected_bytes:
        raise ValueError(
            f"downloaded archive size mismatch: {partial} "
            f"({partial.stat().st_size} != {spec.expected_bytes})"
        )
    partial.replace(destination)
    return str(destination)


def download_archives(
    root: str,
    *,
    workers: int = 2,
) -> tuple[str, ...]:
    """Download with size checks and resumable ``.part`` files."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    resolved = Path(root).resolve()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return tuple(pool.map(lambda spec: _download_one(resolved, spec), archive_plan()))


def _extract_one(root: Path, spec: ArchiveSpec) -> str:
    archive = _archive_path(root, spec)
    if not archive.is_file() or archive.stat().st_size != spec.expected_bytes:
        raise FileNotFoundError(f"complete archive is missing: {archive}")
    destination = root / "videos" / spec.split / f"part_{spec.index}"
    marker = destination / ".complete.json"
    if marker.is_file():
        return str(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(destination, filter="data")
    marker.write_text(
        json.dumps(
            {
                "archive": str(archive),
                "expected_bytes": spec.expected_bytes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(destination)


def extract_archives(root: str) -> tuple[str, ...]:
    """Safely extract each shard into its own identity-preserving directory."""

    resolved = Path(root).resolve()
    return tuple(_extract_one(resolved, spec) for spec in archive_plan())


def build_decodable_manifest(root: str, output: str) -> dict[str, object]:
    """Write every metadata-decodable clip with its stable shard identity.

    The official snapshot contains a small number of truncated source files.
    They are rejected during dataset preparation instead of being silently
    skipped by a benchmark worker.  ``source_id`` is the path relative to the
    extraction root, so equal basenames in different shards remain distinct.
    """

    video_root = (Path(root).resolve() / "videos")
    candidate_files = tuple(
        sorted(
            path
            for path in video_root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
    )
    probes = probe_candidates(str(video_root))
    entries = []
    for probe in probes:
        relative = Path(probe.path).relative_to(video_root)
        if len(relative.parts) < 3 or relative.parts[0] not in {"train", "val"}:
            raise ValueError(
                "Kinetics clip must live under videos/{train,val}/part_N: "
                f"{probe.path}"
            )
        entries.append(
            VideoManifestEntry(
                path=probe.path,
                dataset="kinetics-400-cvdf-snapshot",
                split=relative.parts[0],
                source_id=relative.as_posix(),
                duration_s=probe.duration_s,
                bytes=probe.bytes,
            )
        )
    write_video_manifest(entries, output)

    split_counts = Counter(entry.split for entry in entries)
    split_bytes = Counter()
    for entry in entries:
        split_bytes[entry.split] += entry.bytes or 0
    report = {
        "schema_version": 1,
        "dataset": "kinetics-400-cvdf-snapshot",
        "video_root": str(video_root),
        "output": str(Path(output).resolve()),
        "candidate_files": len(candidate_files),
        "decodable_files": len(entries),
        "rejected_files": len(candidate_files) - len(entries),
        "decodable_bytes": sum(entry.bytes or 0 for entry in entries),
        "duration_s": sum(entry.duration_s or 0.0 for entry in entries),
        "split_counts": dict(sorted(split_counts.items())),
        "split_bytes": dict(sorted(split_bytes.items())),
        "source_identity_unique": len({
            (entry.dataset, entry.split, entry.source_id) for entry in entries
        }) == len(entries),
    }
    report_path = Path(output).with_suffix(".report.json")
    _write_plan(str(report_path), report)
    return report


def _write_plan(path: str, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/tmp/mgv36-kinetics50")
    parser.add_argument("--plan-output")
    parser.add_argument(
        "--action",
        choices=("plan", "download", "extract", "manifest", "all"),
        default="plan",
    )
    parser.add_argument("--manifest-output")
    parser.add_argument("--download-workers", type=int, default=2)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = plan_summary(args.root)
    if args.plan_output:
        _write_plan(args.plan_output, payload)
    if args.action != "manifest":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.action in {"download", "all"}:
        download_archives(args.root, workers=args.download_workers)
    if args.action in {"extract", "all"}:
        extract_archives(args.root)
    if args.action == "manifest":
        if not args.manifest_output:
            raise ValueError("--manifest-output is required for manifest action")
        print(
            json.dumps(
                build_decodable_manifest(args.root, args.manifest_output),
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "ArchiveSpec",
    "archive_plan",
    "build_decodable_manifest",
    "build_parser",
    "download_archives",
    "extract_archives",
    "plan_summary",
]
