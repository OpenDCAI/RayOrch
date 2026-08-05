"""公开视频 benchmark manifest 的生成与审计 CLI。

这个模块不下载或镜像整套数据集。用户先按相应数据集许可获取本地视频目录，再由本 CLI：

1. 枚举可 decode 视频；
2. 按时长分位数分层抽样，刻意保留短/中/长尾；
3. 写出 V3 与 Ray Data 共用的有序 manifest；
4. 记录实际 bytes、duration 与 sampling seed。

这样论文实验的 source identity 不依赖目录遍历顺序，也不需要把数 GB 输入提交进仓库。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .data import VideoManifestEntry, video_manifest_entry, write_video_manifest


VIDEO_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"})


@dataclass(frozen=True, slots=True)
class VideoProbe:
    """候选本地视频的轻量可排序 metadata。"""

    path: str
    duration_s: float
    bytes: int


def _candidate_paths(root: Path) -> tuple[Path, ...]:
    """稳定枚举 root 下的常见视频容器文件。"""

    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
    )


def probe_candidates(root: str) -> tuple[VideoProbe, ...]:
    """探测可由 OpenCV decode 的候选文件；坏文件直接从候选池剔除。"""

    probes = []
    for path in _candidate_paths(Path(root)):
        try:
            entry = video_manifest_entry(
                str(path),
                dataset="probe",
                split="probe",
                source_id=str(path.relative_to(root)),
            )
        except ValueError:
            continue
        if entry.duration_s is None or entry.duration_s <= 0:
            continue
        probes.append(
            VideoProbe(
                path=entry.path,
                duration_s=entry.duration_s,
                bytes=entry.bytes or 0,
            )
        )
    return tuple(probes)


def _bucket(value: float, *, edges: tuple[float, ...]) -> int:
    """返回时长所在的分层 bucket index。"""

    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def select_long_tail(
    probes: Iterable[VideoProbe],
    *,
    count: int,
    seed: int,
    duration_edges_s: tuple[float, ...] = (5.0, 15.0, 45.0),
) -> tuple[VideoProbe, ...]:
    """按短/中/长时长 bucket 轮转抽样，保留长尾而非只取最短视频。

    该规则不是自然分布估计，而是调度压力测试：在数 GB 固定总量内确保同一 run 同时包含
    快 producer 和慢 producer，使 cross-parent rebatching、tail 与 ordered Reduce 都能出现。
    """

    candidates = tuple(probes)
    if count <= 0:
        raise ValueError("count must be positive")
    if len(candidates) < count:
        raise ValueError(
            f"requested {count} videos but only {len(candidates)} decode"
        )
    rng = random.Random(seed)
    buckets: dict[int, list[VideoProbe]] = {}
    for probe in candidates:
        buckets.setdefault(
            _bucket(probe.duration_s, edges=duration_edges_s),
            [],
        ).append(probe)
    for values in buckets.values():
        rng.shuffle(values)

    selected: list[VideoProbe] = []
    cursor = {index: 0 for index in buckets}
    active = sorted(buckets)
    while active and len(selected) < count:
        next_active = []
        for index in active:
            values = buckets[index]
            position = cursor[index]
            if position < len(values) and len(selected) < count:
                selected.append(values[position])
                cursor[index] = position + 1
            if cursor[index] < len(values):
                next_active.append(index)
        active = next_active
    if len(selected) != count:
        raise RuntimeError("long-tail selector exhausted candidates")
    return tuple(selected)


def select_long_tail_by_bytes(
    probes: Iterable[VideoProbe],
    *,
    min_count: int,
    target_bytes: int,
    seed: int,
    duration_edges_s: tuple[float, ...] = (5.0, 15.0, 45.0),
) -> tuple[VideoProbe, ...]:
    """按 duration bucket 轮转取样，直至达到数 GB 输入预算。

    这用于正式容量实验：``target_bytes`` 约束实际读取的视频规模，``min_count`` 防止只选少量
    长视频而丢失跨 parent 调度压力。最后一个文件可使总大小略高于预算；不截断视频文件。
    """

    candidates = tuple(probes)
    if min_count <= 0:
        raise ValueError("min_count must be positive")
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    if len(candidates) < min_count:
        raise ValueError(
            f"requested at least {min_count} videos but only "
            f"{len(candidates)} decode"
        )

    rng = random.Random(seed)
    buckets: dict[int, list[VideoProbe]] = {}
    for probe in candidates:
        buckets.setdefault(
            _bucket(probe.duration_s, edges=duration_edges_s),
            [],
        ).append(probe)
    for values in buckets.values():
        rng.shuffle(values)

    selected: list[VideoProbe] = []
    total_bytes = 0
    cursor = {index: 0 for index in buckets}
    active = sorted(buckets)
    while active and (
        len(selected) < min_count or total_bytes < target_bytes
    ):
        next_active = []
        for index in active:
            values = buckets[index]
            position = cursor[index]
            if position < len(values) and (
                len(selected) < min_count or total_bytes < target_bytes
            ):
                selected.append(values[position])
                total_bytes += values[position].bytes
                cursor[index] = position + 1
            if cursor[index] < len(values):
                next_active.append(index)
        active = next_active
    if len(selected) < min_count or total_bytes < target_bytes:
        raise ValueError(
            "candidate pool cannot satisfy requested video count/byte budget"
        )
    return tuple(selected)


def build_manifest(
    *,
    input_root: str,
    output: str,
    dataset: str,
    split: str,
    count: int | None,
    min_count: int | None,
    target_bytes: int | None,
    seed: int,
) -> dict[str, object]:
    """probe 本地公开视频目录、分层抽样并写入可审计 manifest。"""

    root = Path(input_root).resolve()
    probes = probe_candidates(str(root))
    if target_bytes is None:
        if count is None:
            raise ValueError("count is required without target_bytes")
        selected = select_long_tail(probes, count=count, seed=seed)
    else:
        selected = select_long_tail_by_bytes(
            probes,
            min_count=min_count or 1,
            target_bytes=target_bytes,
            seed=seed,
        )
    entries = tuple(
        video_manifest_entry(
            probe.path,
            dataset=dataset,
            split=split,
            source_id=str(Path(probe.path).relative_to(root)),
            label=Path(probe.path).parent.name,
        )
        for probe in selected
    )
    write_video_manifest(entries, output)
    durations = [entry.duration_s or 0.0 for entry in entries]
    total_bytes = sum(entry.bytes or 0 for entry in entries)
    buckets = Counter(
        _bucket(duration, edges=(5.0, 15.0, 45.0))
        for duration in durations
    )
    report = {
        "schema_version": 1,
        "dataset": dataset,
        "split": split,
        "input_root": str(root),
        "output": str(Path(output).resolve()),
        "seed": seed,
        "candidate_decodable_videos": len(probes),
        "selected_videos": len(entries),
        "selected_bytes": total_bytes,
        "target_bytes": target_bytes,
        "minimum_video_count": min_count,
        "duration_s": {
            "min": min(durations),
            "max": max(durations),
            "sum": sum(durations),
            "bucket_counts_le_5_le_15_le_45_gt_45": [
                buckets[index] for index in range(4)
            ],
        },
    }
    Path(output).with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    """构造公开视频目录到 benchmark manifest 的 CLI。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="validation")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--count", type=int)
    mode.add_argument(
        "--target-gib",
        type=float,
        help="select duration-stratified videos until this byte budget is met",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="minimum independent videos when --target-gib is used",
    )
    parser.add_argument("--seed", default=20260803, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """生成 manifest/report 并打印 report JSON。"""

    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            build_manifest(
                input_root=args.input_root,
                output=args.output,
                dataset=args.dataset,
                split=args.split,
                count=args.count,
                min_count=args.min_count if args.target_gib is not None else None,
                target_bytes=(
                    round(args.target_gib * 1024**3)
                    if args.target_gib is not None
                    else None
                ),
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
