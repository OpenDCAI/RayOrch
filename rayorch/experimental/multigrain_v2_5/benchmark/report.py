"""Raw trial and aggregate report serialization."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..metrics import percentile


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    mode: str
    seed: int
    repetition: int
    order_index: int
    fanout_mode: str
    service_mode: str
    parent_count: int
    total_children: int
    kept_children: int
    replicas: int
    batch_size: int
    max_batch_wait_ms: float
    startup_time_s: float
    measured_wall_time_s: float
    end_to_end_wall_time_s: float
    throughput_children_s: float
    rpc_count: float
    grains_per_rpc: float
    batch_fill_ratio: float
    tail_rpc_fraction: float
    parent_p50_s: float
    parent_p95_s: float
    parent_p99_s: float
    expand_bubble_ratio: float
    map_bubble_ratio: float
    filter_bubble_ratio: float
    reduce_bubble_ratio: float
    flush_full: float
    flush_timeout: float
    flush_port_sealed: float
    output_digest: str
    git_commit: str
    python_version: str
    ray_version: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_GROUP_FIELDS = (
    "mode",
    "fanout_mode",
    "service_mode",
    "parent_count",
    "replicas",
    "batch_size",
    "max_batch_wait_ms",
)

_SUMMARY_METRICS = (
    "measured_wall_time_s",
    "throughput_children_s",
    "rpc_count",
    "grains_per_rpc",
    "batch_fill_ratio",
    "tail_rpc_fraction",
    "parent_p50_s",
    "parent_p95_s",
    "parent_p99_s",
    "map_bubble_ratio",
    "filter_bubble_ratio",
)


def summarize_reports(
    reports: Iterable[BenchmarkReport],
) -> list[dict[str, object]]:
    """Aggregate repetitions without discarding any raw trial."""

    groups: dict[tuple[object, ...], list[BenchmarkReport]] = {}
    for report in reports:
        key = tuple(getattr(report, field) for field in _GROUP_FIELDS)
        groups.setdefault(key, []).append(report)

    summaries = []
    for key in sorted(groups, key=repr):
        trials = groups[key]
        row = {
            field: value for field, value in zip(_GROUP_FIELDS, key)
        }
        row["trials"] = len(trials)
        row["seeds"] = sorted({trial.seed for trial in trials})
        row["output_digests"] = sorted(
            {trial.output_digest for trial in trials}
        )
        for metric in _SUMMARY_METRICS:
            values = [float(getattr(trial, metric)) for trial in trials]
            row[f"{metric}_median"] = statistics.median(values)
            row[f"{metric}_p25"] = percentile(values, 0.25)
            row[f"{metric}_p75"] = percentile(values, 0.75)
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
        summaries.append(row)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for field, value in row.items()
                }
            )


def write_reports(
    reports: Iterable[BenchmarkReport],
    *,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write raw JSONL plus aggregate JSON/CSV into one experiment directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    reports = tuple(reports)
    raw_path = output_dir / "raw_trials.jsonl"
    raw_path.write_text(
        "".join(
            json.dumps(report.to_dict(), sort_keys=True) + "\n"
            for report in reports
        ),
        encoding="utf-8",
    )
    summaries = summarize_reports(reports)
    summary_json = output_dir / "summary.json"
    summary_json.write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_csv = output_dir / "summary.csv"
    _write_csv(summary_csv, summaries)
    return raw_path, summary_json, summary_csv
