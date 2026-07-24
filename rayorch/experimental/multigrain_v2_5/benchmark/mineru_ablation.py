"""Reproducible paired MinerU benchmark orchestration and summarization."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


_IMAGE_REF = re.compile(
    r"images/[0-9a-f]+\.(?:jpg|jpeg|png)",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the small JSON experiment schema."""

    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config.get("common"), dict):
        raise ValueError("config.common must be an object")
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("config.runs must be a non-empty list")
    tags: set[str] = set()
    pair_modes: dict[str, list[str]] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("every config run must be an object")
        for field in ("tag", "pair", "mode", "batch_size"):
            if field not in run:
                raise ValueError(f"config run is missing {field!r}")
        if run["mode"] not in {"parent_bound", "elastic"}:
            raise ValueError("run mode must be parent_bound or elastic")
        if run["tag"] in tags:
            raise ValueError(f"duplicate run tag: {run['tag']}")
        tags.add(run["tag"])
        pair_modes.setdefault(str(run["pair"]), []).append(str(run["mode"]))
    for pair, modes in pair_modes.items():
        if sorted(modes) != ["elastic", "parent_bound"]:
            raise ValueError(
                f"pair {pair!r} must contain one elastic and one "
                "parent_bound run"
            )
    return config


def build_run_command(
    common: dict[str, Any],
    run: dict[str, Any],
    root: Path,
    *,
    python: str = sys.executable,
    flash_repo: str = "",
    model: str = "",
) -> list[str]:
    """Build one isolated ``mineru_cli`` child-process command."""

    tag = str(run["tag"])
    command = [
        python,
        "-u",
        "-m",
        "rayorch.experimental.multigrain_v2_5.benchmark.mineru_cli",
        "--mode",
        str(run["mode"]),
        "--limit",
        str(common["limit"]),
        "--replicas",
        str(common["replicas"]),
        "--microbatch-size",
        str(common["microbatch_size"]),
        "--max-inflight-arenas",
        str(common["max_inflight_arenas"]),
        "--batch-size",
        str(run["batch_size"]),
        "--max-batch-wait-ms",
        str(common["max_batch_wait_ms"]),
        "--gpu-memory-utilization",
        str(common["gpu_memory_utilization"]),
        "--render-replicas",
        str(common["render_replicas"]),
        "--reduce-replicas",
        str(common["reduce_replicas"]),
        "--num-cpus",
        str(common["num_cpus"]),
        "--object-store-gb",
        str(common["object_store_gb"]),
        "--rss-interval-s",
        str(common.get("rss_interval_s", 1)),
        "--output-dir",
        str(root / "outputs" / tag),
        "--timeline-dir",
        str(root / "timelines" / tag),
        "--result-jsonl",
        str(root / "results" / "raw.jsonl"),
    ]
    if flash_repo:
        command.extend(("--flash-repo", flash_repo))
    if model:
        command.extend(("--model", model))
    return command


def run_experiment(args: argparse.Namespace) -> int:
    """Run the configured schedule with per-run logs and resume markers."""

    config = load_config(Path(args.config))
    root = Path(args.output_root).resolve()
    for name in ("logs", "outputs", "timelines", "results", "done"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_environment(root, flash_repo=args.flash_repo)

    environment = dict(os.environ)
    environment.update(
        {
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            "RAY_PROFILING": "1",
            "RAY_task_events_report_interval_ms": "100",
            "LOGURU_LEVEL": environment.get("LOGURU_LEVEL", "WARNING"),
        }
    )
    order_path = root / "run_order.jsonl"
    for run in config["runs"]:
        tag = str(run["tag"])
        marker = root / "done" / tag
        if marker.exists() and not args.force:
            _append_jsonl(
                order_path,
                {"time": time.time(), "event": "skip", "tag": tag},
            )
            continue
        command = build_run_command(
            config["common"],
            run,
            root,
            python=args.python,
            flash_repo=args.flash_repo,
            model=args.model,
        )
        if args.dry_run:
            print(" ".join(command))
            continue
        started = time.time()
        _append_jsonl(
            order_path,
            {
                "time": started,
                "event": "start",
                "tag": tag,
                "command": command,
            },
        )
        with (root / "logs" / f"{tag}.log").open(
            "w",
            encoding="utf-8",
        ) as log:
            completed = subprocess.run(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        finished = time.time()
        _append_jsonl(
            order_path,
            {
                "time": finished,
                "event": "end",
                "tag": tag,
                "returncode": completed.returncode,
                "wall_s": finished - started,
            },
        )
        if completed.returncode:
            raise RuntimeError(
                f"benchmark run {tag!r} failed; see "
                f"{root / 'logs' / f'{tag}.log'}"
            )
        marker.write_text(
            json.dumps(
                {
                    "tag": tag,
                    "finished_at": finished,
                    "command": command,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


def summarize_experiment(args: argparse.Namespace) -> int:
    """Summarize raw payloads, OCR packing, paired speedups, and outputs."""

    config = load_config(Path(args.config))
    root = Path(args.output_root).resolve()
    payloads = _payloads_by_tag(root / "results" / "raw.jsonl")
    rows = []
    for run in config["runs"]:
        tag = str(run["tag"])
        payload = payloads.get(tag)
        if payload is None:
            raise ValueError(f"raw result is missing run {tag!r}")
        row = {
            "tag": tag,
            "pair": run["pair"],
            "rep": int(run.get("rep", 0)),
            "mode": run["mode"],
            "batch_size": int(run["batch_size"]),
            **_selected_payload_fields(payload),
            **_timeline_summary(root / "timelines" / tag),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["batch_size"],
            row["rep"],
            row["mode"],
        )
    )
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "runs.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(results / "runs.csv", rows)

    groups = _group_summary(rows)
    (results / "summary.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pairs = _paired_summary(rows)
    (results / "pairs.json").write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(results / "pairs.csv", pairs)
    pair_summary = _pair_group_summary(pairs)
    (results / "pair_summary.json").write_text(
        json.dumps(pair_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.skip_correctness:
        comparisons = root / "comparisons"
        comparisons.mkdir(parents=True, exist_ok=True)
        correctness = _correctness_summary(root, config["runs"])
        (comparisons / "correctness_summary.json").write_text(
            json.dumps(correctness, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(pair_summary, ensure_ascii=False, indent=2))
    return 0


def _selected_payload_fields(payload: dict[str, Any]) -> dict[str, Any]:
    names = (
        "startup_s",
        "measured_wall_s",
        "end_to_end_wall_s",
        "pages_per_s",
        "pdf_per_s",
        "rpc_count",
        "batch_fill_ratio",
        "tail_rpc_fraction",
        "ocr_bubble_ratio",
        "parent_p50_s",
        "parent_p95_s",
        "parent_p99_s",
        "driver_rss_peak",
        "worker_rss_peak",
        "live_blocks_across_arenas_high_watermark",
        "docs",
        "pages",
    )
    return {name: payload[name] for name in names}


def _timeline_summary(root: Path) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in (root / "dispatch_timeline.jsonl").read_text().splitlines()
    ]
    ocr = [event for event in events if event["node"] == 2]
    sizes = Counter(event["grains"] for event in ocr)
    flushes = Counter(event["flush_reason"] for event in ocr)
    grains = sum(event["grains"] for event in ocr)
    result = {
        "ocr_rpc_count": len(ocr),
        "ocr_grains": grains,
        "ocr_mean_grains_per_rpc": grains / len(ocr),
        "ocr_size_hist": dict(sorted(sizes.items())),
        "ocr_flush_hist": dict(sorted(flushes.items())),
    }
    timed = [
        event
        for event in ocr
        if event["worker_started_at"] is not None
        and event["worker_finished_at"] is not None
    ]
    if timed:
        start = min(event["worker_started_at"] for event in timed)
        stop = max(event["worker_finished_at"] for event in timed)
        busy = sum(
            event["worker_finished_at"] - event["worker_started_at"]
            for event in timed
        )
        result["ocr_span_s"] = stop - start
        result["ocr_actor_capacity_util"] = busy / ((stop - start) * 4)
        gpu = _gpu_summary(root / "gpu_samples.jsonl", start, stop)
        result.update(gpu)
    return result


def _gpu_summary(path: Path, start: float, stop: float) -> dict[str, float]:
    means = []
    all_busy = []
    for line in path.read_text().splitlines():
        sample = json.loads(line)
        if not start <= sample["monotonic_s"] <= stop:
            continue
        values = [
            device["utilization_percent"]
            for device in sorted(
                sample["devices"],
                key=lambda device: device["index"],
            )
            if device["utilization_percent"] is not None
        ]
        if len(values) == 4:
            means.append(sum(values) / 4)
            all_busy.append(all(value > 0 for value in values))
    return {
        "gpu_mean_util": statistics.fmean(means) if means else 0.0,
        "gpu_all4_busy_fraction": (
            statistics.fmean(all_busy) if all_busy else 0.0
        ),
    }


def _payloads_by_tag(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for line in path.read_text().splitlines():
        payload = json.loads(line)
        result[Path(payload["output_dir"]).name] = payload
    return result


def _group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    groups = sorted({(row["batch_size"], row["mode"]) for row in rows})
    for batch_size, mode in groups:
        members = [
            row
            for row in rows
            if row["batch_size"] == batch_size and row["mode"] == mode
        ]
        item: dict[str, Any] = {
            "batch_size": batch_size,
            "mode": mode,
            "n": len(members),
        }
        for name in (
            "measured_wall_s",
            "pages_per_s",
            "ocr_rpc_count",
            "ocr_mean_grains_per_rpc",
            "ocr_actor_capacity_util",
            "gpu_mean_util",
            "worker_rss_peak",
            "live_blocks_across_arenas_high_watermark",
        ):
            values = [float(member[name]) for member in members]
            item[f"{name}_median"] = statistics.median(values)
            item[f"{name}_mean"] = statistics.fmean(values)
            item[f"{name}_min"] = min(values)
            item[f"{name}_max"] = max(values)
            item[f"{name}_stdev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        result.append(item)
    return result


def _paired_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for pair in sorted({row["pair"] for row in rows}):
        members = [row for row in rows if row["pair"] == pair]
        parent = next(row for row in members if row["mode"] == "parent_bound")
        elastic = next(row for row in members if row["mode"] == "elastic")
        result.append(
            {
                "pair": pair,
                "rep": parent["rep"],
                "batch_size": parent["batch_size"],
                "parent_bound_wall_s": parent["measured_wall_s"],
                "elastic_wall_s": elastic["measured_wall_s"],
                "speedup_parent_over_elastic": (
                    parent["measured_wall_s"] / elastic["measured_wall_s"]
                ),
                "wall_reduction_percent": (
                    1
                    - elastic["measured_wall_s"]
                    / parent["measured_wall_s"]
                )
                * 100,
            }
        )
    return result


def _pair_group_summary(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for batch_size in sorted({pair["batch_size"] for pair in pairs}):
        members = [
            pair for pair in pairs if pair["batch_size"] == batch_size
        ]
        item = {"batch_size": batch_size, "n": len(members)}
        for name in ("speedup_parent_over_elastic", "wall_reduction_percent"):
            values = [float(member[name]) for member in members]
            item[f"{name}_median"] = statistics.median(values)
            item[f"{name}_mean"] = statistics.fmean(values)
            item[f"{name}_min"] = min(values)
            item[f"{name}_max"] = max(values)
            item[f"{name}_stdev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        result.append(item)
    return result


def _correctness_summary(
    root: Path,
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for pair in sorted({run["pair"] for run in runs}):
        members = [run for run in runs if run["pair"] == pair]
        parent = next(run for run in members if run["mode"] == "parent_bound")
        elastic = next(run for run in members if run["mode"] == "elastic")
        parent_docs = _markdown_files(root / "outputs" / parent["tag"])
        elastic_docs = _markdown_files(root / "outputs" / elastic["tag"])
        common = sorted(set(parent_docs) & set(elastic_docs))
        values = [
            _token_jaccard(
                _normalize(parent_docs[stem].read_text(errors="replace")),
                _normalize(elastic_docs[stem].read_text(errors="replace")),
            )
            for stem in common
        ]
        result.append(
            {
                "pair": pair,
                "parent_count": len(parent_docs),
                "elastic_count": len(elastic_docs),
                "matched": len(common),
                "only_parent": sorted(set(parent_docs) - set(elastic_docs)),
                "only_elastic": sorted(set(elastic_docs) - set(parent_docs)),
                "jaccard_min": min(values),
                "jaccard_median": statistics.median(values),
                "jaccard_mean": statistics.fmean(values),
                "ge_0_98": sum(value >= 0.98 for value in values),
                "ge_0_95": sum(value >= 0.95 for value in values),
            }
        )
    return result


def _markdown_files(root: Path) -> dict[str, Path]:
    return {path.stem: path for path in root.rglob("*.md")}


def _normalize(value: str) -> str:
    return _WHITESPACE.sub(" ", _IMAGE_REF.sub("images/IMG", value)).strip()


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _write_environment(root: Path, *, flash_repo: str = "") -> None:
    payload: dict[str, Any] = {
        "created_at": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import ray

        payload["ray"] = ray.__version__
    except Exception:
        pass
    try:
        import vllm

        payload["vllm"] = vllm.__version__
    except Exception:
        pass
    commands = [("rayorch_commit", ["git", "rev-parse", "HEAD"])]
    flash_root = flash_repo or os.environ.get("FLASH_MINERU_ROOT", "")
    if flash_root:
        commands.append(
            (
                "flash_mineru_commit",
                ["git", "-C", flash_root, "rev-parse", "HEAD"],
            )
        )
    for name, command in commands:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            payload[name] = completed.stdout.strip()
    (root / "environment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flattened = []
    for row in rows:
        flattened.append(
            {
                key: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or summarize paired MinerU V2.5 ablations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--flash-repo", default="")
    run.add_argument("--model", default="")
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--force", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(function=run_experiment)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--config", required=True)
    summarize.add_argument("--output-root", required=True)
    summarize.add_argument("--skip-correctness", action="store_true")
    summarize.set_defaults(function=summarize_experiment)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
