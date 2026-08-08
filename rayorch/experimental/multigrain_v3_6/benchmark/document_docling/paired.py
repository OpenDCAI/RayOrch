"""Paired V3↔V3.6 Docling 4/48/368 correctness/performance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

from ....multigrain_v3.benchmark.document_docling.core_compare import (
    CoreMatrixConfig,
    build_matrix_plan,
)
from ....multigrain_v3.benchmark.document_docling.core_v3 import run_v3
from ....multigrain_v3.benchmark.document_docling.gpu_monitor import GpuMonitor
from ... import RunResult
from ..paired_stats import paired_timing_summary
from .core_v36 import run_v36


_V3_ONLY_OPTIONS = frozenset(
    {
        "parse_batch_wait_ms",
        "parse_actor_concurrency",
        "layout_batch_wait_ms",
        "layout_actor_concurrency",
        "ocr_batch_wait_ms",
        "ocr_actor_concurrency",
        "table_batch_wait_ms",
        "table_actor_concurrency",
        "microbatch_size",
        "max_inflight_arenas",
        "max_pending_per_actor",
        "actor_max_concurrency",
    }
)


@dataclass(frozen=True, slots=True)
class DoclingManifest:
    """Validated ordered sources and their independently checkable totals."""

    paths: tuple[str, ...]
    pdf_ids: tuple[str, ...]
    input_bytes: int
    expected_pages: int | None
    manifest_sha256: str


def _arm_options(
    source_count: int,
    config: CoreMatrixConfig,
    batch_scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return V3 options and the explicit v3.6-supported projection."""

    arm_name = f"v3_{batch_scope}"
    matches = [
        arm
        for arm in build_matrix_plan(source_count, config)
        if arm.name == arm_name
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown Docling batch scope: {batch_scope}")
    v3_options = dict(matches[0].options)
    v36_options = {
        key: value
        for key, value in v3_options.items()
        if key not in _V3_ONLY_OPTIONS
    }
    return v3_options, v36_options


def _token_jaccard(left: str, right: str) -> float:
    lhs = set(left.lower().split())
    rhs = set(right.lower().split())
    union = lhs | rhs
    return len(lhs & rhs) / len(union) if union else 1.0


def _compare_documents(
    old: Iterable[Mapping[str, Any]],
    new: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    left = tuple(old)
    right = tuple(new)
    if len(left) != len(right):
        return {
            "document_count_matches": False,
            "identity_exact": (),
            "structure_exact": (),
            "markdown_exact": (),
            "markdown_jaccard": (),
        }
    structure_fields = ("pages", "texts", "tables", "pictures")
    return {
        "document_count_matches": True,
        "identity_exact": tuple(
            a.get("pdf") == b.get("pdf") for a, b in zip(left, right)
        ),
        "structure_exact": tuple(
            all(a.get(field) == b.get(field) for field in structure_fields)
            for a, b in zip(left, right)
        ),
        "markdown_exact": tuple(
            a.get("markdown") == b.get("markdown")
            for a, b in zip(left, right)
        ),
        "markdown_jaccard": tuple(
            round(
                _token_jaccard(
                    str(a.get("markdown", "")),
                    str(b.get("markdown", "")),
                ),
                8,
            )
            for a, b in zip(left, right)
        ),
    }


def _validate_arm_outputs(
    engine: str,
    documents: tuple[Mapping[str, Any], ...],
    manifest: DoclingManifest,
    *,
    expected_tables: int | None,
) -> dict[str, int]:
    """Require each arm to match the source contract, not only each other."""

    actual_ids = tuple(str(document.get("pdf", "")) for document in documents)
    if actual_ids != manifest.pdf_ids:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(actual_ids, manifest.pdf_ids)
                )
                if actual != expected
            ),
            min(len(actual_ids), len(manifest.pdf_ids)),
        )
        raise ValueError(
            f"{engine} output identity/order differs from manifest at "
            f"document index {mismatch}"
        )
    pages = sum(int(document["pages"]) for document in documents)
    tables = sum(int(document["tables"]) for document in documents)
    if manifest.expected_pages is not None and pages != manifest.expected_pages:
        raise ValueError(
            f"{engine} produced {pages} pages; manifest requires "
            f"{manifest.expected_pages}"
        )
    if expected_tables is not None and tables != expected_tables:
        raise ValueError(
            f"{engine} produced {tables} tables; golden requires "
            f"{expected_tables}"
        )
    return {
        "documents": len(documents),
        "pages": pages,
        "tables": tables,
    }


def _worker_summary(result: RunResult) -> dict[str, Any]:
    """Aggregate observation-only v3.6 Worker snapshots by UDF class."""

    summary: dict[str, Any] = {}
    for metrics in result.calls:
        observations = metrics.worker_snapshots
        audit_numeric: dict[str, float] = {}
        audit_text: dict[str, set[str]] = {}
        for observation in observations:
            for key, value in observation.audit:
                if isinstance(value, (int, float)):
                    audit_numeric[key] = audit_numeric.get(key, 0.0) + value
                else:
                    audit_text.setdefault(key, set()).add(value)
        summary[metrics.udf_name] = {
            "actors": len(observations),
            "calls": sum(item.lifetime_calls for item in observations),
            "rss_peak_bytes": max(
                (item.rss_bytes for item in observations),
                default=0,
            ),
            "audit": {
                **audit_numeric,
                **{
                    key: sorted(values)
                    for key, values in audit_text.items()
                },
            },
            "observation_errors": [
                item.error for item in observations if item.error is not None
            ],
        }
    return summary


def _audit_violations(worker_summary: Mapping[str, Any]) -> tuple[str, ...]:
    violations = []
    for worker, details in worker_summary.items():
        violations.extend(
            f"{worker}.observation_error={error}"
            for error in details.get("observation_errors", ())
        )
        for key, value in details.get("audit", {}).items():
            if (
                isinstance(value, (int, float))
                and value != 0
                and ("error" in key.lower() or "fallback" in key.lower())
            ):
                violations.append(f"{worker}.{key}={value}")
    return tuple(sorted(violations))


def _v3_audit_violations(metrics: Mapping[str, Any]) -> tuple[str, ...]:
    violations = []
    for key, value in metrics.items():
        if (
            key.startswith("actor_audit_stage_")
            and isinstance(value, (int, float))
            and value != 0
            and ("error" in key.lower() or "fallback" in key.lower())
        ):
            violations.append(f"{key}={value}")
    return tuple(sorted(violations))


def _gpu_summary(monitor: GpuMonitor, samples: Iterable[Any]) -> dict[str, Any]:
    rows = tuple(samples)
    by_device: dict[int, list[Any]] = {}
    for sample in rows:
        for device in sample.devices:
            by_device.setdefault(int(device.index), []).append(device)
    return {
        "available": monitor.available,
        "unavailable_reason": monitor.unavailable_reason,
        "samples": len(rows),
        "devices": {
            str(index): {
                "utilization_mean_percent": (
                    sum(item.utilization_percent or 0 for item in values)
                    / len(values)
                ),
                "utilization_max_percent": max(
                    (item.utilization_percent or 0 for item in values),
                    default=0,
                ),
                "memory_peak_bytes": max(
                    (item.memory_used_bytes or 0 for item in values),
                    default=0,
                ),
            }
            for index, values in sorted(by_device.items())
        },
    }


def _run_trial(
    manifest: DoclingManifest,
    v3_options: dict[str, Any],
    v36_options: dict[str, Any],
    *,
    microbatch_size: int,
    max_active_microbatches: int,
    order: str,
    minimum_jaccard: float,
    expected_tables: int | None,
    gpu_monitor_dir: str | None = None,
) -> dict[str, Any]:
    if order not in {"v3_first", "v36_first"}:
        raise ValueError(f"unknown trial order: {order}")
    sequence = ("v3", "v36") if order == "v3_first" else ("v36", "v3")
    paths = list(manifest.paths)
    documents: dict[str, tuple[Mapping[str, Any], ...]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for engine in sequence:
        monitor_path = (
            str(Path(gpu_monitor_dir) / f"{order}-{engine}.jsonl")
            if gpu_monitor_dir
            else None
        )
        monitor = GpuMonitor(jsonl_path=monitor_path).start()
        wall_started = time.perf_counter()
        try:
            if engine == "v3":
                result = run_v3(paths, **v3_options)
                docs = tuple(cast(Iterable[Mapping[str, Any]], result.get()))
                summaries[engine] = {
                    "startup_s": float(result.metrics["startup_time_s"]),
                    "measured_s": float(result.metrics["measured_wall_time_s"]),
                    "end_to_end_s": float(result.metrics["end_to_end_wall_time_s"]),
                    "rpc_count": int(result.metrics["rpc_count"]),
                    "metrics": dict(result.metrics),
                    "audit_violations": _v3_audit_violations(result.metrics),
                }
            else:
                result = run_v36(
                    paths,
                    microbatch_size=microbatch_size,
                    max_active_microbatches=max_active_microbatches,
                    **v36_options,
                )
                docs = tuple(
                    cast(Iterable[Mapping[str, Any]], result.outputs)
                )
                workers = _worker_summary(result)
                summaries[engine] = {
                    "measured_s": result.elapsed_s,
                    "rpc_count": result.rpc_count,
                    "actor_count": result.actor_count,
                    "released_values": result.released_values,
                    "peak_active_microbatches": result.peak_active_microbatches,
                    "workers": workers,
                    "audit_violations": _audit_violations(workers),
                }
        finally:
            samples = monitor.stop()
        summaries[engine]["outer_wall_s"] = time.perf_counter() - wall_started
        summaries[engine]["gpu_monitor"] = _gpu_summary(monitor, samples)
        summaries[engine].update(
            _validate_arm_outputs(
                engine,
                docs,
                manifest,
                expected_tables=expected_tables,
            )
        )
        documents[engine] = docs

    comparison = _compare_documents(documents["v3"], documents["v36"])
    jaccard = comparison["markdown_jaccard"]
    if not comparison["document_count_matches"]:
        raise ValueError("V3 and V3.6 document counts differ")
    if not all(comparison["identity_exact"]):
        raise ValueError("V3 and V3.6 document identity/order differs")
    if not all(comparison["structure_exact"]):
        raise ValueError("V3 and V3.6 document structures differ")
    if jaccard and min(jaccard) < minimum_jaccard:
        raise ValueError(
            f"V3↔V3.6 minimum Markdown Jaccard {min(jaccard)} "
            f"is below {minimum_jaccard}"
        )
    audit_violations = tuple(
        f"{engine}: {violation}"
        for engine in ("v3", "v36")
        for violation in summaries[engine]["audit_violations"]
    )
    if audit_violations:
        raise ValueError(
            "Docling worker audit gate failed: "
            + ", ".join(audit_violations)
        )
    comparison_summary = {
        "document_count_matches": True,
        "identity_exact_count": sum(comparison["identity_exact"]),
        "structure_exact_count": sum(comparison["structure_exact"]),
        "markdown_exact_count": sum(comparison["markdown_exact"]),
        "markdown_jaccard_min": min(jaccard, default=1.0),
        "markdown_jaccard_median": statistics.median(jaccard) if jaccard else 1.0,
        "below_threshold": sum(value < minimum_jaccard for value in jaccard),
    }
    return {
        "order": order,
        "arms": summaries,
        "correctness": comparison_summary,
    }


def _read_manifest(manifest: str, limit: int) -> DoclingManifest:
    manifest_path = Path(manifest)
    encoded = manifest_path.read_bytes()
    raw = json.loads(encoded)
    if not isinstance(raw, list):
        raise ValueError("manifest must be a JSON list")
    if limit:
        raw = raw[:limit]
    if not raw:
        raise FileNotFoundError("manifest contains no inputs")
    object_rows = [isinstance(item, dict) for item in raw]
    if any(object_rows) and not all(object_rows):
        raise ValueError("manifest must not mix path and object rows")
    if all(object_rows) and any("path" not in item for item in raw):
        raise ValueError("every manifest object must contain path")

    paths = tuple(
        str(item["path"] if isinstance(item, dict) else item)
        for item in raw
    )
    if any(not Path(path).is_file() for path in paths):
        raise FileNotFoundError("manifest contains missing PDFs")
    if len(set(paths)) != len(paths):
        raise ValueError("manifest contains duplicate PDF paths")

    pdf_ids = tuple(Path(path).stem for path in paths)
    if len(set(pdf_ids)) != len(pdf_ids):
        raise ValueError("manifest PDF stems are not unique output identities")

    metadata_rows = [item for item in raw if isinstance(item, dict)]
    pages_present = ["pages" in item for item in metadata_rows]
    if pages_present and any(pages_present) and not all(pages_present):
        raise ValueError("manifest pages metadata is only partially populated")
    expected_pages = (
        sum(int(item["pages"]) for item in metadata_rows)
        if len(metadata_rows) == len(raw) and pages_present and all(pages_present)
        else None
    )
    if expected_pages is not None and expected_pages <= 0:
        raise ValueError("manifest expected page total must be positive")

    input_bytes = 0
    for item, path in zip(raw, paths):
        actual_bytes = Path(path).stat().st_size
        if isinstance(item, dict) and "size_bytes" in item:
            expected_bytes = int(item["size_bytes"])
            if expected_bytes != actual_bytes:
                raise ValueError(
                    f"manifest size mismatch for {path}: "
                    f"{expected_bytes} != {actual_bytes}"
                )
        input_bytes += actual_bytes
    return DoclingManifest(
        paths=paths,
        pdf_ids=pdf_ids,
        input_bytes=input_bytes,
        expected_pages=expected_pages,
        manifest_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def run_paired(args: argparse.Namespace) -> dict[str, Any]:
    import ray  # pyright: ignore[reportMissingImports]

    if args.limit < 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("limit/warmup/repeats are invalid")
    if args.expected_tables is not None and args.expected_tables < 0:
        raise ValueError("expected_tables must be non-negative")
    if not 0 <= args.minimum_jaccard <= 1:
        raise ValueError("minimum_jaccard must be in [0, 1]")
    manifest = _read_manifest(args.manifest, args.limit)
    config = CoreMatrixConfig(
        device=args.device,
        ocr_device=args.ocr_device,
        num_threads=args.num_threads,
        stage_batch_size=args.stage_batch_size,
        table_core_batch_size=args.table_core_batch_size,
        parse_replicas=args.parse_replicas,
        layout_replicas=args.layout_replicas,
        ocr_replicas=args.ocr_replicas,
        table_replicas=args.table_replicas,
        reduce_replicas=args.reduce_replicas,
        parse_batch_wait_ms=args.batch_wait_ms,
        stage_batch_wait_ms=args.batch_wait_ms,
        layout_num_gpus=args.layout_num_gpus,
        table_num_gpus=args.table_num_gpus,
        ocr_batch_mode=args.ocr_batch_mode,
        ocr_recognition_batch_size=args.ocr_recognition_batch_size,
        table_batch_mode=args.table_batch_mode,
        table_batch_max_jobs=args.table_batch_max_jobs,
        actor_num_cpus=args.actor_num_cpus,
        max_pending_per_actor=1,
        microbatch_size=args.microbatch_size,
        max_inflight_arenas=args.max_active_microbatches,
    )
    v3_options, v36_options = _arm_options(
        len(manifest.paths),
        config,
        args.batch_scope,
    )

    started_ray_here = not ray.is_initialized()
    if started_ray_here:
        ray_options: dict[str, Any] = {
            "address": "local",
            "num_cpus": args.ray_num_cpus,
            "num_gpus": args.ray_num_gpus,
            "include_dashboard": False,
        }
        if args.object_store_gb is not None:
            ray_options["object_store_memory"] = int(
                args.object_store_gb * 1024**3
            )
        ray.init(**ray_options)
    try:
        all_trials = [
            _run_trial(
                manifest,
                v3_options,
                v36_options,
                microbatch_size=args.microbatch_size,
                max_active_microbatches=args.max_active_microbatches,
                order="v3_first" if index % 2 == 0 else "v36_first",
                minimum_jaccard=args.minimum_jaccard,
                expected_tables=args.expected_tables,
                gpu_monitor_dir=args.gpu_monitor_dir,
            )
            for index in range(args.warmup + args.repeats)
        ]
    finally:
        if started_ray_here:
            ray.shutdown()

    trials = all_trials[args.warmup :]
    v3_measured = [trial["arms"]["v3"]["measured_s"] for trial in trials]
    v36_measured = [trial["arms"]["v36"]["measured_s"] for trial in trials]
    v3_outer = [trial["arms"]["v3"]["outer_wall_s"] for trial in trials]
    v36_outer = [trial["arms"]["v36"]["outer_wall_s"] for trial in trials]
    payload = {
        "schema_version": 1,
        "workload": "docling_document_page_tablejob_page_document",
        "manifest": str(Path(args.manifest).resolve()),
        "pdfs": len(manifest.paths),
        "input_bytes": manifest.input_bytes,
        "input_contract": {
            "manifest_sha256": manifest.manifest_sha256,
            "ordered_pdf_ids": list(manifest.pdf_ids),
            "expected_pages": manifest.expected_pages,
            "expected_tables": args.expected_tables,
        },
        "batch_scope": args.batch_scope,
        "v3_options": v3_options,
        "v36_options": v36_options,
        "scheduler_contract": {
            "v3": "timed_wait_plus_shallow_outstanding",
            "v36": "immediate_work_conserving",
            "intentionally_unmapped_v3_options": sorted(_V3_ONLY_OPTIONS),
        },
        "minimum_jaccard": args.minimum_jaccard,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "trials": trials,
        "summary": {
            "primary_timing_scope": "startup_materialization_and_teardown_inclusive",
            "v3_median_outer_wall_s": statistics.median(v3_outer),
            "v36_median_outer_wall_s": statistics.median(v36_outer),
            "v36_speedup_outer_wall_median": statistics.median(
                old / new for old, new in zip(v3_outer, v36_outer)
            ),
            "engine_measured_scope_note": (
                "V3 measured excludes RunResult.get(); V3.6 measured includes "
                "materialization, so outer wall is the primary paired metric"
            ),
            "v3_median_measured_s": statistics.median(v3_measured),
            "v36_median_measured_s": statistics.median(v36_measured),
            "v36_engine_measured_ratio_median": statistics.median(
                old / new for old, new in zip(v3_measured, v36_measured)
            ),
            "outputs_pass": True,
            "timing_statistics": paired_timing_summary(
                v3_outer,
                v36_outer,
                (str(trial["order"]) for trial in trials),
            ),
        },
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--gpu-monitor-dir")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--minimum-jaccard", type=float, default=0.99)
    parser.add_argument("--expected-tables", type=int)
    parser.add_argument(
        "--batch-scope",
        choices=("elastic", "parent_bound"),
        default="elastic",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ocr-device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--stage-batch-size", type=int, default=16)
    parser.add_argument("--table-core-batch-size", type=int, default=4)
    parser.add_argument("--parse-replicas", type=int, default=16)
    parser.add_argument("--layout-replicas", type=int, default=2)
    parser.add_argument("--ocr-replicas", type=int, default=6)
    parser.add_argument("--table-replicas", type=int, default=2)
    parser.add_argument("--reduce-replicas", type=int, default=4)
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--layout-num-gpus", type=float, default=1.0)
    parser.add_argument("--table-num-gpus", type=float, default=1.0)
    parser.add_argument(
        "--ocr-batch-mode",
        choices=("reference", "recognition_shadow", "recognition_accelerated"),
        default="recognition_accelerated",
    )
    parser.add_argument("--ocr-recognition-batch-size", type=int, default=6)
    parser.add_argument(
        "--table-batch-mode",
        choices=("reference", "v1_batch", "v2_batch"),
        default="v1_batch",
    )
    parser.add_argument("--table-batch-max-jobs", type=int, default=16)
    parser.add_argument("--actor-num-cpus", type=float, default=1.0)
    parser.add_argument("--microbatch-size", type=int, default=24)
    parser.add_argument("--max-active-microbatches", type=int, default=4)
    parser.add_argument("--ray-num-cpus", type=int, default=128)
    parser.add_argument("--ray-num-gpus", type=float, default=4.0)
    parser.add_argument("--object-store-gb", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_paired(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["DoclingManifest", "build_parser", "run_paired"]
