"""Thin Ray Job wrapper around the repository-owned ``mineru_taiji`` runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any


PROBE_MARKER = "RAYORCH_MINERU_PROBE_RESULT "
FULL_MARKER = "RAYORCH_MINERU_FULL_RESULT "
REQUIRED_MODULES = (
    "rayorch.experimental.multigrain_v3_6.benchmark.mineru_taiji",
    "flash_mineru.mineru_core.data.data_reader_writer",
    "flash_mineru.mineru_core.engine.model_output_to_middle_json",
    "flash_mineru.mineru_core.engine.vlm_middle_json_mkcontent",
    "flash_mineru.mineru_core.utils.pdf_image_tools",
    "mineru_vl_utils",
    "vllm",
    "fitz",
    "torch",
    "psutil",
    "pynvml",
    "pyarrow",
)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


def _required_json_list(name: str, item_type: type) -> list[Any]:
    """Parse a required JSON array from the frozen Ray Job environment."""

    value = json.loads(_required_env(name))
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{name} must contain a non-empty JSON array")
    if any(not isinstance(item, item_type) for item in value):
        raise RuntimeError(f"{name} contains an item with the wrong type")
    return value


def _required_json_object(name: str) -> dict[str, Any]:
    value = json.loads(_required_env(name))
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain a JSON object")
    return value


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError("run id contains unsupported characters")
    return value


def _node_dependencies(
    expected_node_id: str,
    flash_repo: str,
    expect_h20s: int,
) -> dict[str, Any]:
    import ray

    from rayorch.experimental.multigrain_v3_6.benchmark.mineru_taiji import (
        _assert_current_node,
    )

    node_id = _assert_current_node(expected_node_id)
    if not Path(flash_repo).is_dir():
        raise RuntimeError(f"Flash-MinerU repository is unavailable: {flash_repo}")
    if flash_repo not in sys.path:
        sys.path.insert(0, flash_repo)
    versions: dict[str, str] = {}
    for name in REQUIRED_MODULES:
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "available"))
    gpu_names: list[str] = []
    if expect_h20s:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {completed.stderr[-2000:]}")
        gpu_names = [
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        ]
        if len(gpu_names) != expect_h20s:
            raise RuntimeError(
                f"node {node_id} sees {len(gpu_names)} GPUs, expected {expect_h20s}"
            )
        if any("H20" not in name.upper() for name in gpu_names):
            raise RuntimeError(f"node {node_id} contains a non-H20 GPU: {gpu_names}")
    return {
        "node_id": node_id,
        "node_ip": ray.util.get_node_ip_address(),
        "dependency_versions": versions,
        "gpu_names": gpu_names,
    }


def _activate_gpu_node(expect_h20s: int) -> dict[str, Any]:
    """Occupy one complete GPU host to trigger lazy TaiJi worker provisioning."""

    import ray

    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or len(names) != expect_h20s:
        raise RuntimeError(
            "GPU activation task did not land on a complete worker: "
            f"returncode={completed.returncode}, GPUs={names}"
        )
    if any("H20" not in name.upper() for name in names):
        raise RuntimeError(f"GPU activation task found a non-H20 host: {names}")
    return {
        "node_id": str(ray.get_runtime_context().get_node_id()),
        "node_ip": ray.util.get_node_ip_address(),
        "gpu_names": names,
    }


def _hdfs_manifest(uri: str, *, suffix: str | None = None) -> dict[str, Any]:
    from rayorch.experimental.multigrain_v3_6.benchmark.mineru_taiji import (
        _remote_file_infos,
    )

    _, _, _, infos = _remote_file_infos(uri)
    selected = tuple(
        info
        for info in infos
        if suffix is None or PurePosixPath(info.path).suffix.lower() == suffix
    )
    if not selected:
        raise FileNotFoundError(f"HDFS URI has no matching files: {uri}")
    return {
        "uri": uri,
        "files": len(selected),
        "bytes": sum(int(info.size) for info in selected),
    }


def _probe() -> dict[str, Any]:
    import ray

    from rayorch.experimental.multigrain_v3_6.benchmark.mineru_taiji import (
        EXPECTED_GPU_WORKERS,
        EXPECTED_MODEL_BYTES,
        EXPECTED_MODEL_FILES,
        GPUS_PER_WORKER,
        _node_id_text,
        _run_pinned_tasks,
        _wait_for_gpu_topology,
    )

    if not ray.is_initialized():
        ray.init(address="auto")
    flash_repo = _required_env("RAYORCH_FLASH_MINERU_REPO")
    pdf_uris = _required_json_list("RAYORCH_HDFS_INPUT_URIS", str)
    expected_pdf_files = _required_json_list(
        "RAYORCH_EXPECTED_PDF_FILES", int
    )
    expected_pdf_bytes = _required_json_list(
        "RAYORCH_EXPECTED_PDF_BYTES", int
    )
    if not (
        len(pdf_uris) == len(expected_pdf_files) == len(expected_pdf_bytes)
    ):
        raise RuntimeError("PDF input URI/file/byte contracts have unequal lengths")
    model_uri = _required_env("RAYORCH_HDFS_MODEL_URI")
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    activation_group = placement_group(
        [{"GPU": GPUS_PER_WORKER} for _ in range(EXPECTED_GPU_WORKERS)],
        strategy="STRICT_SPREAD",
    )
    try:
        ray.get(activation_group.ready(), timeout=43_200)
        activation_task = ray.remote(num_cpus=0, num_gpus=GPUS_PER_WORKER)(
            _activate_gpu_node
        )
        activation_results = ray.get(
            [
                activation_task.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=activation_group,
                        placement_group_bundle_index=index,
                    )
                ).remote(GPUS_PER_WORKER)
                for index in range(EXPECTED_GPU_WORKERS)
            ],
            timeout=600,
        )
    finally:
        remove_placement_group(activation_group)
    activation_nodes = {item["node_id"] for item in activation_results}
    if len(activation_nodes) != EXPECTED_GPU_WORKERS:
        raise RuntimeError(
            "GPU activation tasks did not occupy distinct complete workers: "
            f"{activation_results}"
        )
    nodes, gpu_nodes = _wait_for_gpu_topology(
        ray,
        timeout_s=1800,
        poll_s=5,
    )
    resource_records = {
        _node_id_text(record.get("NodeID", "")): record.get("Resources") or {}
        for record in ray.nodes()
        if record.get("Alive")
    }
    accelerator_resources = {
        node.node_id: float(
            resource_records.get(node.node_id, {}).get("accelerator_type:H20", 0)
        )
        for node in gpu_nodes
    }
    if any(value <= 0 for value in accelerator_resources.values()):
        raise RuntimeError(
            "GPU workers do not advertise accelerator_type:H20: "
            f"{accelerator_resources}"
        )
    gpu_ids = {node.node_id for node in gpu_nodes}
    dependency_results = _run_pinned_tasks(
        ray,
        _node_dependencies,
        [
            (
                node,
                (flash_repo, GPUS_PER_WORKER if node.node_id in gpu_ids else 0),
            )
            for node in nodes
        ],
    )
    pdf_manifests = [
        _hdfs_manifest(uri, suffix=".pdf") for uri in pdf_uris
    ]
    model_manifest = _hdfs_manifest(model_uri)
    expected_manifests = [
        {"uri": uri, "files": files, "bytes": size}
        for uri, files, size in zip(
            pdf_uris,
            expected_pdf_files,
            expected_pdf_bytes,
            strict=True,
        )
    ]
    if pdf_manifests != expected_manifests:
        raise RuntimeError(f"PDF HDFS manifests mismatch: {pdf_manifests}")
    if model_manifest != {
        "uri": model_uri,
        "files": EXPECTED_MODEL_FILES,
        "bytes": EXPECTED_MODEL_BYTES,
    }:
        raise RuntimeError(f"model HDFS manifest mismatch: {model_manifest}")
    return {
        "status": "ok",
        "read_only": True,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source_digest": _required_env("RAYORCH_SOURCE_DIGEST"),
        "pdf_manifests": pdf_manifests,
        "model_manifest": model_manifest,
        "cluster_nodes": [asdict(node) for node in nodes],
        "gpu_workers": [asdict(node) for node in gpu_nodes],
        "accelerator_resources": accelerator_resources,
        "expected_gpu_workers": EXPECTED_GPU_WORKERS,
        "expected_gpus_per_worker": GPUS_PER_WORKER,
        "activation_results": activation_results,
        "dependency_nodes_checked": len(dependency_results),
        "dependency_results": dependency_results,
        "ray_version": getattr(ray, "__version__", None),
        "ray_commit": getattr(ray, "__commit__", None),
    }


def _full() -> dict[str, Any]:
    from rayorch.experimental.multigrain_v3_6.benchmark.mineru_taiji import (
        build_parser,
        run_taiji,
    )

    run_id = _safe_run_id(_required_env("RAYORCH_RUN_ID"))
    pdf_uris = _required_json_list("RAYORCH_HDFS_INPUT_URIS", str)
    expected_pdf_files = _required_json_list(
        "RAYORCH_EXPECTED_PDF_FILES", int
    )
    expected_pdf_bytes = _required_json_list(
        "RAYORCH_EXPECTED_PDF_BYTES", int
    )
    pdf_limits = _required_json_list("RAYORCH_PDF_LIMITS", int)
    hyperparameters = _required_json_object("RAYORCH_HYPERPARAMETERS")
    if not (
        len(pdf_uris)
        == len(expected_pdf_files)
        == len(expected_pdf_bytes)
        == len(pdf_limits)
    ):
        raise RuntimeError("PDF input and sample-limit contracts have unequal lengths")
    cli = []
    for uri, files, size, limit in zip(
        pdf_uris,
        expected_pdf_files,
        expected_pdf_bytes,
        pdf_limits,
        strict=True,
    ):
        cli.extend(
            [
                "--hdfs-pdf-uri",
                uri,
                "--expected-pdf-files",
                str(files),
                "--expected-pdf-bytes",
                str(size),
                "--pdf-limit",
                str(limit),
            ]
        )
    output_kind = _required_env("RAYORCH_OUTPUT_KIND")
    output_base = _required_env("RAYORCH_OUTPUT_BASE")
    cli.extend(
        [
            "--hdfs-model-uri",
            _required_env("RAYORCH_HDFS_MODEL_URI"),
            "--run-id",
            run_id,
            "--local-root",
            f"/tmp/rayorch-mineru-v3-6-taiji-{run_id}",
            "--local-model-dir",
            "/tmp/rayorch-mineru-v3-6-taiji-model-cache",
            "--flash-repo",
            _required_env("RAYORCH_FLASH_MINERU_REPO"),
            "--microbatch-size",
            str(hyperparameters["microbatch_size"]),
            "--max-active-microbatches",
            str(hyperparameters["max_active_microbatches"]),
            "--batch-size",
            str(hyperparameters["batch_size"]),
            "--gpu-memory-utilization",
            str(hyperparameters["gpu_memory_utilization"]),
            "--ocr-replicas",
            str(hyperparameters["ocr_replicas"]),
            "--gpus-per-ocr-actor",
            str(hyperparameters["gpus_per_ocr_actor"]),
            "--render-replicas",
            str(hyperparameters["render_replicas"]),
            "--reduce-replicas",
            str(hyperparameters["reduce_replicas"]),
            "--spool-upload-workers",
            str(hyperparameters["spool_upload_workers"]),
            "--spool-drain-timeout-s",
            str(hyperparameters["spool_drain_timeout_s"]),
        ]
    )
    if output_kind == "hdfs":
        cli.extend(["--hdfs-output-uri", output_base])
    elif output_kind == "ceph":
        cli.extend(
            [
                "--ceph-output-dir",
                output_base,
                "--ceph-token-file",
                str(Path(__file__).with_name("taijiPATToken.runtime")),
                "--ceph-app-group",
                _required_env("RAYORCH_CEPH_APP_GROUP"),
                "--ceph-location",
                _required_env("RAYORCH_CEPH_LOCATION"),
            ]
        )
    else:
        raise RuntimeError(f"unsupported output kind: {output_kind}")
    if _required_env("RAYORCH_BENCHMARK_ONLY") == "1":
        cli.append("--benchmark-only")
    args = build_parser().parse_args(cli)
    engine = os.environ.get("RAYORCH_ENGINE", "rayorch").strip()
    args.profile_system = engine
    args.cold_e2e_start_epoch_s = __import__("time").time()
    args.cold_e2e_start_monotonic_s = __import__("time").perf_counter()
    if engine == "rayorch":
        result = run_taiji(args)
    elif engine == "raydata":
        from rayorch.experimental.multigrain_v3.benchmark.mineru_ray_data import (
            run_benchmark as run_ray_data_benchmark,
        )

        result = run_taiji(
            args,
            benchmark_runner=run_ray_data_benchmark,
        )
    else:
        raise RuntimeError(f"unsupported execution engine: {engine}")
    if result.get("status") != "completed":
        raise RuntimeError(f"mineru_taiji returned unexpected status: {result}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("probe", "full"))
    return parser


def main(argv: list[str] | None = None) -> int:
    phase = build_parser().parse_args(argv).phase
    marker = PROBE_MARKER if phase == "probe" else FULL_MARKER
    result = _probe() if phase == "probe" else _full()
    print(marker + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FULL_MARKER", "PROBE_MARKER", "build_parser", "main"]
