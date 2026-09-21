#!/usr/bin/env python3
"""Generate a reviewable MinerUScaleBench starting configuration."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any


def _active_batches(gpus: int) -> int:
    if gpus <= 4:
        return 3
    if gpus <= 16:
        return 8
    if gpus <= 32:
        return 16
    return 24


def recommend(
    *,
    gpus: int,
    profile: str,
    cluster_cpus: int | None = None,
    reserve_cpus: int = 4,
    gpu_memory_gb: float | None = None,
) -> dict[str, Any]:
    """Return parameters, resource totals, and review warnings."""

    if gpus <= 0:
        raise ValueError("gpus must be positive")
    if cluster_cpus is not None and cluster_cpus <= 0:
        raise ValueError("cluster_cpus must be positive")
    if reserve_cpus < 0:
        raise ValueError("reserve_cpus must be non-negative")
    if cluster_cpus is not None and reserve_cpus >= cluster_cpus:
        raise ValueError("reserve_cpus must be smaller than cluster_cpus")

    warnings: list[str] = []
    if profile == "smoke":
        parameters: dict[str, Any] = {
            "render_replicas": 2,
            "ocr_replicas": 1,
            "assemble_replicas": 1,
            "batch_size": 8,
            "input_batch_size": 1,
            "max_active_input_batches": 1,
            "gpu_memory_utilization": 0.70,
            "gpus_per_ocr_actor": 1.0,
            "render_dpi": 200,
            "input_limit": 2,
        }
        if gpus > 1:
            warnings.append("smoke profile intentionally reserves only one GPU")
    elif profile == "conservative":
        parameters = {
            "render_replicas": 4 * gpus,
            "ocr_replicas": gpus,
            "assemble_replicas": gpus,
            "batch_size": 64,
            "input_batch_size": 24,
            "max_active_input_batches": _active_batches(gpus),
            "gpu_memory_utilization": 0.80,
            "gpus_per_ocr_actor": 1.0,
            "render_dpi": 200,
        }
        warnings.append(
            "0.80 vLLM utilization was validated on 96-GB H20; smoke-test "
            "other GPU models before a full run"
        )
    elif profile == "shared-h20":
        if gpu_memory_gb is not None and gpu_memory_gb < 80:
            raise ValueError(
                "shared-h20 requires an H20-class high-memory GPU; "
                "use conservative for this memory size"
            )
        parameters = {
            "render_replicas": 4 * gpus,
            "ocr_replicas": 2 * gpus,
            "assemble_replicas": gpus,
            "batch_size": 64,
            "input_batch_size": 24,
            "max_active_input_batches": _active_batches(gpus),
            "gpu_memory_utilization": 0.32,
            "gpus_per_ocr_actor": 0.5,
            "render_dpi": 200,
        }
        warnings.append(
            "shared-h20 assumes two vLLM actors per 96-GB H20; validate "
            "model startup and memory before processing the corpus"
        )
    else:
        raise ValueError(f"unknown profile: {profile}")

    requested_render = int(parameters["render_replicas"])
    requested_assemble = int(parameters["assemble_replicas"])
    requested_actor_cpus = (
        requested_render
        + int(parameters["ocr_replicas"])
        + requested_assemble
        + 1
    )

    if cluster_cpus is not None:
        usable = cluster_cpus - reserve_cpus
        fixed = int(parameters["ocr_replicas"]) + 1
        stage_budget = usable - fixed
        if stage_budget < 2:
            raise ValueError(
                "CPU budget cannot schedule the OCR pool plus render, "
                "assemble, and metadata actors"
            )
        requested_stages = requested_render + requested_assemble
        if requested_stages > stage_budget:
            render_share = requested_render / requested_stages
            render = max(1, math.floor(stage_budget * render_share))
            assemble = max(1, stage_budget - render)
            while render + assemble > stage_budget:
                if render >= assemble and render > 1:
                    render -= 1
                elif assemble > 1:
                    assemble -= 1
                else:
                    break
            parameters["render_replicas"] = render
            parameters["assemble_replicas"] = assemble
            warnings.append(
                "render and assemble pools were reduced proportionally to "
                "fit the supplied CPU budget"
            )

    actor_cpus = (
        int(parameters["render_replicas"])
        + int(parameters["ocr_replicas"])
        + int(parameters["assemble_replicas"])
        + 1
    )
    reserved_gpus = (
        int(parameters["ocr_replicas"])
        * float(parameters["gpus_per_ocr_actor"])
    )
    if reserved_gpus > gpus:
        raise AssertionError("generated OCR pool exceeds physical GPUs")

    return {
        "profile": profile,
        "parameters": parameters,
        "resources": {
            "physical_gpus": gpus,
            "reserved_gpus": reserved_gpus,
            "actor_cpus": actor_cpus,
            "uncapped_actor_cpus": requested_actor_cpus,
            "cluster_cpus": cluster_cpus,
            "reserved_system_cpus": reserve_cpus if cluster_cpus else None,
        },
        "warnings": warnings,
        "next_sweep": {
            "batch_size": [32, 64, 96, 128],
            "max_active_input_batches": [3, 8, 16, 24],
        },
    }


def _cli_fragment(parameters: dict[str, Any]) -> str:
    names = (
        "render_replicas",
        "ocr_replicas",
        "assemble_replicas",
        "batch_size",
        "input_batch_size",
        "max_active_input_batches",
        "gpu_memory_utilization",
        "gpus_per_ocr_actor",
        "render_dpi",
        "input_limit",
    )
    lines = []
    for name in names:
        if name not in parameters:
            continue
        flag = "--" + name.replace("_", "-")
        lines.append(f"{flag} {parameters[name]}")
    return " ".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recommend a starting MinerU scale configuration.",
    )
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke", "conservative", "shared-h20"),
        default="conservative",
    )
    parser.add_argument("--cluster-cpus", type=int)
    parser.add_argument("--reserve-cpus", type=int, default=4)
    parser.add_argument("--gpu-memory-gb", type=float)
    parser.add_argument("--format", choices=("json", "cli"), default="json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = recommend(
            gpus=args.gpus,
            profile=args.profile,
            cluster_cpus=args.cluster_cpus,
            reserve_cpus=args.reserve_cpus,
            gpu_memory_gb=args.gpu_memory_gb,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.format == "cli":
        print(_cli_fragment(result["parameters"]))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
