"""Small import-free registry for Benchmark classes and dependency manifests."""

from __future__ import annotations

import json
from importlib import import_module
from importlib.resources import files
from typing import Any


_SPECS = {
    "document_topology": (
        "DocumentTopologyBench",
        "rayorch.benchmarks.document_topology.benchmark:DocumentTopologyBench",
        "rayorch.benchmarks.document_topology",
        "env.json",
    ),
    "dual_vllm": (
        "DualVllmBench",
        "rayorch.benchmarks.dual_vllm.benchmark:DualVllmBench",
        "rayorch.benchmarks.dual_vllm",
        "env.json",
    ),
    "mineru": (
        "MinerUBench",
        "rayorch.benchmarks.mineru.benchmark:MinerUBench",
        "rayorch.benchmarks.mineru",
        "env.json",
    ),
    "panda70m": (
        "Panda70MBench",
        "rayorch.benchmarks.panda70m.benchmark:Panda70MBench",
        "rayorch.benchmarks.panda70m",
        "env.json",
    ),
    "sglang_vllm": (
        "SglangVllmBench",
        "rayorch.benchmarks.sglang_vllm.benchmark:SglangVllmBench",
        "rayorch.benchmarks.sglang_vllm",
        "env.json",
    ),
    "video_caption_topology": (
        "VideoCaptionTopologyBench",
        "rayorch.benchmarks.video_caption.benchmark:VideoCaptionTopologyBench",
        "rayorch.benchmarks.video_caption",
        "env.json",
    ),
    "video_multimodal_topology": (
        "VideoMultimodalTopologyBench",
        "rayorch.benchmarks.video_multimodal.benchmark:VideoMultimodalTopologyBench",
        "rayorch.benchmarks.video_multimodal",
        "env.json",
    ),
    "yolo_sam": (
        "YoloSamBench",
        "rayorch.benchmarks.yolo_sam.benchmark:YoloSamBench",
        "rayorch.benchmarks.yolo_sam",
        "env.json",
    ),
}


def available() -> tuple[str, ...]:
    return tuple(sorted(_SPECS))


def register(
    name: str,
    *,
    public_name: str,
    benchmark_class: str,
    runtime_package: str,
    runtime_resource: str = "env.json",
    exist_ok: bool = False,
) -> None:
    """Register one Benchmark without importing its implementation."""

    if not all((name, public_name, benchmark_class, runtime_package)):
        raise ValueError("Benchmark registration fields must be non-empty")
    if name in _SPECS and not exist_ok:
        raise ValueError(f"benchmark {name!r} is already registered")
    _SPECS[name] = (
        public_name,
        benchmark_class,
        runtime_package,
        runtime_resource,
    )


def _spec(name: str) -> tuple[str, str, str, str]:
    try:
        return _SPECS[name]
    except KeyError as exc:
        choices = ", ".join(available())
        raise KeyError(f"unknown benchmark {name!r}; available: {choices}") from exc


def load(name: str):
    """Load one registered Benchmark class without importing its dependencies."""

    _, class_path, _, _ = _spec(name)
    module_name, separator, attribute = class_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("benchmark_class must use the 'module:attribute' form")
    benchmark = getattr(import_module(module_name), attribute)
    if not isinstance(benchmark, type):
        raise TypeError(f"{class_path} is not a class")
    return benchmark


def class_path(name: str) -> str:
    return _spec(name)[1]


def runtime_env(name: str) -> dict[str, Any]:
    _, _, package, resource_name = _spec(name)
    resource = files(package).joinpath(resource_name)
    value = json.loads(resource.read_text(encoding="utf-8"))
    if value.pop("schema_version", None) != 1:
        raise ValueError(f"unsupported runtime_env schema in {resource}")
    return value


def public_names() -> dict[str, str]:
    """Return public class name -> registry name without loading Benchmark classes."""

    names: dict[str, str] = {}
    for name in available():
        public_name = _spec(name)[0]
        if public_name in names:
            raise ValueError(f"duplicate benchmark public name {public_name!r}")
        names[public_name] = name
    return names


__all__ = [
    "available",
    "load",
    "register",
]
