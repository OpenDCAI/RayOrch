"""Lazy Benchmark API; importing it never imports workload dependencies."""

from __future__ import annotations

from typing import Any

from .execution import benchmark_config, run_benchmark
from .report import BenchmarkReport
from .submission import (
    BenchmarkRun,
    LocalSource,
    submit_benchmark,
)
from .registry import (
    available,
    load,
    public_names,
    register,
)

__all__ = [
    "BenchmarkReport",
    "BenchmarkRun",
    "DocumentTopologyBench",
    "DualVllmBench",
    "LocalSource",
    "MinerUBench",
    "SglangVllmBench",
    "VideoCaptionTopologyBench",
    "VideoMultimodalTopologyBench",
    "YoloSamBench",
    "available",
    "benchmark_config",
    "load",
    "register",
    "run_benchmark",
    "submit_benchmark",
]


def __getattr__(name: str) -> Any:
    benchmark_name = public_names().get(name)
    if benchmark_name is None:
        raise AttributeError(name)
    benchmark = load(benchmark_name)
    globals()[name] = benchmark
    return benchmark


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(public_names()))
