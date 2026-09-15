"""Lightweight registration metadata for the MinerU UDF group."""

from rayorch.benchmark.plugin import BenchmarkPlugin


PLUGIN = BenchmarkPlugin(
    name="mineru",
    runner_module="rayorch.benchmark.mineru.runner",
    runtime_package="rayorch.benchmark.mineru",
)

__all__ = ["PLUGIN"]
