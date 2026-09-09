"""Lightweight registration metadata for the MinerU UDF group."""

from rayorch.benchmark.plugin import BenchmarkPlugin


PLUGIN = BenchmarkPlugin(
    name="mineru",
    runner_module="rayorch.benchmark.mineru.pipeline",
    runtime_package="rayorch.benchmark.mineru",
    required_modules=(
        "PIL",
        "flash_mineru",
        "mineru_vl_utils",
        "pynvml",
        "pypdf",
        "pypdfium2",
        "ray",
        "torch",
        "vllm",
    ),
)

__all__ = ["PLUGIN"]
