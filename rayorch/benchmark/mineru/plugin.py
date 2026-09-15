"""Lightweight registration metadata for the MinerU UDF group."""

from rayorch.benchmark.plugin import BenchmarkPlugin


PLUGIN = BenchmarkPlugin(
    name="mineru",
    runner_module="rayorch.benchmark.mineru.pipeline",
    runtime_package="rayorch.benchmark.mineru",
    required_modules=(
        "PIL",
        "bs4",
        "cv2",
        "flash_mineru",
        "loguru",
        "magika",
        "mineru_vl_utils",
        "numpy",
        "psutil",
        "pynvml",
        "pypdf",
        "pypdfium2",
        "ray",
        "six",
        "torch",
        "transformers",
        "vllm",
    ),
)

__all__ = ["PLUGIN"]
