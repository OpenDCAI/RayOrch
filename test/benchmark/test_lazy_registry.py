from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rayorch.benchmark.registry import (
    available,
    class_path,
    load,
    register,
    runtime_env,
)


def test_root_and_benchmark_imports_do_not_load_optional_runtime_dependencies():
    code = """
import json, sys
import rayorch
root = sorted(name for name in ('ray', 'numpy', 'torch', 'vllm', 'sglang', 'flash_mineru') if name in sys.modules)
import rayorch.benchmark
benchmark = sorted(name for name in ('ray', 'numpy', 'torch', 'vllm', 'sglang', 'flash_mineru') if name in sys.modules)
print(json.dumps({'root': root, 'benchmark': benchmark}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"root": [], "benchmark": []}


def test_framework_and_builtin_workloads_are_separate_packages():
    from rayorch import benchmark, benchmarks

    assert benchmark.__name__ == "rayorch.benchmark"
    assert benchmarks.__name__ == "rayorch.benchmarks"
    assert not any(
        path.suffix == ".py"
        for path in (Path(benchmark.__file__).parent / "mineru").glob("*")
    )
    mineru = Path(benchmarks.__file__).parent / "mineru"
    assert {
        path.name
        for path in mineru.iterdir()
        if path.name != "__pycache__"
    } == {
        "__init__.py",
        "README.md",
        "benchmark.py",
        "pipeline.py",
        "env.json",
        "udfs.py",
    }

    for name in available():
        package_name = class_path(name).split(":", 1)[0].split(".")[-2]
        package = Path(benchmarks.__file__).parent / package_name
        assert {
            "__init__.py",
            "README.md",
            "benchmark.py",
            "pipeline.py",
            "env.json",
            "udfs.py",
        } <= {
            path.name
            for path in package.iterdir()
            if path.name != "__pycache__"
        }


def test_mineru_pipeline_import_still_defers_heavy_dependencies():
    code = """
import json, sys
import rayorch.benchmarks.mineru.pipeline
watched = (
    'ray', 'numpy', 'torch', 'vllm', 'flash_mineru',
    'mineru_vl_utils', 'PIL', 'pypdf', 'pypdfium2',
)
print(json.dumps(sorted(name for name in watched if name in sys.modules)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []

    code = """
import json, sys
import rayorch.benchmarks.mineru_scale.pipeline
watched = (
    'ray', 'numpy', 'torch', 'vllm', 'flash_mineru', 'pyarrow',
    'mineru_vl_utils', 'PIL', 'pypdf', 'pypdfium2',
)
print(json.dumps(sorted(name for name in watched if name in sys.modules)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_builtin_spec_and_runtime_env_are_lightweight():
    assert available() == (
        "document_topology",
        "dual_vllm",
        "mineru",
        "mineru_scale",
        "sglang_vllm",
        "video_caption_topology",
        "video_multimodal_topology",
        "yolo_sam",
    )
    assert class_path("mineru").endswith(":MinerUBench")
    assert class_path("mineru_scale").endswith(":MinerUScaleBench")
    assert class_path("document_topology").endswith(":DocumentTopologyBench")
    assert "vllm" not in sys.modules

    environment = runtime_env("mineru")
    assert "numpy==2.2.6" in environment["pip"]["packages"]
    assert "transformers>=4.57.3,<5.0.0" in environment["pip"]["packages"]
    scale_environment = runtime_env("mineru_scale")
    assert "mineru-vl-utils==1.2.1" in scale_environment["pip"]["packages"]
    assert "vllm==0.11.0" in scale_environment["pip"]["packages"]
    assert runtime_env("video_caption_topology") == {}


def test_benchmark_attribute_uses_registry_as_its_single_source():
    import rayorch.benchmark as benchmark

    assert benchmark.MinerUBench is load("mineru")
    assert benchmark.MinerUScaleBench is load("mineru_scale")
    assert benchmark.SglangVllmBench is load("sglang_vllm")
    assert benchmark.DocumentTopologyBench is load("document_topology")
    assert benchmark.DualVllmBench is load("dual_vllm")
    assert benchmark.VideoCaptionTopologyBench is load("video_caption_topology")
    assert benchmark.VideoMultimodalTopologyBench is load(
        "video_multimodal_topology"
    )
    assert benchmark.YoloSamBench is load("yolo_sam")
    with pytest.raises(AttributeError):
        getattr(benchmark, "mineru")
    assert "vllm" not in sys.modules


def test_registry_accepts_one_explicit_record(tmp_path: Path, monkeypatch):
    module = tmp_path / "external_benchmark.py"
    module.write_text("class ExternalBench: pass\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    register(
        "external",
        public_name="ExternalBench",
        benchmark_class="external_benchmark:ExternalBench",
        runtime_package="rayorch.benchmarks.mineru",
    )
    try:
        assert "external_benchmark" not in sys.modules
        assert class_path("external") == "external_benchmark:ExternalBench"
        assert load("external").__name__ == "ExternalBench"
    finally:
        from rayorch.benchmark import registry

        registry._SPECS.pop("external")
        sys.modules.pop("external_benchmark", None)


def test_registry_rejects_accidental_overwrite():
    with pytest.raises(ValueError, match="already registered"):
        register(
            "mineru",
            public_name="MinerUBench",
            benchmark_class="unused:MinerUBench",
            runtime_package="rayorch.benchmarks.mineru",
        )
