from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rayorch.benchmark.registry import available, get_plugin
from rayorch.benchmark.runtime import load_runtime_env


def test_root_and_benchmark_imports_do_not_load_optional_runtime_dependencies():
    code = """
import json, sys
import rayorch
root = sorted(name for name in ('ray', 'vllm', 'daft', 'flash_mineru') if name in sys.modules)
import rayorch.benchmark
benchmark = sorted(name for name in ('ray', 'vllm', 'daft', 'flash_mineru') if name in sys.modules)
print(json.dumps({'root': root, 'benchmark': benchmark}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"root": [], "benchmark": []}


def test_mineru_plugin_metadata_and_runtime_env_are_lightweight():
    assert available() == ("mineru",)
    plugin = get_plugin("mineru")
    assert plugin.name == "mineru"
    assert "vllm" not in sys.modules

    first = load_runtime_env(plugin)
    second = load_runtime_env(plugin)
    assert first == second
    assert first.value["pip"]["packages"]
    assert len(first.digest) == 64


def test_runtime_env_adds_built_wheels_and_explicit_pip(tmp_path: Path):
    plugin = get_plugin("mineru")
    wheel = tmp_path / "rayorch-0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    resolved = load_runtime_env(
        plugin,
        py_modules=[wheel],
        extra_pip=["daft==0.7.21"],
    )
    assert resolved.value["py_modules"] == [str(wheel.resolve())]
    assert resolved.value["pip"]["packages"][-1] == "daft==0.7.21"
