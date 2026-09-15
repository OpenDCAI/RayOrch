from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from rayorch.benchmark.registry import available, get_plugin, register_plugin
from rayorch.benchmark.runtime import load_runtime_env


def test_root_and_benchmark_imports_do_not_load_optional_runtime_dependencies():
    code = """
import json, sys
import rayorch
root = sorted(name for name in ('ray', 'numpy', 'torch', 'vllm', 'flash_mineru') if name in sys.modules)
import rayorch.benchmark
benchmark = sorted(name for name in ('ray', 'numpy', 'torch', 'vllm', 'flash_mineru') if name in sys.modules)
print(json.dumps({'root': root, 'benchmark': benchmark}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"root": [], "benchmark": []}


def test_mineru_pipeline_import_still_defers_heavy_dependencies():
    code = """
import json, sys
import rayorch.benchmark.mineru.pipeline
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


def test_mineru_plugin_metadata_and_runtime_env_are_lightweight():
    assert available() == ("mineru",)
    plugin = get_plugin("mineru")
    assert plugin.name == "mineru"
    assert "vllm" not in sys.modules

    first = load_runtime_env(plugin)
    second = load_runtime_env(plugin)
    assert first == second
    assert first.value["pip"]["packages"]
    assert "numpy==2.2.6" in first.value["pip"]["packages"]
    assert "transformers>=4.57.3,<5.0.0" in first.value["pip"]["packages"]
    assert len(first.digest) == 64


def test_registry_supports_lazy_explicit_registration(tmp_path: Path, monkeypatch):
    package = tmp_path / "external_plugin.py"
    package.write_text(
        """
from rayorch.benchmark.plugin import BenchmarkPlugin
PLUGIN = BenchmarkPlugin(
    name="external",
    runner_module="external_runner",
    runtime_package="rayorch.benchmark.mineru",
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    register_plugin("external", "external_plugin")
    try:
        assert "external_plugin" not in sys.modules
        assert "external" in available()
        assert get_plugin("external").name == "external"
        assert "external_plugin" in sys.modules
    finally:
        from rayorch.benchmark import registry

        registry._PLUGIN_MODULES.pop("external")
        sys.modules.pop("external_plugin", None)


def test_registry_rejects_accidental_overwrite():
    with pytest.raises(ValueError, match="already registered"):
        register_plugin("mineru", "some.other.module")


def test_registry_rejects_metadata_name_mismatch(tmp_path: Path, monkeypatch):
    module = tmp_path / "wrong_name_plugin.py"
    module.write_text(
        """
from rayorch.benchmark.plugin import BenchmarkPlugin
PLUGIN = BenchmarkPlugin(
    name="declared-name",
    runner_module="external_runner",
    runtime_package="rayorch.benchmark.mineru",
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    register_plugin("registered-name", "wrong_name_plugin")
    try:
        with pytest.raises(ValueError, match="declared-name"):
            get_plugin("registered-name")
    finally:
        from rayorch.benchmark import registry

        registry._PLUGIN_MODULES.pop("registered-name")
        sys.modules.pop("wrong_name_plugin", None)


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


def test_runtime_env_digest_tracks_module_content_not_temporary_path(
    tmp_path: Path,
):
    plugin = get_plugin("mineru")
    left = tmp_path / "left" / "rayorch.whl"
    right = tmp_path / "right" / "renamed.whl"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_bytes(b"same wheel")
    right.write_bytes(b"same wheel")

    first = load_runtime_env(plugin, py_modules=[left])
    second = load_runtime_env(plugin, py_modules=[right])
    assert first.digest == second.digest

    right.write_bytes(b"different wheel")
    changed = load_runtime_env(plugin, py_modules=[right])
    assert changed.digest != first.digest


def test_runtime_env_digest_ignores_zip_build_timestamps(tmp_path: Path):
    plugin = get_plugin("mineru")
    wheels = []
    for name, timestamp in (
        ("first.whl", (2025, 1, 1, 0, 0, 0)),
        ("second.whl", (2026, 2, 2, 0, 0, 0)),
    ):
        wheel = tmp_path / name
        info = zipfile.ZipInfo("package/__init__.py", timestamp)
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(info, b"value = 1\n")
        wheels.append(wheel)

    first = load_runtime_env(plugin, py_modules=[wheels[0]])
    second = load_runtime_env(plugin, py_modules=[wheels[1]])
    assert first.digest == second.digest
