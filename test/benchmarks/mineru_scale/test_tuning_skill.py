"""Behavioral checks for the embedded MinerU tuning Skill helper."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[3]
SCRIPT = (
    ROOT
    / "rayorch"
    / "benchmarks"
    / "mineru_scale"
    / "skills"
    / "mineru-scale-tuning"
    / "scripts"
    / "recommend.py"
)


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_shared_h20_profile_reproduces_64_gpu_reference_shape():
    completed = _run(
        "--gpus",
        "64",
        "--gpu-memory-gb",
        "96",
        "--cluster-cpus",
        "512",
        "--profile",
        "shared-h20",
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    parameters = result["parameters"]
    assert parameters["render_replicas"] == 256
    assert parameters["ocr_replicas"] == 128
    assert parameters["assemble_replicas"] == 64
    assert parameters["batch_size"] == 64
    assert parameters["max_active_input_batches"] == 24
    assert result["resources"]["reserved_gpus"] == 64
    assert result["resources"]["actor_cpus"] == 449


def test_recommender_caps_cpu_stages_and_rejects_small_shared_gpu():
    completed = _run(
        "--gpus",
        "64",
        "--gpu-memory-gb",
        "96",
        "--cluster-cpus",
        "256",
        "--profile",
        "shared-h20",
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["resources"]["actor_cpus"] <= 252
    assert result["parameters"]["ocr_replicas"] == 128
    assert any("CPU budget" in item for item in result["warnings"])

    rejected = _run(
        "--gpus",
        "8",
        "--gpu-memory-gb",
        "48",
        "--profile",
        "shared-h20",
    )
    assert rejected.returncode != 0
    assert "high-memory GPU" in rejected.stderr


def test_cli_format_emits_benchmark_flags():
    completed = _run(
        "--gpus",
        "1",
        "--profile",
        "smoke",
        "--format",
        "cli",
    )

    assert completed.returncode == 0, completed.stderr
    assert "--ocr-replicas 1" in completed.stdout
    assert "--input-limit 2" in completed.stdout
