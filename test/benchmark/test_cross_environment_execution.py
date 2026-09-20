"""Opt-in integration coverage for per-stage Conda runtime environments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from rayorch import run
from test.benchmark.cross_environment_fixture import (
    BackendEnvironmentPipeline,
    CrossEnvironmentPipeline,
)


def test_stage_executes_in_configured_conda_environment():
    environment = os.environ.get("RAYORCH_TEST_CONDA_ENV")
    if not environment:
        pytest.skip("set RAYORCH_TEST_CONDA_ENV to run the cross-Conda test")

    result = run(
        CrossEnvironmentPipeline(environment),
        ["left", "right"],
        ray_init_kwargs={"include_dashboard": False, "num_cpus": 2},
    )
    outputs = list(result.outputs)

    assert [item["value"] for item in outputs] == ["left", "right"]
    assert all(
        Path(item["executable"]).resolve() != Path(sys.executable).resolve()
        for item in outputs
    )
    assert all(
        item["conda_default_env"] in {environment, Path(environment).name}
        for item in outputs
    )

    import ray

    assert all(item["ray"] == ray.__version__ for item in outputs)


def test_sglang_and_vllm_import_in_separate_conda_environments():
    sglang_env = os.environ.get("RAYORCH_TEST_SGLANG_ENV")
    vllm_env = os.environ.get("RAYORCH_TEST_VLLM_ENV")
    if not sglang_env or not vllm_env:
        pytest.skip(
            "set RAYORCH_TEST_SGLANG_ENV and RAYORCH_TEST_VLLM_ENV "
            "to run the backend environment test"
        )

    result = run(
        BackendEnvironmentPipeline(sglang_env, vllm_env),
        ["probe"],
        ray_init_kwargs={"include_dashboard": False, "num_cpus": 2},
    )
    output = result.outputs[0]

    assert output["backend"] == "vllm"
    assert output["value"]["backend"] == "sglang"
    assert output["value"]["conda_default_env"] in {
        sglang_env,
        Path(sglang_env).name,
    }
    assert output["conda_default_env"] in {
        vllm_env,
        Path(vllm_env).name,
    }
