"""SGLang-vLLM Pipeline structure and environment placement."""

from __future__ import annotations

from rayorch.benchmarks.sglang_vllm.pipeline import SglangVllmPipeline
from rayorch.benchmarks.sglang_vllm.udfs import (
    BuildHandoffPrompts,
    BuildResults,
    SglangGenerate,
    VllmGenerate,
)


def test_sglang_vllm_pipeline_assigns_one_runtime_environment_per_backend():
    compiled = SglangVllmPipeline(
        model="/models/model",
        sglang_env="sglang-env",
        vllm_env="vllm-env",
        sglang_tensor_parallel_size=2,
        vllm_tensor_parallel_size=4,
        batch_size=8,
        stage_options={"vllm": {"batch_size": 3}},
    ).compile()
    targets = [spec.udf.target for spec in compiled.logical.calls.values()]
    calls = list(compiled.logical.calls)
    pools = [compiled.plan.pool(call) for call in calls]

    assert targets == [
        SglangGenerate,
        BuildHandoffPrompts,
        VllmGenerate,
        BuildResults,
    ]
    assert dict(pools[0].ray_options)["runtime_env"] == {
        "conda": "sglang-env"
    }
    assert dict(pools[0].ray_options)["num_gpus"] == 2.0
    assert "runtime_env" not in dict(pools[1].ray_options)
    assert dict(pools[2].ray_options)["runtime_env"] == {"conda": "vllm-env"}
    assert dict(pools[2].ray_options)["num_gpus"] == 4.0
    assert compiled.plan.dispatch(calls[2]).batch_size == 3


def test_stage_options_can_replace_default_runtime_environment():
    compiled = SglangVllmPipeline(
        model="/models/model",
        sglang_env="sglang-env",
        vllm_env="vllm-env",
        stage_options={
            "sglang": {"runtime_env": {"conda": "custom-sglang"}},
        },
    ).compile()
    first_call = next(iter(compiled.logical.calls))
    first_pool = compiled.plan.pool(first_call)

    assert dict(first_pool.ray_options)["runtime_env"] == {
        "conda": "custom-sglang"
    }
