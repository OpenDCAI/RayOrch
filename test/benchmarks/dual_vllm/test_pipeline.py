"""Dual-vLLM Pipeline structure."""

from __future__ import annotations

from rayorch.benchmarks.dual_vllm.pipeline import DualVllmPipeline
from rayorch.benchmarks.dual_vllm.udfs import (
    BuildRefinementPrompts,
    BuildResults,
    VllmGenerate,
)


def test_dual_vllm_pipeline_has_two_independent_model_calls():
    compiled = DualVllmPipeline(
        model_a="/models/a",
        model_b="/models/b",
        tensor_parallel_size=2,
        batch_size=8,
        stage_options={"model_b": {"batch_size": 4}},
    ).compile()
    targets = [spec.udf.target for spec in compiled.logical.calls.values()]
    calls = list(compiled.logical.calls)
    pools = [compiled.plan.pool(call) for call in calls]
    dispatches = [compiled.plan.dispatch(call) for call in calls]

    assert targets == [
        VllmGenerate,
        BuildRefinementPrompts,
        VllmGenerate,
        BuildResults,
    ]
    assert dict(pools[0].ray_options)["num_gpus"] == 2.0
    assert dict(pools[2].ray_options)["num_gpus"] == 2.0
    assert dispatches[0].batch_size == 8
    assert dispatches[2].batch_size == 4
