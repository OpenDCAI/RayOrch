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
    pools = list(compiled.plan.actor_pools_by_call.values())

    assert targets == [
        VllmGenerate,
        BuildRefinementPrompts,
        VllmGenerate,
        BuildResults,
    ]
    assert dict(pools[0].ray_options)["num_gpus"] == 2.0
    assert dict(pools[2].ray_options)["num_gpus"] == 2.0
    assert pools[0].batch_size == 8
    assert pools[2].batch_size == 4
