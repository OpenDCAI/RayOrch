"""MinerU production graph structure tests."""

from __future__ import annotations

import pytest

from rayorch.benchmarks.mineru.udfs import (
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    PdfMetadata,
)
from rayorch.benchmarks.mineru.pipeline import (
    MinerUPipeline,
    normalize_stage_options,
)
from rayorch._program.logical import ExpandOrigin, ReduceOrigin


def _pipeline() -> MinerUPipeline:
    return MinerUPipeline(
        output_dir="/tmp/rayorch-mineru-test",
        model="model",
        num_gpus=4,
        batch_size=64,
    )


def test_mineru_pipeline_has_four_calls_and_no_structural_pools():
    compiled = _pipeline().compile()
    targets = [spec.udf.target for spec in compiled.logical.calls.values()]

    assert targets == [
        MinerUPdfToPages,
        MinerUVlmOcrPage,
        PdfMetadata,
        MinerUAssembleDoc,
    ]
    assert len(compiled.plan.actor_pools_by_call) == 4
    assert all(
        "runtime_env" not in dict(pool.ray_options)
        for pool in compiled.plan.actor_pools_by_call.values()
    )
    assert len(compiled.logical.domains) == 2
    assert sum(
        isinstance(spec.origin, ExpandOrigin)
        for spec in compiled.logical.ports.values()
    ) == 1
    assert sum(
        isinstance(spec.origin, ReduceOrigin)
        for spec in compiled.logical.ports.values()
    ) == 2


def test_stage_options_override_only_selected_ray_options():
    compiled = MinerUPipeline(
        output_dir="/tmp/rayorch-mineru-test",
        model="model",
        num_gpus=4,
        batch_size=64,
        stage_options={
            "render": {"replicas": 2, "num_cpus": 3},
            "ocr": {"batch_size": 16, "resources": {"accelerator": 1}},
            "assemble": {"replicas": 1},
        },
    ).compile()
    pools = list(compiled.plan.actor_pools_by_call.values())

    assert pools[0].replicas == 2
    assert pools[0].batch_size == 1
    assert dict(pools[0].ray_options) == {"num_cpus": 3}
    assert pools[1].replicas == 4
    assert pools[1].batch_size == 16
    assert dict(pools[1].ray_options) == {
        "num_gpus": 1.0,
        "num_cpus": 1,
        "resources": {"accelerator": 1},
    }
    assert pools[2].replicas == 1
    assert pools[2].batch_size == 32
    assert dict(pools[2].ray_options) == {"num_cpus": 1}
    assert pools[3].replicas == 1
    assert pools[3].batch_size == 4
    assert dict(pools[3].ray_options) == {"num_cpus": 1}


def test_stage_options_reject_unknown_stages_and_non_mapping_values():
    with pytest.raises(ValueError, match="unknown MinerU stage"):
        normalize_stage_options({"unknown": {"replicas": 1}})
    with pytest.raises(TypeError, match="must be a mapping"):
        normalize_stage_options({"ocr": 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="num_outputs"):
        normalize_stage_options({"ocr": {"num_outputs": 2}})
