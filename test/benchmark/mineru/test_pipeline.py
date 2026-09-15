"""MinerU production graph structure tests."""

from __future__ import annotations

from rayorch.benchmark.mineru.udfs import (
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    PdfMetadata,
)
from rayorch.benchmark.mineru.pipeline import (
    MinerUPipeline,
)
from rayorch._program.logical import ExpandOrigin, ReduceOrigin


def _pipeline() -> MinerUPipeline:
    return MinerUPipeline(
        output_dir="/tmp/rayorch-mineru-test",
        model="model",
        render={"replicas": 4, "batch_size": 1, "num_cpus": 1},
        ocr={
            "replicas": 4,
            "batch_size": 64,
            "num_gpus": 1.0,
            "num_cpus": 1,
        },
        metadata={"replicas": 1, "batch_size": 32, "num_cpus": 1},
        assemble={"replicas": 4, "batch_size": 4, "num_cpus": 1},
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
