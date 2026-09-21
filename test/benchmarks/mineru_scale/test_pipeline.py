"""MinerU scale graph and resource-contract tests."""

from __future__ import annotations

from rayorch.benchmarks.mineru_scale.pipeline import MinerUScalePipeline
from rayorch.benchmarks.mineru_scale.udfs import (
    MinerUScaleAssembleDoc,
    MinerUScalePdfMetadata,
    MinerUScalePdfToPages,
    MinerUScaleVlmOcrPage,
)


def test_scale_pipeline_compiles_with_reference_64_gpu_shape():
    compiled = MinerUScalePipeline(
        output_dir="/shared/output",
        model="/shared/model",
    ).compile()

    assert [
        spec.udf.target for spec in compiled.logical.calls.values()
    ] == [
        MinerUScalePdfToPages,
        MinerUScaleVlmOcrPage,
        MinerUScalePdfMetadata,
        MinerUScaleAssembleDoc,
    ]
    pools = list(compiled.plan.actor_pools_by_call.values())
    assert [pool.replicas for pool in pools] == [256, 128, 1, 64]
    assert [pool.batch_size for pool in pools] == [1, 64, 32, 4]
    assert dict(pools[1].ray_options)["num_gpus"] == 0.5
    assert pools[1].replicas * dict(pools[1].ray_options)["num_gpus"] == 64


def test_scale_pipeline_allows_explicit_small_smoke_shape():
    compiled = MinerUScalePipeline(
        output_dir="/shared/output",
        model="/shared/model",
        render_replicas=2,
        ocr_replicas=1,
        assemble_replicas=1,
        gpus_per_ocr_actor=1,
        stage_options={"ocr": {"batch_size": 8}},
    ).compile()
    pools = list(compiled.plan.actor_pools_by_call.values())

    assert [pool.replicas for pool in pools] == [2, 1, 1, 1]
    assert pools[1].batch_size == 8
    assert dict(pools[1].ray_options)["num_gpus"] == 1
