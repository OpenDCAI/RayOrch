from __future__ import annotations

from rayorch._program.logical import ExpandOrigin, ReduceOrigin
from rayorch.benchmarks.panda70m.pipeline import Panda70MPipeline
from rayorch.benchmarks.panda70m.udfs import (
    DecodePandaTeacherFrames,
    ExpandPandaClips,
    PandaFusedTeacher,
    SelectPandaCaption,
    SummarizePandaSource,
)


def test_panda_pipeline_uses_current_public_benchmark_api():
    compiled = Panda70MPipeline(
        output_dir="results/panda70m",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        teacher_replicas=2,
    ).compile()

    assert [spec.udf.target for spec in compiled.logical.calls.values()] == [
        ExpandPandaClips,
        DecodePandaTeacherFrames,
        PandaFusedTeacher,
        SelectPandaCaption,
        SummarizePandaSource,
    ]
    assert len(compiled.plan.actor_pools_by_call) == 5
    assert len(compiled.logical.domains) == 2
    assert sum(
        isinstance(spec.origin, ExpandOrigin)
        for spec in compiled.logical.ports.values()
    ) == 1
    assert sum(
        isinstance(spec.origin, ReduceOrigin)
        for spec in compiled.logical.ports.values()
    ) == 1

    teacher_pool = list(compiled.plan.actor_pools_by_call.values())[2]
    assert teacher_pool.replicas == 8
    assert teacher_pool.batch_size == 8
    assert dict(teacher_pool.ray_options) == {
        "num_gpus": 1.0,
        "num_cpus": 1,
    }


def test_panda_stage_options_are_validated_and_applied():
    pipeline = Panda70MPipeline(
        output_dir="results/panda70m",
        model="model",
        stage_options={
            "decode": {"replicas": 3, "num_cpus": 4},
            "teacher": {"batch_size": 4, "resources": {"accelerator": 1}},
        },
    )
    pools = list(pipeline.compile().plan.actor_pools_by_call.values())
    assert pools[1].replicas == 3
    assert dict(pools[1].ray_options) == {"num_cpus": 4}
    assert pools[2].batch_size == 4
    assert dict(pools[2].ray_options) == {
        "num_gpus": 1.0,
        "num_cpus": 1,
        "resources": {"accelerator": 1},
    }
