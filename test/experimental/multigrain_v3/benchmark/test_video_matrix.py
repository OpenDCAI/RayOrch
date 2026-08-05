"""Video A V3 matrix 的 Ray-free 配置与 timeline 汇总测试。"""

from __future__ import annotations

from types import SimpleNamespace

from rayorch.experimental.multigrain_v3.benchmark.video.matrix import (
    VideoMatrixConfig,
    _timeline_summary,
    build_matrix_plan,
    compile_arm,
)


def _config() -> VideoMatrixConfig:
    """构造不访问模型路径的四卡 OpenCV 配置。"""

    return VideoMatrixConfig(
        transform_backend="opencv",
        model_path=None,
        decode_replicas=3,
        reduce_replicas=2,
        transform_replicas=4,
        transform_num_gpus=1.0,
        transform_batch_size=16,
    )


def test_four_gpu_compiled_config_has_four_one_gpu_transform_actors() -> None:
    """编译 DAG 应保留四个 transform replica 和每 actor 一张 GPU。"""

    plan = build_matrix_plan(_config())
    compiled = compile_arm(plan[0])
    stages = {stage.id: stage for stage in compiled.dag.stages}
    transform = stages[2]
    decode = stages[1]
    reduce = stages[3]

    assert transform.execution is not None
    assert transform.execution.replicas == 4
    assert dict(transform.execution.ray_options)["num_gpus"] == 1.0
    assert decode.execution is not None and decode.execution.replicas == 3
    assert reduce.execution is not None and reduce.execution.replicas == 2


def test_parent_and_elastic_differ_only_by_batch_scope() -> None:
    """两臂必须固定所有资源、模型和 batching 参数。"""

    parent, elastic = build_matrix_plan(_config())
    parent_options = dict(parent.options)
    elastic_options = dict(elastic.options)

    assert parent_options.pop("batch_scope") == "parent_bound"
    assert elastic_options.pop("batch_scope") == "elastic"
    assert parent_options == elastic_options


def test_timeline_summary_reports_video_stage_batch_and_busy_metrics() -> None:
    """summary 应按 decode/transform/reduce 归约实际 RPC 与并发信息。"""

    events = [
        SimpleNamespace(
            stage=1,
            grains=1,
            flush_reason="full",
            worker_started_at=1.0,
            worker_finished_at=2.0,
        ),
        SimpleNamespace(
            stage=2,
            grains=16,
            flush_reason="full",
            worker_started_at=1.0,
            worker_finished_at=3.0,
        ),
        SimpleNamespace(
            stage=2,
            grains=4,
            flush_reason="tail",
            worker_started_at=2.0,
            worker_finished_at=4.0,
        ),
        SimpleNamespace(
            stage=3,
            grains=2,
            flush_reason="full",
            worker_started_at=4.0,
            worker_finished_at=5.0,
        ),
    ]

    summary = _timeline_summary(events)

    assert summary["transform"]["rpc_count"] == 2
    assert summary["transform"]["batch_histogram"] == {4: 1, 16: 1}
    assert summary["transform"]["worker_busy_sum_s"] == 4.0
    assert summary["transform"]["worker_span_s"] == 3.0
    assert summary["transform"]["peak_concurrency"] == 2
    assert summary["decode"]["rpc_count"] == 1
    assert summary["reduce"]["grains"] == 2
