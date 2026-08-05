"""SmolVLM caption matrix 的 Ray-free 配置、汇总和正确性测试。"""

from __future__ import annotations

from types import SimpleNamespace

from rayorch.experimental.multigrain_v3.benchmark.video.caption_matrix import (
    CaptionMatrixConfig,
    _timeline_summary,
    build_matrix_plan,
    caption_difference_summary,
    compare_outputs,
    compile_arm,
)


def _config() -> CaptionMatrixConfig:
    """构造不加载模型的纯配置。"""

    return CaptionMatrixConfig(model_path="/models/smolvlm")


def test_formal_four_gpu_compiled_config() -> None:
    """默认正式配置应有四个一 GPU caption actor 和四个 reduce actor。"""

    config = _config()
    assert (
        config.stride,
        config.max_frames,
        config.caption_batch_size,
        config.caption_replicas,
        config.decode_replicas,
        config.reduce_replicas,
        config.max_new_tokens,
        config.microbatch_size,
        config.max_inflight_arenas,
        config.max_pending_per_actor,
    ) == (12, 4, 16, 4, 4, 4, 12, 32, 4, 4)

    compiled = compile_arm(build_matrix_plan(config)[0])
    stages = {stage.id: stage for stage in compiled.dag.stages}
    assert stages[2].execution is not None
    assert stages[2].execution.replicas == 4
    assert dict(stages[2].execution.ray_options)["num_gpus"] == 1
    assert stages[3].execution is not None
    assert stages[3].execution.replicas == 4


def test_parent_and_elastic_only_differ_by_scope() -> None:
    """两臂的资源、模型与 backpressure 配置必须完全相同。"""

    parent, elastic = build_matrix_plan(_config())
    parent_options = dict(parent.options)
    elastic_options = dict(elastic.options)
    assert parent_options.pop("batch_scope") == "parent_bound"
    assert elastic_options.pop("batch_scope") == "elastic"
    assert parent_options == elastic_options


def test_timeline_summary_names_caption_stage() -> None:
    """timeline summary 应使用 decode/caption/reduce 三个业务 stage 名。"""

    events = [
        SimpleNamespace(
            stage=1, grains=1, flush_reason="full",
            worker_started_at=1.0, worker_finished_at=2.0,
        ),
        SimpleNamespace(
            stage=2, grains=16, flush_reason="full",
            worker_started_at=1.0, worker_finished_at=3.0,
        ),
        SimpleNamespace(
            stage=2, grains=4, flush_reason="tail",
            worker_started_at=2.0, worker_finished_at=4.0,
        ),
        SimpleNamespace(
            stage=3, grains=2, flush_reason="full",
            worker_started_at=4.0, worker_finished_at=5.0,
        ),
    ]
    summary = _timeline_summary(events)
    assert summary["caption"]["batch_histogram"] == {4: 1, 16: 1}
    assert summary["caption"]["worker_busy_sum_s"] == 4.0
    assert summary["caption"]["peak_concurrency"] == 2
    assert summary["decode"]["rpc_count"] == 1
    assert summary["reduce"]["grains"] == 2


def test_compare_outputs_requires_exact_caption_fields() -> None:
    """greedy caption report 至少严格校验 frames、索引和文本。"""

    baseline = [
        {"frames": 2, "source_indices": (0, 12), "captions": ("crawl", "stand")}
    ]
    assert compare_outputs(baseline, list(baseline))["exact"] is True

    mismatch = compare_outputs(
        baseline,
        [{"frames": 2, "source_indices": (0, 12), "captions": ("crawl", "walk")}],
    )
    assert mismatch["exact"] is False
    assert mismatch["reason"] == "captions"
    assert mismatch["first_difference"]["video_index"] == 0


def test_caption_difference_summary_quantifies_text_drift() -> None:
    """文本漂移应与 frame/source 结构错误分开统计。"""

    baseline = [
        {
            "frames": 2,
            "source_indices": (0, 12),
            "captions": ("Conversation.", "A dog"),
        }
    ]
    candidate = [
        {
            "frames": 2,
            "source_indices": (0, 12),
            "captions": ("CONVERSATION", "A cat"),
        }
    ]

    result = caption_difference_summary(baseline, candidate)

    assert result["structure_exact"] is True
    assert result["caption_mismatch_frames"] == 2
    assert result["normalized_mismatch_frames"] == 1
