"""Video v3.5 graph topology, UDF reuse and compiler parity tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rayorch.experimental.multigrain_v3.benchmark.video.caption_v3 import (
    CaptionFrames,
    SummarizeCaptionVideos,
)
from rayorch.experimental.multigrain_v3.benchmark.video.multimodal_v3 import (
    DecodeAudioChunks,
    MergeModalities,
    SummarizeAudio,
    SummarizeFrames,
    WhisperChunks,
)
from rayorch.experimental.multigrain_v3.benchmark.video.v3 import (
    DecodeFrames,
    SummarizeVideos,
    TransformFrames,
)
from rayorch.experimental.multigrain_v3_5.benchmark.video import (
    VideoCaptionV35Pipeline,
    VideoMultimodalV35Pipeline,
    VideoV35Pipeline,
)
from rayorch.experimental.multigrain_v3_5.benchmark.video import paired
from rayorch.experimental.multigrain_v3_5.benchmark.video import kinetics50
from rayorch.experimental.multigrain_v3_5.benchmark.video import model_paired
from rayorch.experimental.multigrain_v3_5.logical import ExpandOrigin, GroupOrigin


def _feature_pipeline(batch_scope: str = "elastic") -> VideoV35Pipeline:
    return VideoV35Pipeline(
        stride=3,
        max_frames=8,
        decode_replicas=2,
        reduce_replicas=1,
        transform_replicas=4,
        transform_batch_size=16,
        batch_scope=batch_scope,
    )


def _caption_pipeline(batch_scope: str = "elastic") -> VideoCaptionV35Pipeline:
    return VideoCaptionV35Pipeline(
        model_path="model",
        stride=3,
        max_frames=8,
        batch_scope=batch_scope,
    )


@pytest.mark.parametrize(
    ("pipeline", "targets"),
    [
        (
            _feature_pipeline(),
            [DecodeFrames, TransformFrames, SummarizeVideos],
        ),
        (
            _caption_pipeline(),
            [DecodeFrames, CaptionFrames, SummarizeCaptionVideos],
        ),
    ],
)
@pytest.mark.parametrize("optimize", [False, True])
def test_single_relation_video_graph_is_expand_compute_group(
    pipeline,
    targets,
    optimize,
):
    compiled = pipeline.compile(optimize=optimize)

    assert [spec.kernel.target for spec in compiled.logical.calls.values()] == targets
    assert len(compiled.logical.domains) == 2
    assert len(compiled.runtime.pools) == 3
    assert sum(
        isinstance(spec.origin, ExpandOrigin)
        for spec in compiled.logical.ports.values()
    ) == 1
    assert sum(
        isinstance(spec.origin, GroupOrigin)
        for spec in compiled.logical.ports.values()
    ) == 1


@pytest.mark.parametrize("optimize", [False, True])
def test_multimodal_video_has_two_sibling_relations_and_root_merge(optimize):
    compiled = VideoMultimodalV35Pipeline(
        whisper_model_path="whisper",
        vit_model_path="vit",
    ).compile(optimize=optimize)

    assert [spec.kernel.target for spec in compiled.logical.calls.values()] == [
        DecodeAudioChunks,
        WhisperChunks,
        SummarizeAudio,
        DecodeFrames,
        TransformFrames,
        SummarizeFrames,
        MergeModalities,
    ]
    domains = list(compiled.logical.domains.values())
    assert len(domains) == 3
    assert domains[0].parent is None
    assert domains[1].parent == domains[0].ref
    assert domains[2].parent == domains[0].ref
    assert sum(
        isinstance(spec.origin, ExpandOrigin)
        for spec in compiled.logical.ports.values()
    ) == 2
    assert sum(
        isinstance(spec.origin, GroupOrigin)
        for spec in compiled.logical.ports.values()
    ) == 2

    merge = list(compiled.logical.calls.values())[-1]
    assert merge.execution_domain == domains[0].ref
    assert all(
        compiled.logical.port(input_.port).domain == domains[0].ref
        for input_ in merge.inputs
    )


@pytest.mark.parametrize("factory", [_feature_pipeline, _caption_pipeline])
def test_video_batch_scope_only_changes_heavy_compute_pool(factory):
    elastic = factory("elastic").compile()
    parent = factory("parent_bound").compile()

    assert elastic.logical.calls == parent.logical.calls
    changed = []
    for left, right in zip(
        elastic.runtime.pools.values(),
        parent.runtime.pools.values(),
    ):
        left_options = dict(left.options)
        right_options = dict(right.options)
        if left_options != right_options:
            changed.append((left_options, right_options))
    assert len(changed) == 1
    assert changed[0][0]["batch_scope"] == "elastic"
    assert changed[0][1]["batch_scope"] == "parent_bound"


@pytest.mark.parametrize(
    "option",
    [
        {"stride": 0},
        {"max_frames": 0},
        {"batch_scope": "unknown"},
        {"transform_replicas": 0},
        {"transform_batch_size": 0},
        {"max_retries": -1},
    ],
)
def test_feature_video_rejects_invalid_physical_options(option):
    values = {
        "stride": 3,
        "max_frames": 8,
        "decode_replicas": 2,
        "reduce_replicas": 1,
        "transform_replicas": 4,
        "transform_batch_size": 16,
        "batch_scope": "elastic",
    }
    values.update(option)
    with pytest.raises(ValueError):
        VideoV35Pipeline(**values)


def test_paired_trial_materializes_v3_and_requires_exact_output(monkeypatch):
    calls = []
    expected = [{"frames": 2, "digests": ("a", "b")}]

    class OldResult:
        metrics = {"rpc_count": 3, "grains_per_rpc": 1.5}

        def get(self):
            calls.append("v3.get")
            return tuple(expected)

    def fake_v3(paths, **options):
        calls.append("v3")
        assert paths == ["video.mp4"]
        assert options["microbatch_size"] == 2
        return OldResult()

    def fake_v35(paths, **options):
        calls.append("v35")
        assert paths == ["video.mp4"]
        assert options["arena_size"] == 2
        return SimpleNamespace(
            outputs=list(expected),
            rpc_count=3,
            calls={0: SimpleNamespace(grains=2), 1: SimpleNamespace(grains=2)},
        )

    monkeypatch.setattr(paired, "run_v3", fake_v3)
    monkeypatch.setattr(paired, "run_v35", fake_v35)
    result = paired._run_trial(
        ["video.mp4"],
        {},
        arena_size=2,
        max_in_flight=1,
        order="v35_first",
    )

    assert calls == ["v35", "v3", "v3.get"]
    assert result["v3_rpc_count"] == result["v35_rpc_count"] == 3
    assert result["sampled_frames"] == [2]


def test_paired_trial_reports_first_business_output_difference(monkeypatch):
    old = SimpleNamespace(
        get=lambda: ({"frames": 1},),
        metrics={"rpc_count": 1, "grains_per_rpc": 1.0},
    )
    new = SimpleNamespace(
        outputs=[{"frames": 2}],
        rpc_count=1,
        calls={0: SimpleNamespace(grains=1)},
    )
    monkeypatch.setattr(paired, "run_v3", lambda *args, **kwargs: old)
    monkeypatch.setattr(paired, "run_v35", lambda *args, **kwargs: new)

    with pytest.raises(ValueError, match="video index 0"):
        paired._run_trial(
            ["video.mp4"],
            {},
            arena_size=1,
            max_in_flight=1,
            order="v3_first",
        )


def test_paired_cli_defaults_to_manifest_preserving_full_run():
    args = paired.build_parser().parse_args(["--manifest", "videos.json"])

    assert args.limit == 0
    assert args.repeats == 1
    assert args.arena_size == 24
    assert args.max_in_flight == 4


def test_kinetics50_plan_is_frozen_unique_and_plan_only_by_default(tmp_path):
    archives = kinetics50.archive_plan()
    summary = kinetics50.plan_summary(str(tmp_path))
    args = kinetics50.build_parser().parse_args([])

    assert len(archives) == 33
    assert len({item.url for item in archives}) == len(archives)
    assert sum(item.expected_bytes for item in archives) == 50_607_623_297
    assert summary["expected_archive_gb_decimal"] == pytest.approx(50.607623297)
    assert args.action == "plan"
    assert args.download_workers == 2


def test_caption_model_pair_allows_bounded_text_drift_but_exact_structure(
    monkeypatch,
):
    old_outputs = [
        {"frames": 1, "source_indices": (0,), "captions": ("Action.",)}
    ]
    new_outputs = [
        {"frames": 1, "source_indices": (0,), "captions": ("action",)}
    ]
    old = SimpleNamespace(
        get=lambda: tuple(old_outputs),
        metrics={"rpc_count": 3},
    )
    new = SimpleNamespace(outputs=new_outputs, rpc_count=3)
    monkeypatch.setattr(model_paired, "run_caption_v3", lambda *a, **k: old)
    monkeypatch.setattr(model_paired, "run_caption_v35", lambda *a, **k: new)

    result = model_paired._caption_trial(
        ["video.mp4"],
        {},
        arena_size=1,
        max_in_flight=1,
        order="v3_first",
        maximum_normalized_mismatch=0.02,
    )

    assert result["correctness"]["structure_exact"]
    assert result["correctness"]["caption_mismatch_rate"] == 1.0
    assert result["correctness"]["normalized_mismatch_rate"] == 0.0


def test_multimodal_model_pair_requires_exact_root_merge(monkeypatch):
    outputs = [{"audio_chunks": 2, "transcript": "ok", "frames": 4}]
    old = SimpleNamespace(get=lambda: tuple(outputs), metrics={"rpc_count": 7})
    new = SimpleNamespace(outputs=list(outputs), rpc_count=7)
    monkeypatch.setattr(model_paired, "run_multimodal_v3", lambda *a, **k: old)
    monkeypatch.setattr(model_paired, "run_multimodal_v35", lambda *a, **k: new)

    result = model_paired._multimodal_trial(
        ["video.mp4"],
        {},
        arena_size=1,
        max_in_flight=1,
        order="v35_first",
    )

    assert result["outputs_exact"]
    assert result["v3_rpc_count"] == result["v35_rpc_count"] == 7
