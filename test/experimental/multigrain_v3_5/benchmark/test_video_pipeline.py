"""Video v3.5 graph topology, UDF reuse and compiler parity tests."""

from __future__ import annotations

import json
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
from rayorch.experimental.multigrain_v3_5.benchmark.video import audit
from rayorch.experimental.multigrain_v3_5.benchmark.video import paired
from rayorch.experimental.multigrain_v3_5.benchmark.video import kinetics50
from rayorch.experimental.multigrain_v3_5.benchmark.video import model_paired
from rayorch.experimental.multigrain_v3_5.logical import ExpandOrigin, GroupOrigin
from rayorch.experimental.multigrain_v3_5.model import CallRef


def _fake_v35_result(outputs, *, rpc_count: int, grains: int):
    metric = SimpleNamespace(
        actor_starts=1,
        rpcs=rpc_count,
        grains=grains,
        average_batch=grains / rpc_count,
        batch_sizes=[grains],
        retries=0,
    )
    return SimpleNamespace(
        outputs=outputs,
        rpc_count=rpc_count,
        calls={CallRef(0): metric},
        workers={},
        elapsed_s=0.1,
        max_active_arenas=1,
        released_values=grains,
    )


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
        return _fake_v35_result(list(expected), rpc_count=3, grains=4)

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
    assert result["sampled_frame_count"] == 2
    assert result["sampled_frame_histogram"] == {2: 1}
    assert result["correctness"]["outputs_exact"]


def test_paired_trial_reports_first_business_output_difference(monkeypatch):
    old = SimpleNamespace(
        get=lambda: ({"frames": 1},),
        metrics={"rpc_count": 1, "grains_per_rpc": 1.0},
    )
    new = _fake_v35_result([{"frames": 2}], rpc_count=1, grains=1)
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


def test_paired_model_output_allows_only_bounded_digest_drift(monkeypatch):
    old = SimpleNamespace(
        get=lambda: (
            {
                "frames": 2,
                "source_indices": (0, 4),
                "digests": ("a", "b"),
                "mean_edge_density": 0.5,
            },
        ),
        metrics={"rpc_count": 1, "grains_per_rpc": 2.0},
    )
    new = _fake_v35_result(
        [
            {
                "frames": 2,
                "source_indices": (0, 4),
                "digests": ("a", "c"),
                "mean_edge_density": 0.5,
            }
        ],
        rpc_count=1,
        grains=2,
    )
    monkeypatch.setattr(paired, "run_v3", lambda *args, **kwargs: old)
    monkeypatch.setattr(paired, "run_v35", lambda *args, **kwargs: new)

    result = paired._run_trial(
        ["video.mp4"],
        {},
        arena_size=1,
        max_in_flight=1,
        order="v3_first",
        maximum_digest_mismatch=0.5,
    )

    assert result["correctness"]["structure_exact"]
    assert result["correctness"]["digest_mismatch_rate"] == 0.5
    assert not result["correctness"]["outputs_exact"]


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


def test_kinetics50_manifest_keeps_split_and_shard_identity(
    tmp_path,
    monkeypatch,
):
    video_root = tmp_path / "videos"
    first = video_root / "val" / "part_0" / "same.mp4"
    second = video_root / "train" / "part_0" / "same.mp4"
    rejected = video_root / "train" / "part_1" / "bad.mp4"
    for path in (first, second, rejected):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")

    probes = (
        SimpleNamespace(path=str(first.resolve()), duration_s=1.0, bytes=5),
        SimpleNamespace(path=str(second.resolve()), duration_s=2.0, bytes=5),
    )
    monkeypatch.setattr(kinetics50, "probe_candidates", lambda root: probes)
    output = tmp_path / "manifest.json"

    report = kinetics50.build_decodable_manifest(str(tmp_path), str(output))

    assert report["candidate_files"] == 3
    assert report["decodable_files"] == 2
    assert report["rejected_files"] == 1
    assert report["source_identity_unique"]
    assert report["split_counts"] == {"train": 1, "val": 1}
    rows = json.loads(output.read_text())
    assert [row["source_id"] for row in rows] == [
        "val/part_0/same.mp4",
        "train/part_0/same.mp4",
    ]


def test_kinetics50_manifest_action_requires_explicit_output():
    with pytest.raises(ValueError, match="manifest-output"):
        kinetics50.main(["--action", "manifest"])


def test_full_video_audit_decodes_and_hashes_manifest(tmp_path):
    import cv2
    import numpy as np

    video = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot create MJPG AVI fixture")
    try:
        for index in range(3):
            writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "path": str(video),
                    "dataset": "fixture",
                    "split": "test",
                    "source_id": "tiny",
                    "bytes": video.stat().st_size,
                }
            ]
        )
    )

    report = audit.audit_manifest(str(manifest), workers=1)

    assert report["passed"]
    assert report["clips"] == 1
    assert report["decoded_frames"] == 3
    assert report["decode_errors"] == 0
    assert report["content_duplicate_files"] == 0

    duplicate_manifest = tmp_path / "duplicate-manifest.json"
    rows = json.loads(manifest.read_text())
    rows.append({**rows[0], "source_id": "same-content-second-source"})
    duplicate_manifest.write_text(json.dumps(rows))

    duplicate_report = audit.audit_manifest(
        str(duplicate_manifest),
        workers=1,
    )

    assert not duplicate_report["passed"]
    assert duplicate_report["source_identity_unique"]
    assert duplicate_report["content_duplicate_groups"] == 1
    assert duplicate_report["content_duplicate_files"] == 2


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
    new = _fake_v35_result(new_outputs, rpc_count=3, grains=1)
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
    new = _fake_v35_result(list(outputs), rpc_count=7, grains=1)
    monkeypatch.setattr(model_paired, "run_multimodal_v3", lambda *a, **k: old)
    monkeypatch.setattr(model_paired, "run_multimodal_v35", lambda *a, **k: new)

    result = model_paired._multimodal_trial(
        ["video.mp4"],
        {},
        arena_size=1,
        max_in_flight=1,
        order="v35_first",
    )

    assert result["correctness"]["outputs_exact"]
    assert result["v3_rpc_count"] == result["v35_rpc_count"] == 7


def test_multimodal_difference_separates_lineage_frames_and_text():
    baseline = [
        {
            "audio_chunks": 2,
            "transcript": "Action.",
            "frames": 4,
            "frame_digest": "same",
        },
        {
            "audio_chunks": 1,
            "transcript": "left",
            "frames": 3,
            "frame_digest": "old",
        },
    ]
    candidate = [
        {
            "audio_chunks": 2,
            "transcript": "action",
            "frames": 4,
            "frame_digest": "same",
        },
        {
            "audio_chunks": 2,
            "transcript": "right",
            "frames": 3,
            "frame_digest": "new",
        },
    ]

    difference = model_paired.multimodal_difference_summary(
        baseline,
        candidate,
    )

    assert not difference["outputs_exact"]
    assert not difference["structure_exact"]
    assert difference["structure_mismatch_videos"] == 1
    assert not difference["frame_digests_exact"]
    assert difference["frame_digest_mismatch_videos"] == 1
    assert difference["transcript_mismatch_videos"] == 2
    assert difference["normalized_transcript_mismatch_videos"] == 1
    assert difference["first_difference"]["video_index"] == 0


def test_multimodal_pair_allows_only_bounded_normalized_text_drift(
    monkeypatch,
):
    old_outputs = [
        {
            "audio_chunks": 1,
            "transcript": "left",
            "frames": 2,
            "frame_digest": "same",
        }
    ]
    new_outputs = [
        {
            "audio_chunks": 1,
            "transcript": "right",
            "frames": 2,
            "frame_digest": "same",
        }
    ]
    old = SimpleNamespace(get=lambda: tuple(old_outputs), metrics={"rpc_count": 3})
    new = _fake_v35_result(new_outputs, rpc_count=3, grains=1)
    monkeypatch.setattr(model_paired, "run_multimodal_v3", lambda *a, **k: old)
    monkeypatch.setattr(model_paired, "run_multimodal_v35", lambda *a, **k: new)

    with pytest.raises(ValueError, match="normalized transcript mismatch rate"):
        model_paired._multimodal_trial(
            ["video.mp4"],
            {},
            arena_size=1,
            max_in_flight=1,
            order="v3_first",
            maximum_normalized_mismatch=0.02,
        )

    result = model_paired._multimodal_trial(
        ["video.mp4"],
        {},
        arena_size=1,
        max_in_flight=1,
        order="v3_first",
        maximum_normalized_mismatch=1.0,
    )
    assert not result["correctness"]["outputs_exact"]
    assert result["correctness"]["structure_exact"]
    assert result["correctness"]["frame_digests_exact"]
