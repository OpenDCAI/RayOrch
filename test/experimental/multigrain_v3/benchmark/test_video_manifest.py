"""公开视频 long-tail manifest 的无模型测试。"""

from __future__ import annotations

from rayorch.experimental.multigrain_v3.benchmark.video.manifest import (
    VideoProbe,
    select_long_tail,
    select_long_tail_by_bytes,
)
from rayorch.experimental.multigrain_v3.benchmark.video.compare import (
    build_parser,
)


def test_long_tail_selector_rotates_duration_buckets_deterministically() -> None:
    """选择器应同时取短/中/长视频，且固定 seed 下顺序稳定。"""

    probes = (
        VideoProbe("short-a.mp4", 2.0, 1),
        VideoProbe("short-b.mp4", 3.0, 1),
        VideoProbe("medium-a.mp4", 10.0, 1),
        VideoProbe("medium-b.mp4", 12.0, 1),
        VideoProbe("long-a.mp4", 30.0, 1),
        VideoProbe("tail-a.mp4", 80.0, 1),
    )

    first = select_long_tail(probes, count=4, seed=7)
    second = select_long_tail(probes, count=4, seed=7)

    assert first == second
    durations = [probe.duration_s for probe in first]
    assert any(duration <= 5.0 for duration in durations)
    assert any(5.0 < duration <= 15.0 for duration in durations)
    assert any(15.0 < duration <= 45.0 for duration in durations)
    assert any(duration > 45.0 for duration in durations)


def test_byte_budget_selector_requires_both_size_and_independent_sources() -> None:
    """容量实验不能只靠一个大视频满足字节预算。"""

    probes = (
        VideoProbe("short.mp4", 2.0, 3),
        VideoProbe("medium.mp4", 10.0, 4),
        VideoProbe("long.mp4", 30.0, 9),
        VideoProbe("tail.mp4", 80.0, 20),
    )

    selected = select_long_tail_by_bytes(
        probes,
        min_count=3,
        target_bytes=15,
        seed=3,
    )

    assert len(selected) >= 3
    assert sum(probe.bytes for probe in selected) >= 15


def test_video_compare_parser_exposes_ray_gpu_capacity() -> None:
    """GPU frame stage 的 Ray capacity 必须显式配置，不能只写 actor num_gpus。"""

    args = build_parser().parse_args(
        [
            "--paths",
            "video.mp4",
            "--num-gpus",
            "1",
            "--transform-num-gpus",
            "1",
        ]
    )

    assert args.num_gpus == 1.0
    assert args.transform_num_gpus == 1.0
