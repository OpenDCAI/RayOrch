from rayorch.experimental.multigrain_v3_6.benchmark.mineru_cardinality_matrix import (
    _arms,
    _rotated_arms,
    build_parser,
)


def test_matrix_rotates_all_six_arms_without_changing_membership() -> None:
    first = _rotated_arms(0)
    second = _rotated_arms(1)

    assert len(first) == 6
    assert set(first) == set(second)
    assert second == first[1:] + first[:1]


def test_matrix_builds_requested_poison_arms() -> None:
    arms = _arms((0, 32))

    assert arms == (
        ("v36", 0),
        ("ray_data", 0),
        ("daft", 0),
        ("v36", 32),
        ("ray_data", 32),
        ("daft", 32),
    )
    assert _rotated_arms(2, arms) == arms[2:] + arms[:2]


def test_matrix_defaults_freeze_paper_protocol() -> None:
    args = build_parser().parse_args(
        ["--input-manifest", "manifest.jsonl", "--artifact-dir", "artifacts"]
    )

    assert args.repetitions == 5
    assert args.replicas == 4
    assert args.batch_size == 64
    assert args.work_ms == 100.0
    assert args.page_partitions == 16
    assert args.microbatch_size == 8
    assert args.max_active_microbatches == 1
    assert args.poison_counts == [0, 1]
    assert args.selection_largest is None
    assert args.healthy_max_pages is None
