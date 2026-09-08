from __future__ import annotations

import json

from rayorch.experimental.multigrain_v3_6 import GroupFailure
from rayorch.experimental.multigrain_v3_6.benchmark.mineru_cardinality import (
    CardinalityPipeline,
    CountHealthyParent,
    ManifestParentToPages,
    SyntheticPageWork,
    _expected,
    _load_parents,
    build_parser,
)


def test_manifest_expand_and_group_failure_preserve_exact_cardinality() -> None:
    groups = ManifestParentToPages().run(
        [{"parent_id": 7, "pages": 4, "poison": True}]
    )

    assert [page["page_id"] for page in groups[0]] == [0, 1, 2, 3]
    assert [page["poison"] for page in groups[0]] == [True, False, False, False]
    values = SyntheticPageWork(work_ms=0).run(groups[0])
    assert isinstance(values[0], GroupFailure)
    assert values[1:] == [1, 2, 3]


def test_cardinality_pipeline_compiles_four_calls() -> None:
    pipeline = CardinalityPipeline(
        replicas=2,
        batch_size=4,
        work_ms=0,
        expand_replicas=1,
        reduce_replicas=1,
        cpu_worker_resource=None,
        gpu_worker_resource=None,
    )

    assert len(pipeline.compile().logical.calls) == 4


def test_manifest_selection_poisons_largest_parents(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"path": str(tmp_path / f"{index}.pdf"), "pages": pages})
            for index, pages in enumerate((2, 9, 5, 9))
        )
        + "\n",
        encoding="utf-8",
    )

    parents, poisoned = _load_parents(str(manifest), poison_largest=2)

    assert poisoned == (1, 3)
    assert [parent["poison"] for parent in parents] == [False, True, False, True]
    assert _expected(parents, poisoned) == {
        "documents": 4,
        "pages": 25,
        "poisoned_documents": 2,
        "poisoned_parent_pages": 18,
        "healthy_documents": 2,
        "healthy_pages": 7,
    }


def test_manifest_selection_retains_large_candidates_and_short_healthy(
    tmp_path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"path": str(tmp_path / f"{index}.pdf"), "pages": pages})
            for index, pages in enumerate((2, 20, 5, 30, 4))
        )
        + "\n",
        encoding="utf-8",
    )

    healthy, no_poison = _load_parents(
        str(manifest),
        poison_largest=0,
        selection_largest=2,
        healthy_max_pages=4,
    )
    poisoned, poison_indices = _load_parents(
        str(manifest),
        poison_largest=2,
        selection_largest=2,
        healthy_max_pages=4,
    )

    assert [parent["pages"] for parent in healthy] == [2, 20, 30, 4]
    assert no_poison == ()
    assert [parent["pages"] for parent in poisoned] == [2, 20, 30, 4]
    assert poison_indices == (1, 2)
    assert [parent["poison"] for parent in poisoned] == [False, True, True, False]


def test_healthy_parent_counter_checks_alignment() -> None:
    output = CountHealthyParent().run(
        [[0, 1]],
        [[{"page_id": 0}, {"page_id": 1}]],
        [3],
    )

    assert output == [{"parent_id": 3, "pages": 2}]


def test_parser_freezes_full_corpus_probe_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--engine",
            "v36",
            "--input-manifest",
            "manifest.jsonl",
            "--artifact-dir",
            "artifacts",
        ]
    )

    assert args.replicas == 32
    assert args.batch_size == 8
    assert args.work_ms == 2.0
    assert args.microbatch_size == 32
    assert args.max_active_microbatches == 8
    assert args.page_partitions is None
    assert args.selection_largest is None
    assert args.healthy_max_pages is None


def test_parser_exposes_all_three_control_engines() -> None:
    for engine in ("v36", "ray_data", "daft"):
        args = build_parser().parse_args(
            [
                "--engine",
                engine,
                "--input-manifest",
                "manifest.jsonl",
                "--artifact-dir",
                "artifacts",
            ]
        )

        assert args.engine == engine
