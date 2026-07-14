from __future__ import annotations

import pytest

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.ir import (
    FilterByMaskOp,
    FilterOp,
    MapOp,
    RelatedFrom,
    RelateOp,
    SameAs,
    SubsetOf,
)

from test.experimental.multigrain.test_pdf_mvp import PdfToImages


class KeepPages:
    def run(self, pages: list[str], *unused_columns) -> list[bool]:
        return ["bad" not in page for page in pages]


class ScoreAndKeep:
    def run(self, pages: list[str]) -> tuple[list[bool], list[float]]:
        scores = [0.9 if "good" in page else 0.2 for page in pages]
        return [score >= 0.5 for score in scores], scores


class MatchImagesAndCaptions:
    def run(self, images: list[str], captions: list[str]) -> list[dict[str, int | str]]:
        return [
            {
                "image_idx": image_index,
                "caption_idx": caption_index,
                "pair": f"{image}:{caption}",
            }
            for image_index, image in enumerate(images)
            for caption_index, caption in enumerate(captions)
            if image[-1] == caption[-1]
        ]


def pair_relation(
    pairs: list[dict[str, int | str]],
) -> list[tuple[dict[str, int | str], dict[str, int]]]:
    return [
        (
            pair,
            {
                "image": int(pair["image_idx"]),
                "caption": int(pair["caption_idx"]),
            },
        )
        for pair in pairs
    ]


def test_filter_eager_preserves_kept_identity_and_marks_business_drop() -> None:
    pages = mg.source(["good-0", "bad-1", "good-2"], name="page")
    meta = pages.with_values(
        [{"page": 0}, {"page": 1}, {"page": 2}],
        name="page_meta",
        op_name="make_meta",
    )

    kept_pages, kept_meta = mg.Filter(KeepPages)(pages, meta)

    assert kept_pages.values == ["good-0", "good-2"]
    assert kept_meta.values == [{"page": 0}, {"page": 2}]
    assert kept_pages.record_ids == [pages.record_ids[0], pages.record_ids[2]]
    assert kept_meta.record_ids == kept_pages.record_ids
    assert kept_pages.errors == []
    assert kept_pages.lineage == [("KeepPages",), ("KeepPages",)]


def test_select_eager_returns_filtered_inputs_and_annotations() -> None:
    pages = mg.source(["good-0", "bad-1", "good-2"], name="page")

    kept_pages, kept_scores = mg.Select(ScoreAndKeep, num_annotations=1)(pages)

    assert kept_pages.values == ["good-0", "good-2"]
    assert kept_scores.values == [0.9, 0.9]
    assert kept_pages.record_ids == kept_scores.record_ids
    assert kept_pages.lineage == [
        ("ScoreAndKeep__filter",),
        ("ScoreAndKeep__filter",),
    ]
    assert kept_scores.lineage == [
        ("ScoreAndKeep__map", "ScoreAndKeep__filter"),
        ("ScoreAndKeep__map", "ScoreAndKeep__filter"),
    ]


class FilterSelectPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.keep_pages = mg.Filter(KeepPages)
        self.score_and_keep = mg.Select(ScoreAndKeep, num_annotations=1)

    def forward(self, pages):
        kept = self.keep_pages(pages)
        selected, scores = self.score_and_keep(kept)
        return selected, scores


def test_filter_and_select_lower_to_canonical_ir_shape() -> None:
    graph = FilterSelectPipe().compile()

    assert tuple(node.name for node in graph.nodes) == (
        "KeepPages",
        "ScoreAndKeep__map",
        "ScoreAndKeep__filter",
    )
    assert isinstance(graph.node("KeepPages").operation, FilterOp)
    assert isinstance(graph.node("KeepPages").outputs[0].relation, SubsetOf)

    select_map = graph.node("ScoreAndKeep__map")
    assert isinstance(select_map.operation, MapOp)
    assert len(select_map.outputs) == 2
    assert all(isinstance(output.relation, SameAs) for output in select_map.outputs)

    select_filter = graph.node("ScoreAndKeep__filter")
    assert isinstance(select_filter.operation, FilterByMaskOp)
    assert len(select_filter.inputs) == 3
    assert len(select_filter.outputs) == 2

    assert [port.node for port in graph.outputs] == [
        "ScoreAndKeep__filter",
        "ScoreAndKeep__filter",
    ]


class InvalidFilterCrossGrainPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.expand = mg.Expand(PdfToImages, {"a.pdf": 2}, parent=0)
        self.keep = mg.Filter(KeepPages)

    def forward(self, documents):
        pages = self.expand(documents)
        return self.keep(documents, pages)


def test_filter_rejects_cross_grain_inputs_like_map() -> None:
    with pytest.raises(ValueError, match="same-grain"):
        InvalidFilterCrossGrainPipe().compile()


class RelatePipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.match = mg.Relate(
            MatchImagesAndCaptions,
            roles=("image", "caption"),
            output_grain="pair",
            relation_adapter=(
                "test.experimental.multigrain.relate_adapters:pair_from_fields"
            ),
        )

    def forward(self, images, captions):
        return self.match(images, captions)


def test_relate_records_invocation_local_mn_relation_contract() -> None:
    graph = RelatePipe().compile()
    relate = graph.node("MatchImagesAndCaptions")

    assert isinstance(relate.operation, RelateOp)
    assert relate.outputs[0].grain == "pair"
    relation = relate.outputs[0].relation
    assert isinstance(relation, RelatedFrom)
    assert tuple(binding.role for binding in relation.roles) == (
        "image",
        "caption",
    )
    assert [port.name for port in relate.inputs] == ["images", "captions"]


def test_relate_eager_is_not_faked_with_portbatch() -> None:
    images = mg.source(["img-0"], name="image")
    captions = mg.source(["cap-0"], name="caption")

    with pytest.raises(NotImplementedError, match="relation_fn"):
        mg.Relate(MatchImagesAndCaptions)(images, captions)


def test_relate_eager_uses_adapter_relation_evidence() -> None:
    images = mg.source(["img-0", "img-1"], name="image")
    captions = mg.source(["cap-1", "cap-0"], name="caption")

    pairs = mg.Relate(
        MatchImagesAndCaptions,
        roles=("image", "caption"),
        output_grain="pair",
        relation_fn=pair_relation,
    )(images, captions)

    assert pairs.values == [
        {"image_idx": 0, "caption_idx": 1, "pair": "img-0:cap-0"},
        {"image_idx": 1, "caption_idx": 0, "pair": "img-1:cap-1"},
    ]
    assert pairs.display_keys == [
        "image=img-0/caption=cap-0",
        "image=img-1/caption=cap-1",
    ]
    assert [
        [(ref.role, ref.port, ref.display_key) for ref in relation]
        for relation in pairs.relations
    ] == [
        [("image", "image", "img-0"), ("caption", "caption", "cap-0")],
        [("image", "image", "img-1"), ("caption", "caption", "cap-1")],
    ]
    assert pairs.trace_item(image="img-0")[0]["relations"][0]["role"] == "image"
    assert pairs.record_ids == [
        "MatchImagesAndCaptions:image=image:0|caption=caption:1",
        "MatchImagesAndCaptions:image=image:1|caption=caption:0",
    ]


def test_relate_adapter_inherits_upstream_ancestry() -> None:
    docs = mg.source(["d0", "d1"], name="docs")
    images = docs.with_values(["img-0", "img-1"], name="image", op_name="images")
    captions = mg.source(["cap-1", "cap-0"], name="caption")

    pairs = mg.Relate(
        MatchImagesAndCaptions,
        roles=("image", "caption"),
        relation_fn=pair_relation,
    )(images, captions)

    assert pairs.ancestors[0]["docs"] == docs.record_ids[0]
    assert pairs.ancestor_display[0]["docs"] == docs.display_keys[0]


def test_relate_runtime_check_rejects_out_of_range_parent_ref() -> None:
    images = mg.source(["img-0"], name="image")
    captions = mg.source(["cap-0"], name="caption")

    def bad_relation(pairs):
        return [(pairs[0], {"image": 5, "caption": 0})]

    with pytest.raises(IndexError, match="out of range"):
        mg.Relate(
            MatchImagesAndCaptions,
            roles=("image", "caption"),
            relation_fn=bad_relation,
        )(images, captions)
