"""Ray-free tests for the independent TableFormerV2 V3 adapter."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_table_workflow import (
    ExpandDoclingTableV2Jobs,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_v3_table import (
    DoclingTableFormerV2BatchV3Pipeline,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.tableformer_v2_batch import (
    DoclingTableFormerV2BatchCore,
    generate_tableformer_v2_batch,
    match_prepared_text,
    prepare_text_cells,
    trim_token_ids_at_eos,
)


class _FakeTableFormerV2:
    data_cells = {2}

    def __init__(self) -> None:
        self.decode_calls = 0
        self.encode_calls = 0

    def encode_images(self, images):
        self.encode_calls += 1
        return {"last_hidden_state": images[:, :1, :1, :1]}

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        encoder_outputs,
        past_key_values,
        use_cache,
        return_dict,
    ):
        del attention_mask, encoder_outputs, past_key_values, return_dict
        batch_size = input_ids.size(0)
        logits = torch.zeros(batch_size, 1, 3)
        if use_cache:
            self.decode_calls += 1
            if self.decode_calls == 1:
                logits[0, 0, 1] = 10
                logits[1, 0, 2] = 10
            else:
                # Row 0 deliberately stops emitting EOS.  The adapter must
                # fence it while row 1 reaches EOS.
                logits[0, 0, 2] = 10
                logits[1, 0, 1] = 10
            bboxes = None
        else:
            bboxes = torch.tensor([[0.1, 0.2, 0.8, 0.9]])
        return SimpleNamespace(
            logits=logits,
            past_key_values=(self.decode_calls,),
            predicted_bboxes=bboxes,
        )


def test_batch_generation_fences_finished_rows_and_splits_bboxes() -> None:
    model = _FakeTableFormerV2()
    tokenizer = SimpleNamespace(bos_token_id=0, eos_token_id=1)

    generated, bboxes = generate_tableformer_v2_batch(
        model,
        torch.zeros(2, 3, 4, 4),
        tokenizer,
        max_length=4,
    )

    assert generated.tolist() == [[0, 1, 1], [0, 2, 1]]
    assert [len(value) for value in bboxes] == [0, 1]
    assert model.encode_calls == 1
    assert model.decode_calls == 2


def test_trim_token_ids_stops_at_first_eos() -> None:
    assert trim_token_ids_at_eos(torch.tensor([0, 2, 1, 2, 1]), 1).tolist() == [
        0,
        2,
        1,
    ]


def test_text_cell_matching_precomputes_rectangles_and_keeps_docling_rule() -> None:
    calls = [0, 0]

    def cell(index, box, text):
        def to_bounding_box():
            calls[index] += 1
            return SimpleNamespace(
                l=box[0],
                t=box[1],
                r=box[2],
                b=box[3],
                coord_origin="TOPLEFT",
            )

        return SimpleNamespace(
            text=text,
            rect=SimpleNamespace(to_bounding_box=to_bounding_box),
        )

    prepared = prepare_text_cells(
        (
            cell(0, (0, 0, 10, 10), " first "),
            cell(1, (8, 0, 18, 10), "second"),
        ),
        strip_text=True,
    )
    target = SimpleNamespace(
        l=0,
        t=0,
        r=12,
        b=10,
        coord_origin="TOPLEFT",
    )

    assert match_prepared_text(target, prepared, overlap=0.3) == "first second"
    assert match_prepared_text(target, prepared, overlap=0.5) == "first"
    assert calls == [1, 1]


def test_v2_pipeline_is_distinct_and_keeps_the_same_graph_shape() -> None:
    compiled = DoclingTableFormerV2BatchV3Pipeline().compile()
    stages = compiled.dag.stages

    assert [stage.kind.value for stage in stages] == [
        "source",
        "expand",
        "map",
        "map",
        "map",
        "expand",
        "map",
        "reduce",
        "reduce",
    ]
    assert stages[5].udf.target is ExpandDoclingTableV2Jobs
    assert stages[6].udf.target is DoclingTableFormerV2BatchCore
    assert dict(stages[6].udf.init_kwargs)["table_batch_mode"] == "v2_batch"
