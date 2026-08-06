"""Ray-free contracts for the explicit TableFormer V1 batch kernel."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_table_workflow import (
    ExpandDoclingTableJobs,
    normalize_v1_table_grid,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_v3_table import (
    DoclingTableFormerV1BatchV3Pipeline,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.tableformer_v1_batch import (
    DoclingTableFormerV1BatchCore,
    TableFormerV1ImagePreprocessor,
    TableFormerV1Postprocessor,
    _validate_upstream_contract,
)


def test_v1_batch_pipeline_keeps_graph_and_swaps_only_table_core() -> None:
    compiled = DoclingTableFormerV1BatchV3Pipeline().compile()
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
    assert stages[5].udf.target is ExpandDoclingTableJobs
    assert stages[6].udf.target is DoclingTableFormerV1BatchCore
    assert dict(stages[6].udf.init_kwargs)["table_batch_mode"] == "v1_batch"


def test_v1_grid_normalization_is_a_shared_deterministic_postprocess() -> None:
    responses = [
        {
            "start_col_offset_idx": 8,
            "start_row_offset_idx": 20,
            "col_span": 2,
            "row_span": 1,
        },
        {
            "start_col_offset_idx": 3,
            "start_row_offset_idx": 10,
            "col_span": 1,
            "row_span": 2,
        },
    ]
    details = {}

    normalize_v1_table_grid(responses, details)

    assert responses == [
        {
            "start_col_offset_idx": 1,
            "end_col_offset_idx": 3,
            "start_row_offset_idx": 1,
            "end_row_offset_idx": 2,
            "col_span": 2,
            "row_span": 1,
        },
        {
            "start_col_offset_idx": 0,
            "end_col_offset_idx": 1,
            "start_row_offset_idx": 0,
            "end_row_offset_idx": 2,
            "col_span": 1,
            "row_span": 2,
        },
    ]
    assert details == {"num_cols": 3, "num_rows": 2}


def test_run_v3_selects_v1_batch_pipeline_without_starting_ray(monkeypatch) -> None:
    import rayorch.experimental.multigrain_v3.benchmark.document_docling.core_v3 as core_v3

    observed = {}

    class _Executor:
        def __init__(self, pipeline, **options):
            observed["pipeline"] = pipeline
            observed["options"] = options

        def run(self, paths):
            observed["paths"] = paths
            return "result"

    monkeypatch.setattr(core_v3, "Executor", _Executor)

    assert core_v3.run_v3(["doc.pdf"], table_batch_mode="v1_batch") == "result"
    assert isinstance(
        observed["pipeline"],
        DoclingTableFormerV1BatchV3Pipeline,
    )
    assert observed["paths"] == ["doc.pdf"]


def test_v1_batch_port_never_installs_runtime_replay() -> None:
    import rayorch.experimental.multigrain_v3.benchmark.document_docling.tableformer_v1_batch as module

    source = inspect.getsource(module)
    assert "object.__setattr__" not in source
    assert "._prepare_image" not in source
    assert ".predict =" not in source


def test_v1_batch_port_fails_fast_on_upstream_contract_drift(monkeypatch) -> None:
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda _: "9.9.9")
    with pytest.raises(RuntimeError, match="supports docling-ibm-models 3.13.3"):
        _validate_upstream_contract(SimpleNamespace())

    monkeypatch.setattr(importlib.metadata, "version", lambda _: "3.13.3")
    with pytest.raises(RuntimeError, match="missing _encoder"):
        _validate_upstream_contract(SimpleNamespace())


def test_v1_preprocessor_is_exactly_equivalent_to_upstream_singleton() -> None:
    numpy = pytest.importorskip("numpy")
    tf_predictor_module = pytest.importorskip(
        "docling_ibm_models.tableformer.data_management.tf_predictor"
    )
    config = {
        "dataset": {
            "image_normalization": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "resized_image": 8,
        }
    }
    image = numpy.arange(5 * 7 * 3, dtype=numpy.uint8).reshape(5, 7, 3)
    upstream = object.__new__(tf_predictor_module.TFPredictor)
    upstream._config = config
    upstream._device = "cpu"

    expected = tf_predictor_module.TFPredictor._prepare_image(upstream, image)
    actual = TableFormerV1ImagePreprocessor(config, "cpu")(image)

    assert torch.equal(actual, expected)


def test_v1_postprocessor_preserves_classes_and_scaled_lineage() -> None:
    observed = {}

    class _Matcher:
        def match_cells(self, page, table_bbox, prediction):
            observed["page"] = page
            observed["table_bbox"] = table_bbox
            observed["prediction"] = prediction
            return {
                "table_cells": [{"cell_id": 3}],
                "matches": {3: []},
                "pdf_cells": [{"id": 3}],
                "prediction_bboxes_page": [],
                "prediction": prediction,
            }

    class _Processor:
        def process(self, details, correct_overlapping_cells):
            observed["correct_overlapping_cells"] = correct_overlapping_cells
            return details

    class _Predictor:
        _cell_matcher = _Matcher()
        _post_processor = _Processor()
        _remove_padding = False
        enable_post_process = True

        @staticmethod
        def _get_html_tags(sequence):
            assert sequence == [0, 4, 1]
            return ["fcel"]

        @staticmethod
        def _check_bbox_sync(prediction):
            return True, prediction["bboxes"]

        @staticmethod
        def _generate_tf_response(table_cells, matches):
            assert table_cells == [{"cell_id": 3}]
            assert matches == {3: []}
            return [{"cell_id": 2}, {"cell_id": 1}]

        @staticmethod
        def _merge_tf_output(docling_output, pdf_cells):
            observed["sorted_docling_output"] = list(docling_output)
            observed["pdf_cells"] = pdf_cells
            return [{"merged": True}]

    job = SimpleNamespace(
        table_bbox=(2.0, 4.0, 6.0, 8.0),
        scale_factor=2.0,
        iocr_page={"tokens": [{"id": 3}]},
    )
    prediction = SimpleNamespace(
        tag_sequence=[0, 4, 1],
        classes=torch.tensor([[0.1, 0.9]]),
        coordinates=torch.tensor([[0.5, 0.5, 0.4, 0.2]]),
    )

    output, details = TableFormerV1Postprocessor(_Predictor()).run(
        job,
        prediction,
    )

    assert output == [{"merged": True}]
    assert details["prediction"]["classes"] == [1]
    assert observed["table_bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert observed["correct_overlapping_cells"] is False
    assert observed["sorted_docling_output"] == [
        {"cell_id": 1},
        {"cell_id": 2},
    ]


def test_v1_core_chunks_and_scatters_without_model_mutation(monkeypatch) -> None:
    import rayorch.experimental.multigrain_v3.benchmark.document_docling.tableformer_v1_batch as module

    core = object.__new__(DoclingTableFormerV1BatchCore)
    core.model = SimpleNamespace(marker="unchanged")
    core.preprocess = lambda value: torch.full(
        (1, 1, 1, 1),
        float(value),
    )
    core.table_batch_max_jobs = 2
    core.total_table_batch_audit = {
        "jobs": 0,
        "batches": 0,
        "batch_errors": 0,
        "serial_fallback_jobs": 0,
    }
    batch_rows = []

    def fake_predict(model, images):
        assert model is core.model
        rows = [int(value) for value in images[:, 0, 0, 0].tolist()]
        batch_rows.append(rows)
        return [([row], None, None) for row in rows]

    monkeypatch.setattr(module, "predict_tableformer_batch", fake_predict)
    core._build_table = lambda job, prediction: (
        job.name,
        prediction.tag_sequence[0],
    )
    jobs = [
        SimpleNamespace(name=f"job-{index}", table_image=index)
        for index in range(5)
    ]

    assert core.run(jobs) == [
        ("job-0", 0),
        ("job-1", 1),
        ("job-2", 2),
        ("job-3", 3),
        ("job-4", 4),
    ]
    assert batch_rows == [[0, 1], [2, 3], [4]]
    assert vars(core.model) == {"marker": "unchanged"}
    assert core.batch_audit() == {
        "jobs": 5,
        "batches": 3,
        "batch_errors": 0,
        "serial_fallback_jobs": 0,
    }
