"""CPU-only compiler and UDF contracts for the V3 MinerU bridge."""

from __future__ import annotations

import builtins
import inspect
from typing import Any, get_type_hints

import pytest

from rayorch.experimental.multigrain_v3.benchmark.mineru import (
    DummyAssembleDoc,
    DummyDocument,
    DummyOcrContent,
    DummyOcrPage,
    DummyPage,
    DummyPdf,
    DummyPdfToPages,
    MinerUAssembleDoc,
    MinerUDummyPipeline,
    MinerUPdfToPages,
    MinerUV3Pipeline,
    MinerUVlmOcrPage,
    artifact_key_for_pdf,
    compile_pipeline,
    make_dummy_pdfs,
    real_runtime_env,
)
from rayorch.experimental.multigrain_v3.benchmark.mineru_cli import build_parser


def _op_names(pipeline) -> tuple[str, ...]:
    """Compile a pipeline and return stable frozen operation type names."""

    graph = compile_pipeline(pipeline)
    return tuple(type(node.op).__name__ for node in graph.nodes)


def test_real_pipeline_compiles_to_map_system_map_system_map(
    monkeypatch,
    tmp_path,
):
    """Compilation must not import model packages or turn system ops into UDFs."""

    real_import = builtins.__import__
    forbidden = {"flash_mineru", "mineru_vl_utils", "vllm"}

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError(f"unexpected eager import of {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    pipeline = MinerUV3Pipeline(
        output_dir=str(tmp_path),
        model="/not/loaded/model",
        replicas=3,
        batch_size=7,
        render_replicas=2,
        assemble_replicas=2,
        runtime_env={"env_vars": {"MINERU_TEST": "1"}},
    )
    assert _op_names(pipeline) == (
        "SourceOp",
        "MapOp",
        "ExpandOp",
        "MapOp",
        "ReduceOp",
        "MapOp",
    )
    graph = compile_pipeline(pipeline)
    map_ops = [
        node.op for node in graph.nodes if type(node.op).__name__ == "MapOp"
    ]
    assert [
        operation.execution.resources.num_gpus for operation in map_ops
    ] == [0.0, 1.0, 0.0]
    assert map_ops[1].execution.resources.replicas == 3
    assert map_ops[1].execution.batch.max_size == 7


def test_dummy_pipeline_has_identical_cpu_only_semantic_topology():
    """The dependency-free graph must exercise the real structural bridge."""

    pipeline = MinerUDummyPipeline(
        replicas=2,
        batch_size=3,
        render_replicas=1,
        assemble_replicas=1,
    )
    graph = compile_pipeline(pipeline)
    assert tuple(type(node.op).__name__ for node in graph.nodes) == (
        "SourceOp",
        "MapOp",
        "ExpandOp",
        "MapOp",
        "ReduceOp",
        "MapOp",
    )
    map_ops = [
        node.op for node in graph.nodes if type(node.op).__name__ == "MapOp"
    ]
    assert all(op.execution.resources.num_gpus == 0.0 for op in map_ops)


def test_udf_run_signatures_use_strict_physical_batch_annotations():
    """Every UDF input/output leaf must expose the V3 outer-list batch ABI."""

    render_hints = get_type_hints(MinerUPdfToPages.run)
    assert render_hints["pdfs"] == list[str]
    assert render_hints["return"] == list[list[dict[str, Any]]]
    ocr_hints = get_type_hints(MinerUVlmOcrPage.run)
    assert ocr_hints == {
        "pages": list[dict[str, Any]],
        "return": list[Any],
    }
    real_assemble_hints = get_type_hints(MinerUAssembleDoc.run)
    assert real_assemble_hints == {
        "pdfs": list[str],
        "contents": list[list[Any]],
        "pages": list[list[dict[str, Any]]],
        "return": list[dict[str, Any]],
    }

    dummy_render_hints = get_type_hints(DummyPdfToPages.run)
    assert dummy_render_hints == {
        "pdfs": list[DummyPdf],
        "return": list[list[DummyPage]],
    }
    dummy_ocr_hints = get_type_hints(DummyOcrPage.run)
    assert dummy_ocr_hints == {
        "pages": list[DummyPage],
        "return": list[DummyOcrContent],
    }
    assemble_hints = get_type_hints(DummyAssembleDoc.run)
    assert assemble_hints == {
        "pdfs": list[DummyPdf],
        "contents": list[list[DummyOcrContent]],
        "pages": list[list[DummyPage]],
        "return": list[DummyDocument],
    }
    for udf in (
        MinerUPdfToPages,
        MinerUVlmOcrPage,
        MinerUAssembleDoc,
        DummyPdfToPages,
        DummyOcrPage,
        DummyAssembleDoc,
    ):
        signature = inspect.signature(udf.run)
        assert all(
            parameter.annotation is not inspect.Parameter.empty
            for name, parameter in signature.parameters.items()
            if name != "self"
        )
        assert signature.return_annotation is not inspect.Signature.empty


def test_dummy_udfs_cover_variable_fanout_and_ordered_assembly():
    """Direct UDF calls validate multiple documents and zero-page assembly."""

    pdfs = make_dummy_pdfs(5, page_counts=(3, 1, 4, 2, 0))
    page_groups = DummyPdfToPages().run(pdfs)
    assert [len(group) for group in page_groups] == [3, 1, 4, 2, 0]
    content_groups = [DummyOcrPage().run(group) for group in page_groups]
    documents = DummyAssembleDoc(separator="|").run(
        pdfs,
        content_groups,
        page_groups,
    )
    assert [document.page_ids for document in documents] == [
        (0, 1, 2),
        (0,),
        (0, 1, 2, 3),
        (0, 1),
        (),
    ]
    assert documents[-1].markdown == ""

    with pytest.raises(ValueError, match="unordered"):
        DummyAssembleDoc().run(
            [pdfs[0]],
            [list(reversed(content_groups[0]))],
            [page_groups[0]],
        )


def test_real_runtime_env_is_copied_and_prepends_flash_repo(monkeypatch):
    """Runtime environment construction must not mutate caller configuration."""

    monkeypatch.setenv("PYTHONPATH", "/inherited")
    base = {"env_vars": {"EXAMPLE": "yes"}}
    configured = real_runtime_env("/flash/repo", base)
    assert configured is not base
    assert base == {"env_vars": {"EXAMPLE": "yes"}}
    assert configured["env_vars"]["EXAMPLE"] == "yes"
    assert configured["env_vars"]["PYTHONPATH"].split(":")[0] == "/flash/repo"


def test_mineru_artifact_keys_include_stable_full_path_digest(tmp_path):
    """Equal stems from different directories must never share artifact paths."""

    left = tmp_path / "left" / "report.pdf"
    right = tmp_path / "right" / "report.pdf"
    left_key = artifact_key_for_pdf(str(left))
    right_key = artifact_key_for_pdf(str(right))

    assert left_key.startswith("report-")
    assert right_key.startswith("report-")
    assert left_key != right_key
    assert artifact_key_for_pdf(str(left)) == left_key


def test_cli_exposes_explicit_safe_dummy_and_real_configuration():
    """CLI defaults stay CPU-safe while accepting all real workload paths."""

    defaults = build_parser().parse_args([])
    assert defaults.workload == "dummy"
    assert defaults.limit == 4
    assert defaults.replicas == 4
    assert defaults.batch_size == 16
    assert defaults.runtime_env == {}

    args = build_parser().parse_args(
        [
            "--workload",
            "real",
            "--flash-repo",
            "/flash",
            "--input-dir",
            "/pdfs",
            "--model",
            "/model",
            "--limit",
            "8",
            "--replicas",
            "4",
            "--batch-size",
            "16",
            "--runtime-env",
            '{"env_vars":{"TOKENIZERS_PARALLELISM":"false"}}',
            "--output",
            "/output",
            "--result",
            "/result.jsonl",
        ]
    )
    assert args.workload == "real"
    assert args.flash_repo == "/flash"
    assert args.input_dir == "/pdfs"
    assert args.model == "/model"
    assert args.limit == 8
    assert args.replicas == 4
    assert args.batch_size == 16
    assert args.runtime_env == {
        "env_vars": {"TOKENIZERS_PARALLELISM": "false"}
    }
    assert args.output_dir == "/output"
    assert args.result_jsonl == "/result.jsonl"
