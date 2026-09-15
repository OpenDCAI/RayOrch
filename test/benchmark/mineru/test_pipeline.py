"""MinerU pipeline structure, UDF reuse, manifest, and CLI regression tests."""

from __future__ import annotations

import pytest

from rayorch.benchmark.mineru.udfs import (
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    PdfMetadata,
)
from rayorch.benchmark.mineru.pipeline import (
    MinerUPipeline,
    _corpus_digest,
    _golden_comparable,
    _load_pdf_paths,
    _validate_inputs,
    build_parser,
)
from rayorch.multigrain.program.logical import ExpandOrigin, ReduceOrigin


def _pipeline() -> MinerUPipeline:
    return MinerUPipeline(
        output_dir="/tmp/rayorch-mineru-test",
        model="model",
        replicas=4,
        batch_size=64,
        gpu_memory_utilization=0.8,
        render_replicas=4,
        reduce_replicas=4,
        runtime_env={},
    )


def test_mineru_pipeline_has_four_calls_and_no_structural_pools():
    compiled = _pipeline().compile()
    targets = [spec.udf.target for spec in compiled.logical.calls.values()]

    assert targets == [
        MinerUPdfToPages,
        MinerUVlmOcrPage,
        PdfMetadata,
        MinerUAssembleDoc,
    ]
    assert len(compiled.plan.actor_pools_by_call) == 4
    assert len(compiled.logical.domains) == 2
    assert sum(
        isinstance(spec.origin, ExpandOrigin)
        for spec in compiled.logical.ports.values()
    ) == 1
    assert sum(
        isinstance(spec.origin, ReduceOrigin)
        for spec in compiled.logical.ports.values()
    ) == 2


def test_mineru_cli_defaults_to_four_pdf_correctness_gate():
    args = build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--artifact-dir",
            "/tmp/artifacts",
            "--result-jsonl",
            "/tmp/results.jsonl",
        ]
    )

    assert args.limit == 4
    assert args.batch_size == 64
    assert args.max_active_microbatches == 3
    assert args.num_cpus == 32
    assert args.object_store_gb == 100
    assert args.input_manifest is None
    assert args.golden_corpus_sha256 is None


def test_pdf_manifest_preserves_order_and_limit(tmp_path):
    manifest = tmp_path / "corpus.jsonl"
    manifest.write_text(
        '{"path":"/data/b.pdf","pages":9}\n'
        '{"path":"/data/a.pdf","pages":3}\n',
        encoding="utf-8",
    )

    assert _load_pdf_paths(str(manifest)) == ("/data/b.pdf", "/data/a.pdf")
    assert _load_pdf_paths(str(manifest), limit=1) == ("/data/b.pdf",)


def test_pdf_manifest_accepts_json_array(tmp_path):
    manifest = tmp_path / "corpus.json"
    manifest.write_text(
        '[{"path":"/data/a.pdf","pages":3},'
        '{"path":"/data/b.pdf","pages":9}]',
        encoding="utf-8",
    )

    assert _load_pdf_paths(str(manifest)) == ("/data/a.pdf", "/data/b.pdf")


def test_mineru_inputs_are_validated_before_ray_startup(tmp_path):
    flash_repo = tmp_path / "flash-mineru"
    model = tmp_path / "model"
    pdf = tmp_path / "input.pdf"
    flash_repo.mkdir()
    model.mkdir()
    pdf.write_bytes(b"%PDF")

    _validate_inputs((str(pdf),), flash_repo=str(flash_repo), model=str(model))

    with pytest.raises(FileNotFoundError, match="PDF inputs"):
        _validate_inputs(
            (str(tmp_path / "missing.pdf"),),
            flash_repo=str(flash_repo),
            model=str(model),
        )
    with pytest.raises(FileNotFoundError, match="Flash-MinerU"):
        _validate_inputs(
            (str(pdf),),
            flash_repo=str(tmp_path / "missing-flash"),
            model=str(model),
        )
    _validate_inputs(
        (str(pdf),),
        flash_repo=str(tmp_path / "missing-flash"),
        model=str(model),
        packaged_runtime=True,
    )
    with pytest.raises(FileNotFoundError, match="MinerU model"):
        _validate_inputs(
            (str(pdf),),
            flash_repo=str(flash_repo),
            model=str(tmp_path / "missing-model"),
        )

    duplicate_stem = tmp_path / "nested" / "input.pdf"
    duplicate_stem.parent.mkdir()
    duplicate_stem.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="unique stems"):
        _validate_inputs(
            (str(pdf), str(duplicate_stem)),
            flash_repo=str(flash_repo),
            model=str(model),
        )


def test_corpus_digest_tracks_order_paths_and_contents(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    digest = _corpus_digest((str(first), str(second)))
    assert len(digest) == 64
    assert digest != _corpus_digest((str(second), str(first)))

    second.write_bytes(b"changed")
    assert digest != _corpus_digest((str(first), str(second)))


def test_historical_performance_gate_requires_exact_corpus_and_shape():
    digest = "a" * 64
    assert _golden_comparable(
        corpus_digest=digest,
        expected_digest=digest.upper(),
        pdf_count=368,
        page_count=7_072,
    )
    assert not _golden_comparable(
        corpus_digest=digest,
        expected_digest="b" * 64,
        pdf_count=368,
        page_count=7_072,
    )
    assert not _golden_comparable(
        corpus_digest=digest,
        expected_digest=digest,
        pdf_count=368,
        page_count=7_071,
    )
    with pytest.raises(ValueError, match="64-character hex digest"):
        _golden_comparable(
            corpus_digest=digest,
            expected_digest="not-a-digest",
            pdf_count=368,
            page_count=7_072,
        )
