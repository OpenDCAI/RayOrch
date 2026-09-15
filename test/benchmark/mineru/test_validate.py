"""MinerU input identity and historical comparison tests."""

from __future__ import annotations

import pytest

from rayorch.benchmark.mineru.validate import (
    corpus_digest,
    golden_comparable,
    load_pdf_paths,
    validate_inputs,
)


def test_pdf_manifest_preserves_order_and_limit(tmp_path):
    manifest = tmp_path / "corpus.jsonl"
    manifest.write_text(
        '{"path":"/data/b.pdf","pages":9}\n'
        '{"path":"/data/a.pdf","pages":3}\n',
        encoding="utf-8",
    )

    assert load_pdf_paths(str(manifest)) == ("/data/b.pdf", "/data/a.pdf")
    assert load_pdf_paths(str(manifest), limit=1) == ("/data/b.pdf",)


def test_pdf_manifest_accepts_json_array(tmp_path):
    manifest = tmp_path / "corpus.json"
    manifest.write_text(
        '[{"path":"/data/a.pdf","pages":3},'
        '{"path":"/data/b.pdf","pages":9}]',
        encoding="utf-8",
    )

    assert load_pdf_paths(str(manifest)) == ("/data/a.pdf", "/data/b.pdf")


def test_mineru_inputs_are_validated_before_ray_startup(tmp_path):
    flash_repo = tmp_path / "flash-mineru"
    model = tmp_path / "model"
    pdf = tmp_path / "input.pdf"
    flash_repo.mkdir()
    model.mkdir()
    pdf.write_bytes(b"%PDF")

    validate_inputs((str(pdf),), flash_repo=str(flash_repo), model=str(model))

    with pytest.raises(FileNotFoundError, match="PDF inputs"):
        validate_inputs(
            (str(tmp_path / "missing.pdf"),),
            flash_repo=str(flash_repo),
            model=str(model),
        )
    with pytest.raises(FileNotFoundError, match="Flash-MinerU"):
        validate_inputs(
            (str(pdf),),
            flash_repo=str(tmp_path / "missing-flash"),
            model=str(model),
        )
    validate_inputs(
        (str(pdf),),
        flash_repo=str(tmp_path / "missing-flash"),
        model=str(model),
        packaged_runtime=True,
    )
    with pytest.raises(FileNotFoundError, match="MinerU model"):
        validate_inputs(
            (str(pdf),),
            flash_repo=str(flash_repo),
            model=str(tmp_path / "missing-model"),
        )

    duplicate_stem = tmp_path / "nested" / "input.pdf"
    duplicate_stem.parent.mkdir()
    duplicate_stem.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="unique stems"):
        validate_inputs(
            (str(pdf), str(duplicate_stem)),
            flash_repo=str(flash_repo),
            model=str(model),
        )


def test_corpus_digest_tracks_order_paths_and_contents(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    digest = corpus_digest((str(first), str(second)))
    assert len(digest) == 64
    assert digest != corpus_digest((str(second), str(first)))

    second.write_bytes(b"changed")
    assert digest != corpus_digest((str(first), str(second)))


def test_historical_performance_gate_requires_exact_corpus_and_shape():
    digest = "a" * 64
    assert golden_comparable(
        corpus_digest=digest,
        expected_digest=digest.upper(),
        pdf_count=368,
        page_count=7_072,
    )
    assert not golden_comparable(
        corpus_digest=digest,
        expected_digest="b" * 64,
        pdf_count=368,
        page_count=7_072,
    )
    assert not golden_comparable(
        corpus_digest=digest,
        expected_digest=digest,
        pdf_count=368,
        page_count=7_071,
    )
    with pytest.raises(ValueError, match="64-character hex digest"):
        golden_comparable(
            corpus_digest=digest,
            expected_digest="not-a-digest",
            pdf_count=368,
            page_count=7_072,
        )
