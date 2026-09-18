"""MinerU's thin input and configuration adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from rayorch.benchmarks.mineru.benchmark import _load_pdfs, _validate


def test_input_path_accepts_one_pdf_or_directory(tmp_path: Path):
    first = tmp_path / "a.pdf"
    second = tmp_path / "nested" / "b.pdf"
    first.write_bytes(b"%PDF")
    second.parent.mkdir()
    second.write_bytes(b"%PDF")

    assert _load_pdfs(first, None) == (str(first.resolve()),)
    assert _load_pdfs(tmp_path, 1) == (str(first.resolve()),)


def test_validation_checks_files_model_and_unique_stems(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    first = tmp_path / "a.pdf"
    first.write_bytes(b"%PDF")
    _validate((str(first),), model)
    with pytest.raises(FileNotFoundError, match="PDF input"):
        _validate((str(tmp_path / "missing.pdf"),), model)
    with pytest.raises(FileNotFoundError, match="model"):
        _validate((str(first),), tmp_path / "missing-model")

    duplicate = tmp_path / "nested" / "a.pdf"
    duplicate.parent.mkdir()
    duplicate.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="unique stems"):
        _validate((str(first), str(duplicate)), model)
