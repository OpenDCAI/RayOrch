"""MinerU scale configuration and input discovery tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rayorch.benchmarks.mineru_scale.benchmark import (
    MinerUScaleBench,
    _load_pdfs,
)


def test_local_inputs_are_recursive_ordered_and_limited(tmp_path: Path):
    first = tmp_path / "a.pdf"
    second = tmp_path / "nested" / "b.pdf"
    first.write_bytes(b"%PDF")
    second.parent.mkdir()
    second.write_bytes(b"%PDF")

    assert _load_pdfs((str(tmp_path),), None) == (
        str(first),
        str(second),
    )
    assert _load_pdfs((str(tmp_path),), 1) == (str(first),)


def test_duplicate_stems_across_roots_are_rejected(tmp_path: Path):
    first = tmp_path / "one" / "same.pdf"
    second = tmp_path / "two" / "same.pdf"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"%PDF")
    second.write_bytes(b"%PDF")

    with pytest.raises(ValueError, match="duplicate"):
        _load_pdfs((str(first.parent), str(second.parent)), None)


def test_benchmark_defaults_encode_reference_scale(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    bench = MinerUScaleBench(
        input_paths=("hdfs://namenode/pdfs",),
        output_dir=tmp_path / "output",
        model=model,
    )

    assert bench.input_paths == ("hdfs://namenode/pdfs",)
    assert bench.render_replicas == 256
    assert bench.ocr_replicas == 128
    assert bench.assemble_replicas == 64
    assert bench.ocr_replicas * bench.gpus_per_ocr_actor == 64
    assert bench.artifact_dir == str(
        (tmp_path / "output" / ".rayorch-benchmark").resolve()
    )
