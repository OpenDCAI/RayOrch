"""Input and configuration checks for the YOLO-SAM Benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest

from rayorch.benchmark import YoloSamBench, benchmark_config
from rayorch.benchmarks.yolo_sam.benchmark import _load_images


def test_yolo_sam_configuration_round_trips(tmp_path: Path):
    benchmark = YoloSamBench(
        input_path=tmp_path / "images",
        output_dir=tmp_path / "output",
        yolo_model=tmp_path / "yolo.pt",
        sam_checkpoint=tmp_path / "sam.pth",
        stage_options={"sam": {"num_gpus": 0.5}},
    )

    assert YoloSamBench(**benchmark_config(benchmark)) == benchmark


def test_load_images_accepts_one_image_or_directory(tmp_path: Path):
    first = tmp_path / "a.jpg"
    second = tmp_path / "nested" / "b.png"
    first.write_bytes(b"image")
    second.parent.mkdir()
    second.write_bytes(b"image")

    assert _load_images(first, None) == (str(first.resolve()),)
    assert _load_images(tmp_path, 1) == (str(first.resolve()),)

    empty = tmp_path / "nested-empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no image"):
        _load_images(empty, None)
