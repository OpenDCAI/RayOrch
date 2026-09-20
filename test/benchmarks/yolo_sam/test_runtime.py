"""Dependency-free end-to-end execution of the YOLO-SAM graph."""

from __future__ import annotations

from pathlib import Path

from rayorch import RayModule
from rayorch.benchmark import YoloSamBench
from rayorch.benchmarks.yolo_sam import benchmark as yolo_sam
from rayorch.benchmarks.yolo_sam.pipeline import YoloSamPipeline


class _Load:
    def run(self, paths):
        return [f"image:{path}" for path in paths], [
            {"input_path": path} for path in paths
        ]


class _Detect:
    def run(self, images, metadata):
        return images, [{**item, "yolo_boxes": 2} for item in metadata]


class _Segment:
    def run(self, images, metadata):
        return [[{"mask": index}] for index, _ in enumerate(images)], [
            {**item, "sam_masks": 1} for item in metadata
        ]


class _Render:
    def run(self, images, masks, metadata):
        return [f"overlay:{image}" for image in images], metadata


class _Save:
    def run(self, images, metadata):
        return [
            {**item, "output_path": f"/output/{index}.jpg"}
            for index, item in enumerate(metadata)
        ]


class _Metadata:
    def run(self, outputs):
        return outputs


class _SyntheticPipeline(YoloSamPipeline):
    def __init__(self):
        self.load = RayModule(_Load, num_outputs=2).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )
        self.yolo = RayModule(_Detect, num_outputs=2).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )
        self.sam = RayModule(_Segment, num_outputs=2).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )
        self.render = RayModule(_Render, num_outputs=2).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )
        self.save = RayModule(_Save).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )
        self.metadata = RayModule(_Metadata).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )


class _SyntheticBench(YoloSamBench):
    def _pipeline(self):
        return _SyntheticPipeline()


def test_yolo_sam_benchmark_runs_end_to_end(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        yolo_sam,
        "_load_images",
        lambda path, limit: ("a.jpg", "b.jpg"),
    )
    monkeypatch.setattr(yolo_sam, "_validate", lambda benchmark, images: None)

    report = _SyntheticBench(
        input_path=tmp_path / "images",
        output_dir=tmp_path / "output",
        yolo_model=tmp_path / "yolo.pt",
        sam_checkpoint=tmp_path / "sam.pth",
        batch_size=2,
        input_batch_size=2,
    ).run(
        ray_init_kwargs={"include_dashboard": False, "num_gpus": 0},
        profile=False,
        run_id="synthetic-yolo-sam",
    )

    assert report.metrics["images"] == 2
    assert report.metrics["detections"] == 4
    assert report.metrics["masks"] == 2
    assert len(report.outputs) == 2
