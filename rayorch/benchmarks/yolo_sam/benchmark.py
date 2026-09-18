"""Small public entrypoint for the YOLO -> SAM workload."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rayorch import RunResult
from rayorch.benchmark import (
    BenchmarkReport,
    BenchmarkRun,
    LocalSource,
    benchmark_config,
    run_benchmark,
    submit_benchmark,
)

from .pipeline import YoloSamPipeline, _normalize_options


@dataclass(frozen=True, slots=True)
class YoloSamBench:
    input_path: str | Path
    output_dir: str | Path
    yolo_model: str | Path
    sam_checkpoint: str | Path
    input_limit: int | None = None
    device: str = "cuda"
    yolo_replicas: int = 1
    sam_replicas: int = 1
    batch_size: int = 4
    input_batch_size: int = 16
    max_active_input_batches: int = 2
    confidence: float = 0.25
    sam_model_type: str = "vit_b"
    overlay_alpha: float = 0.35
    stage_options: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        for name in ("input_path", "output_dir", "yolo_model", "sam_checkpoint"):
            object.__setattr__(
                self,
                name,
                str(Path(getattr(self, name)).expanduser().resolve()),
            )
        for name in (
            "yolo_replicas",
            "sam_replicas",
            "batch_size",
            "input_batch_size",
            "max_active_input_batches",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.input_limit is not None and (
            type(self.input_limit) is not int or self.input_limit <= 0
        ):
            raise ValueError("input_limit must be a positive integer or None")
        if not 0 < self.confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        if not 0 <= self.overlay_alpha <= 1:
            raise ValueError("overlay_alpha must be in [0, 1]")
        object.__setattr__(
            self,
            "stage_options",
            _normalize_options(self.stage_options),
        )

    def run(
        self,
        *,
        ray_address: str | None = None,
        ray_init_kwargs: Mapping[str, Any] | None = None,
        profile: bool = True,
        profile_interval_s: float = 1.0,
        run_id: str | None = None,
    ) -> BenchmarkReport:
        images = _load_images(Path(self.input_path), self.input_limit)
        _validate(self, images)
        return run_benchmark(
            name="yolo_sam",
            pipeline=self._pipeline(),
            source_columns=(images,),
            config=benchmark_config(self),
            artifact_root=Path(self.output_dir) / ".rayorch-benchmark",
            input_batch_size=self.input_batch_size,
            max_active_input_batches=self.max_active_input_batches,
            ray_address=ray_address,
            ray_init_kwargs=ray_init_kwargs,
            profile=profile,
            profile_interval_s=profile_interval_s,
            run_id=run_id,
            extra_metrics=_metrics,
        )

    def submit(
        self,
        target: str,
        *,
        source: LocalSource | None = None,
        profile: bool = True,
        profile_interval_s: float = 1.0,
        run_id: str | None = None,
    ) -> BenchmarkRun:
        return submit_benchmark(
            benchmark="yolo_sam",
            config=benchmark_config(self),
            artifact_root=Path(self.output_dir) / ".rayorch-benchmark",
            target=target,
            source=source,
            profile=profile,
            profile_interval_s=profile_interval_s,
            run_id=run_id,
        )

    def _pipeline(self) -> YoloSamPipeline:
        return YoloSamPipeline(
            output_dir=str(self.output_dir),
            yolo_model=str(self.yolo_model),
            sam_checkpoint=str(self.sam_checkpoint),
            device=self.device,
            yolo_replicas=self.yolo_replicas,
            sam_replicas=self.sam_replicas,
            batch_size=self.batch_size,
            confidence=self.confidence,
            sam_model_type=self.sam_model_type,
            overlay_alpha=self.overlay_alpha,
            stage_options=self.stage_options,
        )


def _load_images(path: Path, limit: int | None) -> tuple[str, ...]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.is_dir():
        images = [
            item
            for item in sorted(path.rglob("*"))
            if item.is_file() and item.suffix.lower() in suffixes
        ]
    elif path.is_file() and path.suffix.lower() in suffixes:
        images = [path]
    else:
        raise FileNotFoundError(
            f"input_path must be an image or directory of images: {path}"
        )
    if limit is not None:
        images = images[:limit]
    if not images:
        raise ValueError(f"no image inputs found under {path}")
    return tuple(str(image.resolve()) for image in images)


def _validate(benchmark: YoloSamBench, images: tuple[str, ...]) -> None:
    for name in ("yolo_model", "sam_checkpoint"):
        path = Path(getattr(benchmark, name))
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    stems = [Path(path).stem for path in images]
    if len(stems) != len(set(stems)):
        raise ValueError("image filenames must have unique stems")


def _metrics(result: RunResult) -> dict[str, int]:
    outputs = list(result.outputs)
    return {
        "images": len(outputs),
        "detections": sum(int(item["yolo_boxes"]) for item in outputs),
        "masks": sum(int(item["sam_masks"]) for item in outputs),
    }


__all__ = ["YoloSamBench"]
