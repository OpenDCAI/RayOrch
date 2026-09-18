"""Small public entrypoint for the MinerU workload."""

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

from .pipeline import MinerUPipeline, normalize_stage_options
from .udfs import DEFAULT_MODEL


@dataclass(frozen=True, slots=True)
class MinerUBench:
    """Run PDF -> page -> OCR -> document with a configurable GPU actor pool."""

    input_path: str | Path
    output_dir: str | Path
    model: str | Path = DEFAULT_MODEL
    input_limit: int | None = 4
    num_gpus: int = 1
    batch_size: int = 64
    input_batch_size: int = 24
    max_active_input_batches: int = 3
    gpu_memory_utilization: float = 0.8
    render_dpi: int = 200
    stage_options: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        for name in ("input_path", "output_dir", "model"):
            path = Path(getattr(self, name)).expanduser().resolve()
            object.__setattr__(self, name, str(path))
        for name in (
            "num_gpus",
            "batch_size",
            "input_batch_size",
            "max_active_input_batches",
            "render_dpi",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.input_limit is not None and (
            type(self.input_limit) is not int or self.input_limit <= 0
        ):
            raise ValueError("input_limit must be a positive integer or None")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        object.__setattr__(
            self,
            "stage_options",
            normalize_stage_options(self.stage_options),
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
        pdfs = _load_pdfs(Path(self.input_path), self.input_limit)
        _validate(pdfs, Path(self.model))
        return run_benchmark(
            name="mineru",
            pipeline=self._pipeline(),
            source_columns=(pdfs,),
            config=benchmark_config(self),
            artifact_root=Path(self.output_dir) / ".rayorch-benchmark",
            input_batch_size=self.input_batch_size,
            max_active_input_batches=self.max_active_input_batches,
            ray_address=ray_address,
            ray_init_kwargs=ray_init_kwargs,
            profile=profile,
            profile_interval_s=profile_interval_s,
            run_id=run_id,
            extra_metrics=_mineru_metrics,
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
            benchmark="mineru",
            config=benchmark_config(self),
            artifact_root=Path(self.output_dir) / ".rayorch-benchmark",
            target=target,
            source=source,
            profile=profile,
            profile_interval_s=profile_interval_s,
            run_id=run_id,
        )

    def _pipeline(self) -> MinerUPipeline:
        return MinerUPipeline(
            output_dir=str(self.output_dir),
            model=str(self.model),
            num_gpus=self.num_gpus,
            batch_size=self.batch_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            render_dpi=self.render_dpi,
            stage_options=self.stage_options,
        )


def _load_pdfs(path: Path, limit: int | None) -> tuple[str, ...]:
    if path.is_dir():
        pdfs = [item for item in sorted(path.rglob("*.pdf")) if item.is_file()]
    elif path.suffix.lower() == ".pdf" and path.is_file():
        pdfs = [path]
    else:
        raise FileNotFoundError(
            f"input_path must be a PDF or directory of PDFs: {path}"
        )
    if limit is not None:
        pdfs = pdfs[:limit]
    if not pdfs:
        raise ValueError(f"no PDF inputs found under {path}")
    return tuple(str(pdf) for pdf in pdfs)


def _validate(pdfs: tuple[str, ...], model: Path) -> None:
    missing = [path for path in pdfs if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"PDF input does not exist: {missing[0]}")
    if not model.exists():
        raise FileNotFoundError(f"MinerU model does not exist: {model}")
    stems = [Path(path).stem for path in pdfs]
    if len(stems) != len(set(stems)):
        raise ValueError("PDF filenames must have unique stems")


def _mineru_metrics(result: RunResult) -> dict[str, Any]:
    outputs = list(result.outputs)
    pages = sum(int(output["pages"]) for output in outputs)
    return {
        "pages": pages,
        "pages_per_s": round(pages / max(result.elapsed_s, 1e-9), 4),
    }


__all__ = ["MinerUBench"]
