"""Public Benchmark entrypoint for the scale-oriented MinerU workload."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from rayorch import RunResult
from rayorch.benchmark import (
    BenchmarkReport,
    BenchmarkRun,
    LocalSource,
    benchmark_config,
    run_benchmark,
    submit_benchmark,
)

from .pipeline import MinerUScalePipeline, normalize_stage_options
from .udfs import DEFAULT_MODEL, is_hdfs_uri, pdf_stem


@dataclass(frozen=True, slots=True)
class MinerUScaleBench:
    """Run the production MinerU topology with proven 64-GPU defaults."""

    input_paths: tuple[str | Path, ...]
    output_dir: str | Path
    model: str | Path = DEFAULT_MODEL
    artifact_dir: str | Path | None = None
    input_limit: int | None = None
    render_replicas: int = 256
    ocr_replicas: int = 128
    assemble_replicas: int = 64
    batch_size: int = 64
    input_batch_size: int = 24
    max_active_input_batches: int = 24
    gpu_memory_utilization: float = 0.32
    gpus_per_ocr_actor: float = 0.5
    render_dpi: int = 200
    stage_options: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        inputs = tuple(_normalize_location(value) for value in self.input_paths)
        if not inputs:
            raise ValueError("input_paths must contain at least one PDF location")
        object.__setattr__(self, "input_paths", inputs)

        output = _normalize_location(self.output_dir)
        object.__setattr__(self, "output_dir", output)

        model = str(Path(self.model).expanduser().resolve())
        object.__setattr__(self, "model", model)

        artifact = self.artifact_dir
        if artifact is None:
            artifact_path = (
                Path.cwd() / "results" / "mineru-scale"
                if is_hdfs_uri(output)
                else Path(output) / ".rayorch-benchmark"
            )
        else:
            artifact_path = Path(artifact).expanduser().resolve()
        object.__setattr__(self, "artifact_dir", str(artifact_path.resolve()))

        for name in (
            "render_replicas",
            "ocr_replicas",
            "assemble_replicas",
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
        if self.gpus_per_ocr_actor <= 0:
            raise ValueError("gpus_per_ocr_actor must be positive")
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
        pdfs = _load_pdfs(self.input_paths, self.input_limit)
        _validate(pdfs, Path(self.model))
        return run_benchmark(
            name="mineru_scale",
            pipeline=self._pipeline(),
            source_columns=(pdfs,),
            config=benchmark_config(self),
            artifact_root=str(self.artifact_dir),
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
            benchmark="mineru_scale",
            config=benchmark_config(self),
            artifact_root=str(self.artifact_dir),
            target=target,
            source=source,
            profile=profile,
            profile_interval_s=profile_interval_s,
            run_id=run_id,
        )

    def _pipeline(self) -> MinerUScalePipeline:
        return MinerUScalePipeline(
            output_dir=str(self.output_dir),
            model=str(self.model),
            render_replicas=self.render_replicas,
            ocr_replicas=self.ocr_replicas,
            assemble_replicas=self.assemble_replicas,
            batch_size=self.batch_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            gpus_per_ocr_actor=self.gpus_per_ocr_actor,
            render_dpi=self.render_dpi,
            stage_options=self.stage_options,
        )


def _normalize_location(value: str | Path) -> str:
    text = str(value)
    if is_hdfs_uri(text):
        return text.rstrip("/")
    return str(Path(text).expanduser().resolve())


def _load_pdfs(
    paths: Sequence[str],
    limit: int | None,
) -> tuple[str, ...]:
    pdfs: list[str] = []
    for location in paths:
        pdfs.extend(
            _load_hdfs_pdfs(location)
            if is_hdfs_uri(location)
            else _load_local_pdfs(Path(location))
        )
    if limit is not None:
        pdfs = pdfs[:limit]
    if not pdfs:
        raise ValueError("no PDF inputs found")
    stems = [pdf_stem(path) for path in pdfs]
    duplicates = sorted(stem for stem in set(stems) if stems.count(stem) > 1)
    if duplicates:
        raise ValueError(
            "PDF inputs must have unique output stems; duplicate: "
            + duplicates[0]
        )
    return tuple(pdfs)


def _load_local_pdfs(path: Path) -> list[str]:
    if path.is_dir():
        values = [item for item in sorted(path.rglob("*.pdf")) if item.is_file()]
    elif path.suffix.lower() == ".pdf" and path.is_file():
        values = [path]
    else:
        raise FileNotFoundError(
            f"input location must be a PDF or directory of PDFs: {path}"
        )
    return [str(value) for value in values]


def _load_hdfs_pdfs(uri: str) -> list[str]:
    from pyarrow import fs as pyarrow_fs  # pyright: ignore[reportMissingImports]

    filesystem, root = pyarrow_fs.FileSystem.from_uri(uri)
    root_info = filesystem.get_file_info(root)
    if root_info.type == pyarrow_fs.FileType.File:
        infos = [root_info] if Path(root_info.path).suffix.lower() == ".pdf" else []
    elif root_info.type == pyarrow_fs.FileType.Directory:
        selector = pyarrow_fs.FileSelector(
            root,
            recursive=True,
            allow_not_found=False,
        )
        infos = [
            info
            for info in filesystem.get_file_info(selector)
            if info.type == pyarrow_fs.FileType.File
            and Path(info.path).suffix.lower() == ".pdf"
        ]
    else:
        raise FileNotFoundError(f"HDFS PDF location does not exist: {uri}")

    parsed = urlsplit(uri)
    return sorted(
        urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                f"/{info.path.lstrip('/')}",
                "",
                "",
            )
        )
        for info in infos
    )


def _validate(pdfs: tuple[str, ...], model: Path) -> None:
    missing = [
        path
        for path in pdfs
        if not is_hdfs_uri(path) and not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"PDF input does not exist: {missing[0]}")
    if not model.exists():
        raise FileNotFoundError(f"MinerU model does not exist: {model}")


def _mineru_metrics(result: RunResult) -> dict[str, Any]:
    outputs = list(result.outputs)
    pages = sum(int(output["pages"]) for output in outputs)
    failed = sum(output.get("status") != "completed" for output in outputs)
    return {
        "documents": len(outputs),
        "failed_documents": failed,
        "pages": pages,
        "pages_per_s": round(pages / max(result.elapsed_s, 1e-9), 4),
    }


__all__ = ["MinerUScaleBench"]
