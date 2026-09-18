"""Public Benchmark for the nested document reference workload."""

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

from .pipeline import DocumentTopologyPipeline


@dataclass(frozen=True, slots=True)
class DocumentTopologyBench:
    output_dir: str | Path = "."
    document_count: int = 2
    pages_per_document: int = 3
    tables_per_page: int = 2
    workers: int = 2
    batch_size: int = 4
    input_batch_size: int = 1
    max_active_input_batches: int = 2

    def __post_init__(self) -> None:
        _positive(
            self,
            "document_count",
            "pages_per_document",
            "workers",
            "batch_size",
            "input_batch_size",
            "max_active_input_batches",
        )
        if type(self.tables_per_page) is not int or self.tables_per_page < 0:
            raise ValueError("tables_per_page must be a non-negative integer")
        object.__setattr__(self, "output_dir", str(Path(self.output_dir).resolve()))

    def run(
        self,
        *,
        ray_address: str | None = None,
        ray_init_kwargs: Mapping[str, Any] | None = None,
        profile: bool = True,
        profile_interval_s: float = 1.0,
        run_id: str | None = None,
    ) -> BenchmarkReport:
        documents = tuple(
            (
                f"document-{document}",
                tuple(
                    0 if (document + page) % 3 == 1 else self.tables_per_page
                    for page in range(self.pages_per_document)
                ),
            )
            for document in range(self.document_count)
        )
        return run_benchmark(
            name="document_topology",
            pipeline=DocumentTopologyPipeline(
                workers=self.workers,
                batch_size=self.batch_size,
            ),
            source_columns=(documents,),
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
            benchmark="document_topology",
            config=benchmark_config(self),
            artifact_root=Path(self.output_dir) / ".rayorch-benchmark",
            target=target,
            source=source,
            profile=profile,
            profile_interval_s=profile_interval_s,
            run_id=run_id,
        )


def _positive(value: object, *names: str) -> None:
    for name in names:
        item = getattr(value, name)
        if type(item) is not int or item <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _metrics(result: RunResult) -> dict[str, int]:
    outputs = list(result.outputs)
    return {
        "pages": sum(len(output["pages"]) for output in outputs),
        "tables": sum(
            len(tables)
            for output in outputs
            for tables in output["tables"]
        ),
    }


__all__ = ["DocumentTopologyBench"]
