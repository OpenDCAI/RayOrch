"""Ray-backed regressions for the MinerU graph and public Benchmark API."""

from __future__ import annotations

import json
from pathlib import Path

from rayorch import Executor, RayModule
from rayorch.benchmark import MinerUBench
from rayorch.benchmarks.mineru import benchmark as mineru
from rayorch.benchmarks.mineru.pipeline import MinerUPipeline


class _Render:
    def run(self, documents):
        return [
            [
                {
                    "pdf_path": document,
                    "page_id": page_id,
                    "pdf_len": page_count,
                }
                for page_id in range(page_count)
            ]
            for document, page_count in documents
        ]


class _Ocr:
    def run(self, pages):
        return [
            f"{Path(page['pdf_path']).stem}:{page['page_id']}"
            for page in pages
        ]


class _Metadata:
    def run(self, documents):
        return [Path(document).stem for document, _ in documents]


class _Assemble:
    def run(self, grouped_contents, grouped_pages, stems):
        return [
            {
                "pdf": stem,
                "pages": len(pages),
                "contents": tuple(contents),
                "page_ids": tuple(page["page_id"] for page in pages),
            }
            for contents, pages, stem in zip(
                grouped_contents,
                grouped_pages,
                stems,
                strict=True,
            )
        ]


class _SyntheticMinerUPipeline(MinerUPipeline):
    """Reuse the production graph with dependency-free test UDFs."""

    def __init__(self) -> None:
        self.render = RayModule(_Render).ray_options(
            replicas=2,
            batch_size=1,
            num_cpus=0,
        )
        self.ocr = RayModule(_Ocr).ray_options(
            replicas=2,
            batch_size=3,
            num_cpus=0,
        )
        self.metadata = RayModule(_Metadata).ray_options(
            replicas=1,
            batch_size=8,
            num_cpus=0,
        )
        self.assemble = RayModule(_Assemble).ray_options(
            replicas=1,
            batch_size=4,
            num_cpus=0,
        )


def test_mineru_topology_runs_end_to_end_on_ray():
    with Executor(
        _SyntheticMinerUPipeline(),
        ray_init_kwargs={"include_dashboard": False},
    ) as executor:
        result = executor.run(
            [("alpha.pdf", 2), ("beta.pdf", 3)],
            input_batch_size=2,
            max_active_input_batches=1,
        )

    assert result.outputs == [
        {
            "pdf": "alpha",
            "pages": 2,
            "contents": ("alpha:0", "alpha:1"),
            "page_ids": (0, 1),
        },
        {
            "pdf": "beta",
            "pages": 3,
            "contents": ("beta:0", "beta:1", "beta:2"),
            "page_ids": (0, 1, 2),
        },
    ]
    ocr = next(metrics for metrics in result.calls if metrics.udf_name.endswith("_Ocr"))
    assert ocr.grain_dispatches == 5
    assert sum(ocr.batch_sizes) == 5


class _SyntheticMinerUBench(MinerUBench):
    """Exercise the public Benchmark API without optional MinerU dependencies."""

    def _pipeline(self):
        return _SyntheticMinerUPipeline()


def test_public_mineru_benchmark_runs_and_writes_profiled_report(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        mineru,
        "_load_pdfs",
        lambda path, limit: (("alpha.pdf", 2), ("beta.pdf", 3)),
    )
    monkeypatch.setattr(mineru, "_validate", lambda pdfs, model: None)
    benchmark = _SyntheticMinerUBench(
        input_path=tmp_path / "inputs",
        input_limit=None,
        output_dir=tmp_path / "output",
        num_gpus=2,
        batch_size=3,
        input_batch_size=2,
        max_active_input_batches=1,
    )

    report = benchmark.run(
        ray_init_kwargs={"include_dashboard": False, "num_gpus": 0},
        profile=True,
        profile_interval_s=0.01,
        run_id="synthetic",
    )

    assert report.outputs == [
        {
            "pdf": "alpha",
            "pages": 2,
            "contents": ("alpha:0", "alpha:1"),
            "page_ids": (0, 1),
        },
        {
            "pdf": "beta",
            "pages": 3,
            "contents": ("beta:0", "beta:1", "beta:2"),
            "page_ids": (0, 1, 2),
        },
    ]
    assert report.metrics["input_rows"] == 2
    assert report.metrics["output_rows"] == 2
    assert report.metrics["pages"] == 5
    ocr = next(
        call for call in report.metrics["calls"] if call["udf_name"].endswith("_Ocr")
    )
    assert ocr["grain_dispatches"] == 5
    assert report.config["num_gpus"] == 2
    assert report.profile["status"] == "collected"
    summary = (
        tmp_path
        / "output"
        / ".rayorch-benchmark"
        / "synthetic"
        / "summary.json"
    )
    assert Path(report.artifacts["summary"]) == summary
    persisted = json.loads(summary.read_text())
    assert persisted["benchmark"] == "mineru"
    assert persisted["metrics"]["pages"] == 5
