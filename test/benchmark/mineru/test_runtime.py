"""Ray-backed regression for the released MinerU pipeline topology."""

from __future__ import annotations

from pathlib import Path

from rayorch import Executor, RayModule
from rayorch.benchmark.mineru.pipeline import MinerUPipeline


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
            microbatch_size=2,
            max_active_microbatches=1,
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
    assert ocr.grains == 5
    assert sum(ocr.batch_sizes) == 5
