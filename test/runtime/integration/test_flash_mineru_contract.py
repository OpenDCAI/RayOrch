from __future__ import annotations

import pytest

from rayorch import (
    BadRecordError,
    DagPipeline,
    RuntimeDagExecutor,
    RuntimeRayModule,
)

pytestmark = pytest.mark.usefixtures("ray_cluster")


class Pdf2ImageOp:
    def run(self, pdf_paths: list[str]) -> list[list[str]]:
        images = []
        for index, path in enumerate(pdf_paths):
            if "bad-pdf" in path:
                raise BadRecordError("PDF decode failed", index=index)
            images.append([f"{path}:page-0", f"{path}:page-1"])
        return images


class ProcessImagesOp:
    def run(self, images: list[list[str]]) -> list[list[str]]:
        results = []
        for pages in images:
            if "bad-model" in pages[0]:
                # A third-party model may raise without identifying the row.
                # Runtime isolates it by recursively splitting this microbatch.
                raise RuntimeError("model inference failed")
            results.append([f"model<{page}>" for page in pages])
        return results


class ConvertToMarkdownOp:
    def run(
        self,
        model_results: list[list[str]],
        images: list[list[str]],
    ) -> list[str]:
        return [
            f"{pages[0].split(':')[0]}.md results={len(results)}"
            for results, pages in zip(model_results, images)
        ]


class FlashMineruContractPipeline(DagPipeline):
    """The same three-stage DAG shape as Flash-MinerU's current pipeline."""

    def __init__(self) -> None:
        self.pdf2img = RuntimeRayModule(Pdf2ImageOp, replicas=2).pre_init()
        self.process_img = RuntimeRayModule(ProcessImagesOp, replicas=2).pre_init()
        self.img2md = RuntimeRayModule(ConvertToMarkdownOp, replicas=2).pre_init()
        super().__init__()

    def forward(self, pdf_paths: list[str]) -> list[str]:
        images = self.pdf2img(pdf_paths)
        model_results = self.process_img(images)
        return self.img2md(model_results, images)


def test_exact_flash_mineru_dag_isolates_errors_and_rejoins_images() -> None:
    pdf_paths = [
        "good-a.pdf",
        "bad-pdf.pdf",
        "bad-model.pdf",
        "good-b.pdf",
    ]

    with RuntimeDagExecutor(
        FlashMineruContractPipeline(),
        batch_size=4,
        dataset="mineru-contract",
    ) as executor:
        result = executor.run(pdf_paths=pdf_paths)[0]

    # Flash-MinerU returns the final module call directly, so the compiler uses
    # the stable fallback name instead of inventing a business-level name.
    assert result.batch.columns["img2md.out0"] == [
        "good-a.pdf.md results=2",
        "good-b.pdf.md results=2",
    ]
    assert result.batch.row_ids == [
        "mineru-contract:0",
        "mineru-contract:3",
    ]

    errors = {record.row_id: record for record in result.quarantined}
    assert set(errors) == {"mineru-contract:1", "mineru-contract:2"}

    pdf_error = errors["mineru-contract:1"]
    assert pdf_error.op == "pdf2img"
    assert result.trace(pdf_error.path_id) == []
    assert pdf_error.values["pdf_paths"] == "bad-pdf.pdf"

    model_error = errors["mineru-contract:2"]
    assert model_error.op == "process_img"
    assert result.trace(model_error.path_id) == ["pdf2img"]
    assert model_error.values["images"][0] == "bad-model.pdf:page-0"

    assert result.trace_row("mineru-contract:0") == [
        "pdf2img",
        "process_img",
        "img2md",
    ]
