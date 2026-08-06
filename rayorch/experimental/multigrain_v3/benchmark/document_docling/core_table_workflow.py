"""TableJob-first Docling V3 workflow stages.

Docling is used only for its core postprocess, TableFormer and PageAssemble
models.  V3 owns Page->TableJob fan-out, cross-document batching and both
lineage reductions.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

from .core_stages import _conversion_result, _group_indices_by_document
from .core_values import (
    DoclingOcrResult,
    DoclingPageAssembly,
    DoclingPageSource,
    DoclingPostprocessedPage,
    DoclingTableJob,
    DoclingTableV2Job,
    PdfRenderCache,
    page_from_source,
)


class DoclingPostprocessPages:
    """Run only Docling's layout postprocessor and return page values."""

    def __init__(self) -> None:
        from docling.datamodel.pipeline_options import (
            LayoutOptions,
            LayoutPostprocessorOptions,
        )
        from docling.models.stages.layout.layout_postprocessing_model import (
            LayoutPostprocessingModel,
        )

        layout_options = LayoutOptions()
        self.postprocess = LayoutPostprocessingModel(
            options=LayoutPostprocessorOptions(
                skip_cell_assignment=layout_options.skip_cell_assignment,
                keep_empty_clusters=layout_options.keep_empty_clusters,
                create_orphan_clusters=layout_options.create_orphan_clusters,
                run_postprocessor=True,
            )
        )

    def run(
        self,
        sources: list[DoclingPageSource],
        layouts: list[Any],
        ocr_results: list[DoclingOcrResult],
    ) -> list[DoclingPostprocessedPage]:
        outputs: list[DoclingPostprocessedPage | None] = [None] * len(sources)
        for path, document_hash, indices in _group_indices_by_document(sources):
            pages = [
                page_from_source(
                    sources[index],
                    layout=layouts[index],
                    ocr=ocr_results[index],
                )
                for index in indices
            ]
            context = _conversion_result(path, document_hash, pages)
            pages = list(self.postprocess(context, pages))
            for index, page in zip(indices, pages):
                outputs[index] = DoclingPostprocessedPage(
                    source=sources[index],
                    layout=copy.deepcopy(page.predictions.layout),
                    ocr=DoclingOcrResult(
                        segmented_page=copy.deepcopy(page.parsed_page)
                    ),
                )
        assert all(output is not None for output in outputs)
        return [output for output in outputs if output is not None]


class ExpandDoclingTableJobs:
    """Expand Page->TableJob without rendering pages that have no tables."""

    def __init__(self, scale: float = 2.0) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = scale
        self.render_cache = PdfRenderCache()
        self.total_audit = {
            "pages": 0,
            "table_pages": 0,
            "skipped_pages": 0,
            "jobs": 0,
        }

    def run(
        self,
        pages: list[DoclingPostprocessedPage],
    ) -> list[list[DoclingTableJob]]:
        outputs: list[list[DoclingTableJob]] = []
        self.total_audit["pages"] += len(pages)
        table_labels = {"table", "document_index"}
        for value in pages:
            clusters = [
                cluster
                for cluster in value.layout.clusters
                if getattr(cluster.label, "value", cluster.label)
                in table_labels
            ]
            if not clusters:
                self.total_audit["skipped_pages"] += 1
                outputs.append([])
                continue

            self.total_audit["table_pages"] += 1
            import cv2
            import numpy
            from docling_core.types.doc import (
                BoundingRectangle,
                TextCellUnit,
            )

            page = page_from_source(
                value.source,
                layout=value.layout,
                ocr=value.ocr,
                render_cache=self.render_cache,
            )
            page_image = numpy.asarray(page.get_image(scale=self.scale))
            height, width = page_image.shape[:2]
            scale_factor = 1024 / float(height)
            resized = cv2.resize(
                page_image,
                (int(width * scale_factor), 1024),
                interpolation=cv2.INTER_AREA,
            )
            segmented_page = page._backend.get_segmented_page()
            jobs: list[DoclingTableJob] = []
            for cluster in clusters:
                tcells = segmented_page.get_cells_in_bbox(
                    cell_unit=TextCellUnit.WORD,
                    bbox=cluster.bbox,
                )
                if not tcells:
                    tcells = cluster.cells
                tokens = []
                for cell in tcells:
                    if not cell.text.strip():
                        continue
                    new_cell = copy.deepcopy(cell)
                    new_cell.rect = BoundingRectangle.from_bounding_box(
                        new_cell.rect.to_bounding_box().scaled(scale=self.scale)
                    )
                    tokens.append(
                        {
                            "id": new_cell.index,
                            "text": new_cell.text,
                            "bbox": new_cell.rect.to_bounding_box().model_dump(),
                        }
                    )

                original_box = (
                    round(cluster.bbox.l) * self.scale,
                    round(cluster.bbox.t) * self.scale,
                    round(cluster.bbox.r) * self.scale,
                    round(cluster.bbox.b) * self.scale,
                )
                table_bbox = tuple(
                    coordinate * scale_factor for coordinate in original_box
                )
                table_image = resized[
                    round(table_bbox[1]) : round(table_bbox[3]),
                    round(table_bbox[0]) : round(table_bbox[2]),
                ]
                if table_image.size == 0:
                    raise ValueError("TableFormer crop is empty")
                jobs.append(
                    DoclingTableJob(
                        page_no=value.source.page_no,
                        table_cluster=copy.deepcopy(cluster),
                        iocr_page={
                            "width": value.source.size.width * self.scale,
                            "height": value.source.size.height * self.scale,
                            "tokens": tokens,
                        },
                        table_bbox=table_bbox,
                        table_image=numpy.ascontiguousarray(table_image),
                        scale_factor=scale_factor,
                        scale=self.scale,
                    )
                )
            outputs.append(jobs)
            self.total_audit["jobs"] += len(jobs)
        return outputs

    def batch_audit(self) -> dict[str, int]:
        return dict(self.total_audit)


class ExpandDoclingTableV2Jobs:
    """Expand Page -> TableV2Job using Docling V2's native 2x crop contract."""

    def __init__(self, scale: float = 2.0) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = scale
        self.render_cache = PdfRenderCache()
        self.total_audit = {
            "pages": 0,
            "table_pages": 0,
            "skipped_pages": 0,
            "jobs": 0,
        }

    def run(
        self,
        pages: list[DoclingPostprocessedPage],
    ) -> list[list[DoclingTableV2Job]]:
        """Crop each table directly from the 2x page image without V1 resize."""

        outputs: list[list[DoclingTableV2Job]] = []
        self.total_audit["pages"] += len(pages)
        table_labels = {"table", "document_index"}
        for value in pages:
            clusters = [
                cluster
                for cluster in value.layout.clusters
                if getattr(cluster.label, "value", cluster.label)
                in table_labels
            ]
            if not clusters:
                self.total_audit["skipped_pages"] += 1
                outputs.append([])
                continue

            import numpy

            self.total_audit["table_pages"] += 1
            page = page_from_source(
                value.source,
                layout=value.layout,
                ocr=value.ocr,
                render_cache=self.render_cache,
            )
            page_image = numpy.asarray(page.get_image(scale=self.scale))
            textline_cells = tuple(
                copy.deepcopy(value.ocr.segmented_page.textline_cells)
            )
            jobs = []
            for cluster in clusters:
                scaled_box = (
                    round(cluster.bbox.l) * self.scale,
                    round(cluster.bbox.t) * self.scale,
                    round(cluster.bbox.r) * self.scale,
                    round(cluster.bbox.b) * self.scale,
                )
                x1, y1, x2, y2 = [int(coordinate) for coordinate in scaled_box]
                table_image = page_image[y1:y2, x1:x2]
                if table_image.size == 0:
                    raise ValueError("TableFormerV2 crop is empty")
                table_bbox = tuple(
                    coordinate / self.scale for coordinate in scaled_box
                )
                nearby_cells = tuple(
                    cell
                    for cell in textline_cells
                    if cell.rect.to_bounding_box().get_intersection_bbox(
                        cluster.bbox
                    )
                    is not None
                )
                jobs.append(
                    DoclingTableV2Job(
                        page_no=value.source.page_no,
                        table_cluster=copy.deepcopy(cluster),
                        table_bbox=table_bbox,
                        table_image=numpy.ascontiguousarray(table_image),
                        text_cells=nearby_cells,
                    )
                )
            outputs.append(jobs)
            self.total_audit["jobs"] += len(jobs)
        return outputs

    def batch_audit(self) -> dict[str, int]:
        return dict(self.total_audit)


def normalize_v1_table_grid(
    responses: list[dict[str, Any]],
    details: dict[str, Any],
) -> None:
    """Normalize sparse V1 row/column offsets into a dense Docling grid."""

    start_cols = sorted({cell["start_col_offset_idx"] for cell in responses})
    start_rows = sorted({cell["start_row_offset_idx"] for cell in responses})
    max_end_col = 0
    max_end_row = 0
    for cell in responses:
        cell["start_col_offset_idx"] = start_cols.index(
            cell["start_col_offset_idx"]
        )
        cell["end_col_offset_idx"] = (
            cell["start_col_offset_idx"] + cell["col_span"]
        )
        max_end_col = max(max_end_col, cell["end_col_offset_idx"])
        cell["start_row_offset_idx"] = start_rows.index(
            cell["start_row_offset_idx"]
        )
        cell["end_row_offset_idx"] = (
            cell["start_row_offset_idx"] + cell["row_span"]
        )
        max_end_row = max(max_end_row, cell["end_row_offset_idx"])
    details["num_cols"] = max_end_col
    details["num_rows"] = max_end_row


def build_v1_table(
    job: DoclingTableJob,
    responses: list[dict[str, Any]],
    details: dict[str, Any],
) -> Any:
    """Build one Docling Table from explicit V1 postprocess outputs."""

    from docling.datamodel.base_models import Table
    from docling_core.types.doc import TableCell

    normalize_v1_table_grid(responses, details)
    table_cells = []
    for element in responses:
        cell = TableCell.model_validate(element)
        if cell.bbox is not None:
            cell.bbox = cell.bbox.scaled(1 / job.scale)
        table_cells.append(cell)
    cluster = job.table_cluster
    return Table(
        otsl_seq=details.get("prediction", {}).get("rs_seq", []),
        table_cells=table_cells,
        num_rows=details.get("num_rows", 0),
        num_cols=details.get("num_cols", 0),
        id=cluster.id,
        page_no=job.page_no,
        cluster=cluster,
        label=cluster.label,
    )


class DoclingTableCore:
    """Reference and legacy-reproduction TableFormer V1 UDF.

    New accelerated experiments use ``DoclingTableFormerV1BatchCore``.  The
    mutation-based modes remain here only so historical artifacts can be
    reproduced; new behavior must not be added to them.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        num_threads: int = 4,
        table_batch_mode: str = "reference",
        table_batch_max_jobs: int = 16,
    ) -> None:
        if table_batch_mode not in {
            "reference",
            "encoder_shadow",
            "encoder_accelerated",
            "decoder_accelerated",
        }:
            raise ValueError("unsupported table_batch_mode")
        if table_batch_max_jobs <= 0:
            raise ValueError("table_batch_max_jobs must be positive")

        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.pipeline_options import TableStructureOptions
        from docling.models.stages.table_structure.table_structure_model import (
            TableStructureModel,
        )

        self.table = TableStructureModel(
            enabled=True,
            artifacts_path=None,
            options=TableStructureOptions(),
            accelerator_options=AcceleratorOptions(
                device=device,
                num_threads=num_threads,
            ),
        )
        self.table_batch_mode = table_batch_mode
        self.table_batch_max_jobs = table_batch_max_jobs
        self.total_table_batch_audit: dict[str, int] = {
            "jobs": 0,
            "batches": 0,
            "batch_errors": 0,
            "serial_fallback_jobs": 0,
        }

    def run(self, jobs: list[DoclingTableJob]) -> list[Any]:
        if not jobs:
            return []
        self.total_table_batch_audit["jobs"] += len(jobs)
        if self.table_batch_mode in {"reference", "encoder_shadow"}:
            self.total_table_batch_audit["batches"] += len(jobs)
            return [self._predict_one(job) for job in jobs]
        try:
            if self.table_batch_mode == "decoder_accelerated":
                from .table_decoder_adapter_v2 import run_decoder_batched

                return run_decoder_batched(self, jobs)
            return self._run_encoder_batched(jobs)
        except Exception:
            self.total_table_batch_audit["batch_errors"] += 1
            if self.table_batch_mode == "decoder_accelerated":
                raise
            self.total_table_batch_audit["serial_fallback_jobs"] += len(jobs)
            return [self._predict_one(job) for job in jobs]

    def batch_audit(self) -> dict[str, int]:
        return dict(self.total_table_batch_audit)

    def _run_encoder_batched(self, jobs: list[DoclingTableJob]) -> list[Any]:
        import torch

        predictor = self.table.tf_predictor
        model = predictor.get_model()
        outputs = []
        batch_count = 0
        with torch.no_grad():
            for offset in range(0, len(jobs), self.table_batch_max_jobs):
                chunk = jobs[offset : offset + self.table_batch_max_jobs]
                tensors = [
                    predictor._prepare_image(job.table_image) for job in chunk
                ]
                encoded = model._encoder(torch.cat(tensors, dim=0))
                outputs.extend(
                    encoded[index : index + 1]
                    for index in range(len(chunk))
                )
                batch_count += 1
        self.total_table_batch_audit["batches"] += batch_count

        pending = deque(outputs)
        original_encoder = model._encoder

        class _EncoderReplay:
            def __call__(self, _: Any) -> Any:
                if not pending:
                    raise RuntimeError("TableFormer encoder replay exhausted")
                return pending.popleft()

        object.__setattr__(model, "_encoder", _EncoderReplay())
        try:
            result = [self._predict_one(job) for job in jobs]
            if pending:
                raise RuntimeError("TableFormer encoder outputs were not consumed")
            return result
        finally:
            object.__setattr__(model, "_encoder", original_encoder)

    def _predict_one(
        self,
        job: DoclingTableJob,
        eval_res_preds: dict[str, Any] | None = None,
    ) -> Any:
        tf_responses, predict_details = self.table.tf_predictor.predict(
            job.iocr_page,
            list(job.table_bbox),
            job.table_image,
            job.scale_factor,
            eval_res_preds,
            False,
        )
        return build_v1_table(job, tf_responses, predict_details)


class ReduceDoclingPage:
    """Reduce Table children back to Page, including zero-table groups."""

    def __init__(self) -> None:
        from docling.models.stages.page_assemble.page_assemble_model import (
            PageAssembleModel,
            PageAssembleOptions,
        )

        self.assemble = PageAssembleModel(PageAssembleOptions())

    def run(
        self,
        grouped_tables: list[list[Any]],
        pages: list[DoclingPostprocessedPage],
    ) -> list[DoclingPageAssembly]:
        from docling.datamodel.base_models import TableStructurePrediction

        if len(grouped_tables) != len(pages):
            raise ValueError("table groups and page anchors are not aligned")
        temporary = []
        for tables, value in zip(grouped_tables, pages):
            page = page_from_source(
                value.source,
                layout=value.layout,
                ocr=value.ocr,
            )
            prediction = TableStructurePrediction()
            for table in tables:
                prediction.table_map[table.id] = copy.deepcopy(table)
            page.predictions.tablestructure = prediction
            temporary.append(page)

        assembled_by_index: dict[int, Any] = {}
        sources = [value.source for value in pages]
        for path, document_hash, indices in _group_indices_by_document(sources):
            batch = [temporary[index] for index in indices]
            context = _conversion_result(path, document_hash, batch)
            batch = list(self.assemble(context, batch))
            for index, page in zip(indices, batch):
                assembled_by_index[index] = page
        return [
            DoclingPageAssembly(
                document_path=value.source.document_path,
                document_hash=value.source.document_hash,
                page_no=value.source.page_no,
                size=value.source.size,
                assembled=copy.deepcopy(assembled_by_index[index].assembled),
            )
            for index, value in enumerate(pages)
        ]
