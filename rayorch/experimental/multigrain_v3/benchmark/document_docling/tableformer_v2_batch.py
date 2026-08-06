"""Version-pinned TableFormerV2 batch adapter for the Docling V3 workload.

The adapter calls the upstream TableFormerV2 model directly.  V3 owns only
TableJob batching, per-sequence EOS masking and lineage-preserving scatter; it
does not monkeypatch model methods or replay private encoder/decoder outputs.
"""

from __future__ import annotations

import copy
from typing import Any

from .core_values import DoclingTableV2Job


PreparedTextCell = tuple[float, float, float, float, float, Any, str]


def prepare_text_cells(
    cells: Any,
    *,
    strip_text: bool,
) -> tuple[PreparedTextCell, ...]:
    """Convert OCR rectangles once instead of once per predicted table cell."""

    prepared = []
    for cell in cells:
        bbox = cell.rect.to_bounding_box()
        left, right = sorted((float(bbox.l), float(bbox.r)))
        top, bottom = sorted((float(bbox.t), float(bbox.b)))
        area = (right - left) * (bottom - top)
        text = cell.text.strip() if strip_text else cell.text
        prepared.append(
            (left, top, right, bottom, area, bbox.coord_origin, text)
        )
    return tuple(prepared)


def match_prepared_text(
    bbox: Any,
    cells: tuple[PreparedTextCell, ...],
    *,
    overlap: float,
) -> str:
    """Match Docling text cells with the same intersection-over-self rule."""

    left, right = sorted((float(bbox.l), float(bbox.r)))
    top, bottom = sorted((float(bbox.t), float(bbox.b)))
    matches = []
    for cell_left, cell_top, cell_right, cell_bottom, area, origin, text in cells:
        if origin != bbox.coord_origin:
            raise ValueError("BoundingBoxes have different CoordOrigin")
        intersection_width = min(right, cell_right) - max(left, cell_left)
        intersection_height = min(bottom, cell_bottom) - max(top, cell_top)
        if (
            intersection_width > 0
            and intersection_height > 0
            and area > 0
            and intersection_width * intersection_height / area > overlap
        ):
            matches.append(text)
    return " ".join(matches)


def trim_token_ids_at_eos(token_ids: Any, eos_token_id: int) -> Any:
    """Return one sequence through its first EOS, ignoring padded EOS tokens."""

    positions = (token_ids == eos_token_id).nonzero(as_tuple=False)
    if positions.numel() == 0:
        return token_ids
    return token_ids[: int(positions[0].item()) + 1]


def generate_tableformer_v2_batch(
    model: Any,
    images: Any,
    tokenizer: Any,
    *,
    max_length: int = 512,
) -> tuple[Any, tuple[Any, ...]]:
    """Run TableFormerV2 autoregressive generation with per-sample EOS fencing.

    Upstream ``TableFormerV2.generate`` accepts a batch tensor but stops only
    when every row emits EOS in the same decoding step.  This adapter keeps
    finished rows on EOS while unfinished rows continue.  All neural work is
    delegated to the upstream ``encode_images``/``forward`` functions.
    """

    import torch

    if images.ndim != 4 or images.size(0) < 1:
        raise ValueError("TableFormerV2 images must have shape (B, C, H, W)")
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("TableFormerV2 tokenizer must define BOS and EOS")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    with torch.no_grad():
        encoder_outputs = model.encode_images(images)
        batch_size = images.size(0)
        generated_ids = torch.full(
            (batch_size, 1),
            tokenizer.bos_token_id,
            dtype=torch.long,
            device=images.device,
        )
        current_input = generated_ids
        past_key_values = None
        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=images.device,
        )

        for _ in range(max_length):
            outputs = model.forward(
                input_ids=current_input,
                attention_mask=None,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            if outputs.logits is None:
                raise RuntimeError("TableFormerV2 forward returned no logits")
            next_token = outputs.logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, tokenizer.eos_token_id),
                next_token,
            )
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            finished |= next_token.squeeze(1) == tokenizer.eos_token_id
            past_key_values = outputs.past_key_values
            current_input = next_token
            if bool(torch.all(finished)):
                break

        final_outputs = model.forward(
            input_ids=generated_ids,
            attention_mask=torch.ones_like(generated_ids),
            encoder_outputs=encoder_outputs,
            past_key_values=None,
            use_cache=False,
            return_dict=True,
        )
        flat_bboxes = final_outputs.predicted_bboxes
        if flat_bboxes is None:
            flat_bboxes = images.new_empty((0, 4))

        bbox_counts = []
        for row in generated_ids:
            trimmed = trim_token_ids_at_eos(row, tokenizer.eos_token_id)
            bbox_counts.append(
                sum(
                    int(token) in model.data_cells
                    for token in trimmed.tolist()
                )
            )
        if sum(bbox_counts) != len(flat_bboxes):
            raise RuntimeError(
                "TableFormerV2 bbox count is not aligned with generated cells"
            )
        split_bboxes = tuple(torch.split(flat_bboxes, bbox_counts, dim=0))
    return generated_ids, split_bboxes


class DoclingTableFormerV2BatchCore:
    """Map UDF implementing ``TableV2Job -> Table`` with real model batching."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        num_threads: int = 4,
        table_batch_mode: str = "v2_batch",
        table_batch_max_jobs: int = 16,
    ) -> None:
        if table_batch_mode != "v2_batch":
            raise ValueError("TableFormerV2 core requires table_batch_mode=v2_batch")
        if table_batch_max_jobs <= 0:
            raise ValueError("table_batch_max_jobs must be positive")

        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.pipeline_options import TableStructureV2Options
        from docling.models.stages.table_structure.table_structure_model_v2 import (
            TableStructureModelV2,
        )

        self.table = TableStructureModelV2(
            enabled=True,
            artifacts_path=None,
            options=TableStructureV2Options(),
            accelerator_options=AcceleratorOptions(
                device=device,
                num_threads=num_threads,
            ),
        )
        self.table_batch_mode = table_batch_mode
        self.table_batch_max_jobs = table_batch_max_jobs
        self.total_table_batch_audit = {
            "jobs": 0,
            "batches": 0,
            "batch_errors": 0,
            "serial_fallback_jobs": 0,
            "generated_tokens": 0,
            "predicted_cells": 0,
            "max_generated_tokens": 0,
            "max_predicted_cells": 0,
            "max_length_jobs": 0,
        }

    def run(self, jobs: list[DoclingTableV2Job]) -> list[Any]:
        """Batch table crops and restore one Docling Table per input job."""

        if not jobs:
            return []
        import torch
        from PIL import Image

        outputs = []
        self.total_table_batch_audit["jobs"] += len(jobs)
        try:
            for offset in range(0, len(jobs), self.table_batch_max_jobs):
                chunk = jobs[offset : offset + self.table_batch_max_jobs]
                images = torch.cat(
                    [
                        self.table.transform(
                            Image.fromarray(job.table_image).convert("RGB")
                        ).unsqueeze(0)
                        for job in chunk
                    ],
                    dim=0,
                ).to(self.table.device)
                generated_ids, batch_bboxes = generate_tableformer_v2_batch(
                    self.table.model,
                    images,
                    self.table.tokenizer,
                )
                generated_lengths = [
                    len(
                        trim_token_ids_at_eos(
                            row,
                            self.table.tokenizer.eos_token_id,
                        )
                    )
                    for row in generated_ids
                ]
                predicted_cells = [len(value) for value in batch_bboxes]
                self.total_table_batch_audit["generated_tokens"] += sum(
                    generated_lengths
                )
                self.total_table_batch_audit["predicted_cells"] += sum(
                    predicted_cells
                )
                self.total_table_batch_audit["max_generated_tokens"] = max(
                    self.total_table_batch_audit["max_generated_tokens"],
                    *generated_lengths,
                )
                self.total_table_batch_audit["max_predicted_cells"] = max(
                    self.total_table_batch_audit["max_predicted_cells"],
                    *predicted_cells,
                )
                self.total_table_batch_audit["max_length_jobs"] += sum(
                    length == 513 for length in generated_lengths
                )
                outputs.extend(
                    self._build_table(job, generated_ids[index], batch_bboxes[index])
                    for index, job in enumerate(chunk)
                )
                self.total_table_batch_audit["batches"] += 1
        except Exception:
            self.total_table_batch_audit["batch_errors"] += 1
            raise
        return outputs

    def batch_audit(self) -> dict[str, int]:
        return dict(self.total_table_batch_audit)

    def _build_table(
        self,
        job: DoclingTableV2Job,
        token_ids: Any,
        predicted_bboxes: Any,
    ) -> Any:
        """Reuse Docling V2 decoding/cell construction after batched inference."""

        import torch
        from docling.datamodel.base_models import Table
        from docling_core.types.doc import BoundingBox, TableCell

        token_ids = trim_token_ids_at_eos(
            token_ids,
            self.table.tokenizer.eos_token_id,
        )
        otsl_seq = self.table._decode_otsl_sequence(token_ids)
        if predicted_bboxes.numel():
            valid = predicted_bboxes.sum(dim=-1) > 0
            predicted_bboxes = predicted_bboxes[valid]
        else:
            predicted_bboxes = torch.empty(0, 4)
        cell_data, num_rows, num_cols = self.table._build_table_cells(
            otsl_seq,
            predicted_bboxes,
            list(job.table_bbox),
        )
        table_cells = []
        cluster = job.table_cluster
        cluster_cells = prepare_text_cells(
            cluster.cells,
            strip_text=True,
        )
        fallback_cells = prepare_text_cells(
            job.text_cells,
            strip_text=False,
        )
        for element in cell_data:
            if element["bbox"] is not None:
                bbox = BoundingBox.model_validate(element["bbox"])
                text = match_prepared_text(
                    bbox,
                    cluster_cells,
                    overlap=0.3,
                )
                if not text.strip():
                    text = match_prepared_text(
                        bbox,
                        fallback_cells,
                        overlap=0.5,
                    )
                element["bbox"]["token"] = text
            table_cells.append(TableCell.model_validate(element))
        return Table(
            otsl_seq=otsl_seq,
            table_cells=table_cells,
            num_rows=num_rows,
            num_cols=num_cols,
            id=cluster.id,
            page_no=job.page_no,
            cluster=copy.deepcopy(cluster),
            label=cluster.label,
        )
