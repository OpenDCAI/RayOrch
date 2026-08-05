"""Connect active-mask decoder batching to Docling's predictor postprocess."""

from __future__ import annotations

from typing import Any

from .table_decoder_batch import predict_tableformer_batch


def run_decoder_batched(table_core: Any, jobs: list[Any]) -> list[Any]:
    """Run core encoder+decoder batches, then reuse TFPredictor matching."""

    import torch
    import docling_ibm_models.tableformer.utils.utils as table_utils

    predictor = table_core.table.tf_predictor
    model = predictor.get_model()
    outputs = []
    batch_count = 0
    with torch.no_grad():
        for offset in range(0, len(jobs), table_core.table_batch_max_jobs):
            chunk = jobs[offset : offset + table_core.table_batch_max_jobs]
            image_batch = torch.cat(
                [predictor._prepare_image(job.table_image) for job in chunk],
                dim=0,
            )
            predictions = predict_tableformer_batch(model, image_batch)
            for job, (tag_sequence, _, coordinates) in zip(chunk, predictions):
                if coordinates is None or len(coordinates) == 0:
                    bboxes = []
                else:
                    bboxes = table_utils.box_cxcywh_to_xyxy(
                        coordinates
                    ).tolist()
                outputs.append(
                    table_core._predict_one(
                        job,
                        eval_res_preds={
                            "bboxes": bboxes,
                            "tag_seq": tag_sequence,
                        },
                    )
                )
            batch_count += 1
    table_core.total_table_batch_audit["batches"] += batch_count
    return outputs
