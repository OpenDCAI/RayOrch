"""Replay batched TableFormer outputs through the native predictor contract."""

from __future__ import annotations

from collections import deque
from typing import Any

from .table_decoder_batch import predict_tableformer_batch


def run_decoder_batched(table_core: Any, jobs: list[Any]) -> list[Any]:
    """Batch core inference while retaining native matching and postprocess."""

    import torch

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
            pending = deque(predict_tableformer_batch(model, image_batch))
            original_predict = model.predict

            class _PredictionReplay:
                def __call__(self, *_: Any, **__: Any) -> Any:
                    if not pending:
                        raise RuntimeError(
                            "TableFormer prediction replay exhausted"
                        )
                    return pending.popleft()

            object.__setattr__(model, "predict", _PredictionReplay())
            try:
                outputs.extend(table_core._predict_one(job) for job in chunk)
                if pending:
                    raise RuntimeError(
                        "TableFormer predictions were not consumed"
                    )
            finally:
                object.__setattr__(model, "predict", original_predict)
            batch_count += 1
    table_core.total_table_batch_audit["batches"] += batch_count
    return outputs
