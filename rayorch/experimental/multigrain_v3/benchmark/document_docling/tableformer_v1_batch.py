"""Explicit TableFormer V1 batch kernel for the Docling V3 workload.

The upstream V1 predictor exposes only singleton inference.  This module keeps
that integration behind one version-pinned port: image preparation, neural
batch inference, and deterministic matching/postprocess are separate steps.
No upstream object is monkeypatched and no prediction replay is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core_values import DoclingTableJob
from .table_decoder_batch import predict_tableformer_batch


SUPPORTED_DOCLING_IBM_MODELS_VERSION = "3.13.3"


def _validate_upstream_contract(model: Any) -> None:
    """Fail fast instead of silently drifting with another private V1 layout."""

    from importlib.metadata import version

    installed = version("docling-ibm-models")
    if installed != SUPPORTED_DOCLING_IBM_MODELS_VERSION:
        raise RuntimeError(
            "TableFormer V1 batch port supports docling-ibm-models "
            f"{SUPPORTED_DOCLING_IBM_MODELS_VERSION}, found {installed}"
        )
    required = (
        "_encoder",
        "_tag_transformer",
        "_bbox_decoder",
        "_bbox",
        "_init_data",
        "_device",
        "_max_pred_len",
        "mergebboxes",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise RuntimeError(
            "unsupported TableFormer V1 model contract; missing "
            + ", ".join(missing)
        )


@dataclass(frozen=True, slots=True)
class TableFormerV1Prediction:
    """One row emitted by the batched V1 neural kernel."""

    tag_sequence: Any
    classes: Any
    coordinates: Any


class TableFormerV1ImagePreprocessor:
    """Stable callable equivalent of TFPredictor's singleton image transform."""

    def __init__(self, config: dict[str, Any], device: str) -> None:
        from docling_ibm_models.tableformer.data_management import transforms

        normalization = config["dataset"]["image_normalization"]
        self.normalize = transforms.Normalize(
            mean=normalization["mean"],
            std=normalization["std"],
        )
        resized_image = config["dataset"]["resized_image"]
        self.resize = transforms.Resize([resized_image, resized_image])
        self.device = device

    def __call__(self, image: Any) -> Any:
        """Return one ``[1,C,W,H]`` tensor without touching predictor state."""

        import torch

        transformed, _ = self.normalize(image, None)
        transformed, _ = self.resize(transformed, None)
        transformed = transformed.transpose(2, 1, 0)
        return torch.FloatTensor(transformed / 255.0).unsqueeze(0).to(
            device=self.device
        )


class TableFormerV1Postprocessor:
    """Convert one explicit neural prediction through Docling V1 matching.

    Docling IBM Models 3.13.3 does not expose this half of ``TFPredictor`` as a
    public function.  The required deterministic helpers are resolved once at
    construction and kept inside this port instead of being spread through the
    orchestration UDF.
    """

    _REQUIRED_HELPERS = (
        "_cell_matcher",
        "_post_processor",
        "_check_bbox_sync",
        "_generate_tf_response",
        "_merge_tf_output",
        "_get_html_tags",
    )

    def __init__(self, predictor: Any) -> None:
        missing = [
            name for name in self._REQUIRED_HELPERS if not hasattr(predictor, name)
        ]
        if missing:
            raise RuntimeError(
                "unsupported TableFormer V1 predictor contract; missing "
                + ", ".join(missing)
            )
        self.predictor = predictor

    def run(
        self,
        job: DoclingTableJob,
        prediction: TableFormerV1Prediction,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run matching/postprocess without re-running image preparation."""

        import torch
        from docling_ibm_models.tableformer.otsl import otsl_to_html
        from docling_ibm_models.tableformer.utils import utils as table_utils
        from docling_ibm_models.tableformer.data_management.tf_predictor import (
            otsl_sqr_chk,
        )

        predictor = self.predictor
        coordinates = prediction.coordinates
        if coordinates is None or len(coordinates) == 0:
            bboxes = []
        else:
            bboxes = table_utils.box_cxcywh_to_xyxy(coordinates).tolist()

        classes = prediction.classes
        if classes is None or len(classes) == 0:
            class_ids = []
        else:
            class_ids = torch.argmax(classes, dim=1).tolist()

        tag_sequence = prediction.tag_sequence
        if getattr(predictor, "_remove_padding", False):
            tag_sequence, _ = table_utils.remove_padding(tag_sequence)
        model_prediction = {
            "bboxes": bboxes,
            "classes": class_ids,
            "tag_seq": tag_sequence,
        }
        model_prediction["rs_seq"] = predictor._get_html_tags(tag_sequence)
        model_prediction["html_seq"] = otsl_to_html(
            model_prediction["rs_seq"],
            False,
        )
        otsl_sqr_chk(model_prediction["rs_seq"], False)

        synchronized, corrected_bboxes = predictor._check_bbox_sync(
            model_prediction
        )
        if not synchronized:
            model_prediction["bboxes"] = corrected_bboxes

        matching_details: dict[str, Any] = {
            "table_cells": [],
            "matches": {},
            "pdf_cells": [],
            "prediction_bboxes_page": [],
        }
        scaled_table_bbox = [
            coordinate / job.scale_factor for coordinate in job.table_bbox
        ]
        if model_prediction["bboxes"]:
            matching_details = predictor._cell_matcher.match_cells(
                job.iocr_page,
                scaled_table_bbox,
                model_prediction,
            )
            if (
                job.iocr_page["tokens"]
                and getattr(predictor, "enable_post_process", False)
            ):
                matching_details = predictor._post_processor.process(
                    matching_details,
                    False,
                )

        docling_output = predictor._generate_tf_response(
            matching_details["table_cells"],
            matching_details["matches"],
        )
        docling_output.sort(key=lambda item: item["cell_id"])
        matching_details["docling_responses"] = docling_output
        table_output = predictor._merge_tf_output(
            docling_output,
            matching_details["pdf_cells"],
        )
        return table_output, matching_details


class DoclingTableFormerV1BatchCore:
    """Map UDF implementing ``TableJob -> Table`` with a mutation-free V1 batch."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        num_threads: int = 4,
        table_batch_mode: str = "v1_batch",
        table_batch_max_jobs: int = 16,
    ) -> None:
        if table_batch_mode != "v1_batch":
            raise ValueError("TableFormerV1 batch core requires v1_batch mode")
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
        predictor = self.table.tf_predictor
        self.model = predictor.get_model()
        _validate_upstream_contract(self.model)
        self.preprocess = TableFormerV1ImagePreprocessor(
            self.table.tm_config,
            predictor.get_device(),
        )
        self.postprocess = TableFormerV1Postprocessor(predictor)
        self.table_batch_max_jobs = table_batch_max_jobs
        self.total_table_batch_audit = {
            "jobs": 0,
            "batches": 0,
            "batch_errors": 0,
            "serial_fallback_jobs": 0,
        }

    def run(self, jobs: list[DoclingTableJob]) -> list[Any]:
        """Batch neural inference and preserve one output row per input job."""

        if not jobs:
            return []
        import torch

        outputs = []
        self.total_table_batch_audit["jobs"] += len(jobs)
        try:
            with torch.no_grad():
                for offset in range(0, len(jobs), self.table_batch_max_jobs):
                    chunk = jobs[offset : offset + self.table_batch_max_jobs]
                    image_batch = torch.cat(
                        [self.preprocess(job.table_image) for job in chunk],
                        dim=0,
                    )
                    raw_predictions = predict_tableformer_batch(
                        self.model,
                        image_batch,
                    )
                    if len(raw_predictions) != len(chunk):
                        raise RuntimeError(
                            "TableFormer V1 output count does not match input jobs"
                        )
                    outputs.extend(
                        self._build_table(
                            job,
                            TableFormerV1Prediction(*raw_prediction),
                        )
                        for job, raw_prediction in zip(chunk, raw_predictions)
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
        job: DoclingTableJob,
        prediction: TableFormerV1Prediction,
    ) -> Any:
        from .core_table_workflow import build_v1_table

        responses, details = self.postprocess.run(job, prediction)
        return build_v1_table(job, responses, details)
