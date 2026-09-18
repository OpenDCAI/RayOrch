"""YOLO-SAM Pipeline structure."""

from __future__ import annotations

from rayorch.benchmarks.yolo_sam.pipeline import YoloSamPipeline
from rayorch.benchmarks.yolo_sam.udfs import (
    DetectObjects,
    LoadImages,
    RenderMasks,
    SaveImages,
    SaveMetadata,
    SegmentImages,
)


def test_yolo_sam_pipeline_has_explicit_stages_and_resources():
    compiled = YoloSamPipeline(
        output_dir="/tmp/yolo-sam",
        yolo_model="/models/yolo.pt",
        sam_checkpoint="/models/sam.pth",
        yolo_replicas=2,
        sam_replicas=3,
        batch_size=8,
        stage_options={"sam": {"batch_size": 2}},
    ).compile()
    targets = [spec.udf.target for spec in compiled.logical.calls.values()]
    pools = list(compiled.plan.actor_pools_by_call.values())

    assert targets == [
        LoadImages,
        DetectObjects,
        SegmentImages,
        RenderMasks,
        SaveImages,
        SaveMetadata,
    ]
    assert pools[1].replicas == 2
    assert dict(pools[1].ray_options)["num_gpus"] == 1.0
    assert pools[2].replicas == 3
    assert pools[2].batch_size == 2
    assert dict(pools[2].ray_options)["num_gpus"] == 1.0
