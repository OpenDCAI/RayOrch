"""Regression coverage for the promoted historical workload shapes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rayorch import Pipeline
from rayorch._program.logical import ExpandOrigin, ReduceOrigin
from rayorch.benchmark import (
    DocumentTopologyBench,
    VideoCaptionTopologyBench,
    VideoMultimodalTopologyBench,
)
from rayorch.benchmarks.document_topology.pipeline import (
    DocumentTopologyPipeline,
)
from rayorch.benchmarks.document_topology.udfs import (
    ExpandTableJobs,
    LayoutPages,
    OcrPages,
    ParseDocuments,
    PostprocessPages,
    ReduceDocument,
    ReducePage,
    TableCore,
)
from rayorch.benchmarks.video_caption.pipeline import (
    VideoCaptionTopologyPipeline,
)
from rayorch.benchmarks.video_caption.udfs import (
    CaptionFrames,
    DecodeFrames,
    SummarizeCaptions,
)
from rayorch.benchmarks.video_multimodal.pipeline import (
    VideoMultimodalTopologyPipeline,
)
from rayorch.benchmarks.video_multimodal.udfs import (
    DecodeAudio,
    DecodeFrames as DecodeMultimodalFrames,
    MergeModalities,
    ProcessFrames,
    SummarizeAudio,
    SummarizeFrames,
    TranscribeAudio,
)


@pytest.fixture(scope="module", autouse=True)
def ray_runtime():
    import ray

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(
            address=os.environ.get("RAY_ADDRESS", "local"),
            include_dashboard=False,
            num_cpus=8,
            num_gpus=0,
            object_store_memory=1024**3,
        )
    yield
    if started_here:
        ray.shutdown()


def _assert_topology(
    pipeline: Pipeline,
    *,
    targets: list[type],
    domains: int,
    expansions: int,
    reductions: int,
) -> None:
    compiled = pipeline.compile()

    assert [spec.udf.target for spec in compiled.logical.calls.values()] == targets
    assert len(compiled.logical.domains) == domains
    assert len(compiled.plan.actor_pools) == len(targets)
    assert (
        sum(
            isinstance(spec.origin, ExpandOrigin)
            for spec in compiled.logical.ports.values()
        )
        == expansions
    )
    assert (
        sum(
            isinstance(spec.origin, ReduceOrigin)
            for spec in compiled.logical.ports.values()
        )
        == reductions
    )


def test_document_topology_is_a_runnable_benchmark(tmp_path: Path):
    _assert_topology(
        DocumentTopologyPipeline(),
        targets=[
            ParseDocuments,
            LayoutPages,
            OcrPages,
            PostprocessPages,
            ExpandTableJobs,
            TableCore,
            ReducePage,
            ReduceDocument,
        ],
        domains=3,
        expansions=2,
        reductions=2,
    )

    report = DocumentTopologyBench(
        output_dir=tmp_path,
        document_count=2,
        pages_per_document=3,
        tables_per_page=2,
    ).run(profile=False, run_id="document")

    assert report.metrics["pages"] == 6
    assert report.metrics["tables"] == 8
    assert report.metrics["peak_active_input_batches"] == 2
    assert report.outputs[0] == {
        "document": "document-0",
        "pages": (0, 1, 2),
        "tables": ((0, 1), (), (0, 1)),
    }
    assert Path(report.artifacts["summary"]).is_file()


def test_video_caption_topology_is_a_runnable_benchmark(tmp_path: Path):
    _assert_topology(
        VideoCaptionTopologyPipeline(),
        targets=[DecodeFrames, CaptionFrames, SummarizeCaptions],
        domains=2,
        expansions=1,
        reductions=1,
    )

    report = VideoCaptionTopologyBench(
        output_dir=tmp_path,
        video_count=3,
        frames_per_video=4,
    ).run(profile=False, run_id="caption")

    assert report.metrics["frames"] == 12
    assert [output["video"] for output in report.outputs] == [
        "video-0",
        "video-1",
        "video-2",
    ]
    assert all(output["frames"] == (0, 1, 2, 3) for output in report.outputs)


def test_video_multimodal_topology_is_a_runnable_benchmark(tmp_path: Path):
    _assert_topology(
        VideoMultimodalTopologyPipeline(),
        targets=[
            DecodeAudio,
            TranscribeAudio,
            SummarizeAudio,
            DecodeMultimodalFrames,
            ProcessFrames,
            SummarizeFrames,
            MergeModalities,
        ],
        domains=3,
        expansions=2,
        reductions=2,
    )

    report = VideoMultimodalTopologyBench(
        output_dir=tmp_path,
        video_count=2,
        frames_per_video=4,
        audio_chunks_per_video=3,
    ).run(profile=False, run_id="multimodal")

    assert report.metrics["frames"] == 8
    assert report.metrics["audio_chunks"] == 6
    assert report.outputs == [
        {
            "video": "clip-0",
            "audio": (0, 1, 2),
            "vision": (0, 1, 2, 3),
        },
        {
            "video": "clip-1",
            "audio": (0, 1, 2),
            "vision": (0, 1, 2, 3),
        },
    ]
