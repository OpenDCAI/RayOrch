"""Regression tests for workload shapes recorded before the release cleanup.

The release wheel intentionally omits experiment runners, but the promoted
runtime must still express their production dataflows.  These dependency-free
UDFs preserve the cardinality and stage boundaries of the recorded Docling,
video-caption, and dual-modality video workloads.
"""

from __future__ import annotations

import os

import pytest

from rayorch import Executor, F, Pipeline, Port, RayModule
from rayorch._program.logical import ExpandOrigin, ReduceOrigin


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
    assert len(compiled.plan.actor_pools_by_call) == len(targets)
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


class _ParseDocuments:
    def run(self, documents):
        return [
            [
                {
                    "document": name,
                    "page": page,
                    "table_count": table_counts[page],
                }
                for page in range(len(table_counts))
            ]
            for name, table_counts in documents
        ]


class _LayoutPages:
    def run(self, pages):
        return [
            {"page": page["page"], "layout": f"layout:{page['page']}"}
            for page in pages
        ]


class _OcrPages:
    def run(self, pages, layouts):
        return [
            {
                "page": page["page"],
                "text": f"{page['document']}:{layout['layout']}",
            }
            for page, layout in zip(pages, layouts, strict=True)
        ]


class _PostprocessPages:
    def run(self, pages, layouts, ocr):
        return [
            {
                **page,
                "layout": layout["layout"],
                "text": text["text"],
            }
            for page, layout, text in zip(pages, layouts, ocr, strict=True)
        ]


class _ExpandTableJobs:
    def run(self, pages):
        return [
            [
                {
                    "document": page["document"],
                    "page": page["page"],
                    "table": table,
                }
                for table in range(page["table_count"])
            ]
            for page in pages
        ]


class _TableCore:
    def run(self, jobs):
        return [
            {
                **job,
                "cells": job["table"] + 1,
            }
            for job in jobs
        ]


class _ReducePage:
    def run(self, table_groups, pages):
        return [
            {
                "document": page["document"],
                "page": page["page"],
                "text": page["text"],
                "tables": tuple(table["table"] for table in tables),
            }
            for tables, page in zip(table_groups, pages, strict=True)
        ]


class _ReduceDocument:
    def run(self, page_groups):
        return [
            {
                "document": pages[0]["document"],
                "pages": tuple(page["page"] for page in pages),
                "tables": tuple(page["tables"] for page in pages),
            }
            for pages in page_groups
        ]


class _DoclingTopology(Pipeline):
    """Document -> Page -> TableJob -> Page -> Document."""

    def __init__(self) -> None:
        self.parse = RayModule(_ParseDocuments).ray_options(
            replicas=2, batch_size=1, num_cpus=0
        )
        self.layout = RayModule(_LayoutPages).ray_options(
            replicas=2, batch_size=4, num_cpus=0
        )
        self.ocr = RayModule(_OcrPages).ray_options(
            replicas=2, batch_size=4, num_cpus=0
        )
        self.postprocess = RayModule(_PostprocessPages).ray_options(
            replicas=2, batch_size=4, num_cpus=0
        )
        self.table_prepare = RayModule(_ExpandTableJobs).ray_options(
            replicas=2, batch_size=4, num_cpus=0
        )
        self.table_core = RayModule(_TableCore).ray_options(
            replicas=2, batch_size=4, num_cpus=0
        )
        self.page_assemble = RayModule(_ReducePage).ray_options(
            replicas=2, batch_size=4, num_cpus=0
        )
        self.document_assemble = RayModule(_ReduceDocument).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )

    def forward(self, documents: Port) -> Port:
        pages = F.expand(self.parse(documents))
        layouts = self.layout(pages)
        ocr = self.ocr(pages, layouts)
        postprocessed = self.postprocess(pages, layouts, ocr)

        table_jobs = F.expand(self.table_prepare(postprocessed))
        tables = self.table_core(table_jobs)
        table_groups = F.reduce(tables)
        assembled_pages = self.page_assemble(table_groups, postprocessed)

        page_groups = F.reduce(assembled_pages)
        return self.document_assemble(page_groups)


def test_recorded_docling_stages_compile_and_run_with_nested_empty_groups():
    pipeline = _DoclingTopology()
    _assert_topology(
        pipeline,
        targets=[
            _ParseDocuments,
            _LayoutPages,
            _OcrPages,
            _PostprocessPages,
            _ExpandTableJobs,
            _TableCore,
            _ReducePage,
            _ReduceDocument,
        ],
        domains=3,
        expansions=2,
        reductions=2,
    )

    with Executor(pipeline) as executor:
        result = executor.run(
            [
                ("paper-a", (2, 0, 1)),
                ("paper-b", (0, 3)),
            ],
            input_batch_size=1,
            max_active_input_batches=2,
        )

    assert result.outputs == [
        {
            "document": "paper-a",
            "pages": (0, 1, 2),
            "tables": ((0, 1), (), (0,)),
        },
        {
            "document": "paper-b",
            "pages": (0, 1),
            "tables": ((), (0, 1, 2)),
        },
    ]
    assert result.peak_active_input_batches == 2
    assert all(call.grain_requeues == 0 for call in result.calls)


class _DecodeFrames:
    def run(self, videos):
        return [
            [
                {"video": name, "frame": frame}
                for frame in range(frame_count)
            ]
            for name, frame_count, _audio_chunks in videos
        ]


class _CaptionFrames:
    def run(self, frames):
        return [
            {
                **frame,
                "caption": f"{frame['video']}:caption:{frame['frame']}",
            }
            for frame in frames
        ]


class _SummarizeCaptions:
    def run(self, caption_groups):
        return [
            {
                "video": captions[0]["video"],
                "frames": tuple(caption["frame"] for caption in captions),
                "captions": tuple(caption["caption"] for caption in captions),
            }
            for captions in caption_groups
        ]


class _VideoCaptionTopology(Pipeline):
    """Video -> Frame -> VLM caption -> Video."""

    def __init__(self) -> None:
        self.decode = RayModule(_DecodeFrames).ray_options(
            replicas=2, batch_size=1, num_cpus=0
        )
        self.caption = RayModule(_CaptionFrames).ray_options(
            replicas=2, batch_size=4, num_cpus=0
        )
        self.summary = RayModule(_SummarizeCaptions).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )

    def forward(self, videos: Port) -> Port:
        frames = F.expand(self.decode(videos))
        captions = self.caption(frames)
        return self.summary(F.reduce(captions))


def test_recorded_vlm_caption_stages_compile_and_preserve_frame_order():
    pipeline = _VideoCaptionTopology()
    _assert_topology(
        pipeline,
        targets=[_DecodeFrames, _CaptionFrames, _SummarizeCaptions],
        domains=2,
        expansions=1,
        reductions=1,
    )

    with Executor(pipeline) as executor:
        result = executor.run(
            [
                ("video-a", 3, 2),
                ("video-b", 5, 1),
                ("video-c", 2, 3),
            ],
            input_batch_size=1,
            max_active_input_batches=3,
        )

    assert [output["video"] for output in result.outputs] == [
        "video-a",
        "video-b",
        "video-c",
    ]
    assert [output["frames"] for output in result.outputs] == [
        (0, 1, 2),
        (0, 1, 2, 3, 4),
        (0, 1),
    ]
    caption = next(
        call for call in result.calls if call.udf_name.endswith("_CaptionFrames")
    )
    assert caption.grain_dispatches == 10
    assert all(call.grain_requeues == 0 for call in result.calls)


class _DecodeAudio:
    def run(self, videos):
        return [
            [
                {"video": name, "chunk": chunk}
                for chunk in range(audio_chunks)
            ]
            for name, _frame_count, audio_chunks in videos
        ]


class _TranscribeAudio:
    def run(self, chunks):
        return [
            {
                **chunk,
                "text": f"{chunk['video']}:audio:{chunk['chunk']}",
            }
            for chunk in chunks
        ]


class _SummarizeAudio:
    def run(self, transcript_groups):
        return [
            {
                "video": transcripts[0]["video"],
                "audio": tuple(item["chunk"] for item in transcripts),
            }
            for transcripts in transcript_groups
        ]


class _VisionFrames:
    def run(self, frames):
        return [
            {
                **frame,
                "feature": f"{frame['video']}:vision:{frame['frame']}",
            }
            for frame in frames
        ]


class _SummarizeVision:
    def run(self, feature_groups):
        return [
            {
                "video": features[0]["video"],
                "vision": tuple(item["frame"] for item in features),
            }
            for features in feature_groups
        ]


class _MergeModalities:
    def run(self, audio, vision):
        return [
            {
                "video": audio_item["video"],
                "audio": audio_item["audio"],
                "vision": vision_item["vision"],
            }
            for audio_item, vision_item in zip(audio, vision, strict=True)
        ]


class _VideoMultimodalTopology(Pipeline):
    """Two sibling child Domains reduced and joined at their Video parent."""

    def __init__(self) -> None:
        self.audio_decode = RayModule(_DecodeAudio).ray_options(
            replicas=2, batch_size=1, num_cpus=0
        )
        self.asr = RayModule(_TranscribeAudio).ray_options(
            replicas=2, batch_size=4, num_cpus=0
        )
        self.audio_summary = RayModule(_SummarizeAudio).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )
        self.frame_decode = RayModule(_DecodeFrames).ray_options(
            replicas=2, batch_size=1, num_cpus=0
        )
        self.vision = RayModule(_VisionFrames).ray_options(
            replicas=2, batch_size=4, num_cpus=0
        )
        self.vision_summary = RayModule(_SummarizeVision).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )
        self.merge = RayModule(_MergeModalities).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )

    def forward(self, videos: Port) -> Port:
        audio_chunks = F.expand(self.audio_decode(videos))
        transcripts = self.asr(audio_chunks)
        audio = self.audio_summary(F.reduce(transcripts))

        frames = F.expand(self.frame_decode(videos))
        features = self.vision(frames)
        vision = self.vision_summary(F.reduce(features))
        return self.merge(audio, vision)


def test_recorded_dual_modality_stages_form_sibling_domains_and_root_join():
    pipeline = _VideoMultimodalTopology()
    _assert_topology(
        pipeline,
        targets=[
            _DecodeAudio,
            _TranscribeAudio,
            _SummarizeAudio,
            _DecodeFrames,
            _VisionFrames,
            _SummarizeVision,
            _MergeModalities,
        ],
        domains=3,
        expansions=2,
        reductions=2,
    )
    compiled = pipeline.compile()
    root, audio, vision = compiled.logical.domains.values()
    assert root.parent is None
    assert audio.parent == root.ref
    assert vision.parent == root.ref

    with Executor(pipeline) as executor:
        result = executor.run(
            [
                ("clip-a", 3, 2),
                ("clip-b", 2, 4),
            ],
            input_batch_size=1,
            max_active_input_batches=2,
        )

    assert result.outputs == [
        {
            "video": "clip-a",
            "audio": (0, 1),
            "vision": (0, 1, 2),
        },
        {
            "video": "clip-b",
            "audio": (0, 1, 2, 3),
            "vision": (0, 1),
        },
    ]
    assert result.peak_active_input_batches == 2
    assert all(call.grain_requeues == 0 for call in result.calls)
