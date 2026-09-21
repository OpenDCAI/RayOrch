from __future__ import annotations

import json
from pathlib import Path

import pytest

from rayorch.benchmarks.panda70m.benchmark import load_panda_sources
from rayorch.benchmarks.panda70m.udfs import (
    SelectPandaCaption,
    SummarizePandaSource,
    VideoCaptionUDF,
)


def test_qwen_template_is_deterministic():
    assert VideoCaptionUDF._qwen_template("Describe the frame.") == (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Describe the frame.<|im_end|>\n<|im_start|>assistant\n"
    )


def test_manifest_normalization_and_sampling(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "source-0",
                        "clips": [
                            {
                                "clip_index": 1,
                                "path": "hdfs://videos/source-0.mp4",
                            },
                            {
                                "clip_index": 0,
                                "path": "hdfs://videos/source-0.mp4",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    sources = load_panda_sources(manifest, sample_multiplier=2)
    assert [source["source_id"] for source in sources] == [
        "source-0__sample_000",
        "source-0__sample_001",
    ]
    assert [clip["clip_index"] for clip in sources[0]["clips"]] == [0, 1]
    assert all(
        clip["source_id"] == "source-0__sample_000"
        for clip in sources[0]["clips"]
    )


def test_selector_and_source_reduce_write_ordered_outputs(tmp_path: Path):
    decoded = [
        {
            "source_id": "source-0",
            "clip_index": 0,
            "path": "video.mp4",
            "reference_caption": "person opens a door",
            "matching_score": 0.8,
        }
    ]
    candidates = [
        [
            {"role": "opening_action", "caption": "a person opens a door"},
            {"role": "scene_objects", "caption": "person and door"},
            {"role": "action_context", "caption": "a person moves"},
            {"role": "closing_action", "caption": "door"},
        ]
    ]
    selected = SelectPandaCaption().run(decoded, candidates)
    outputs = SummarizePandaSource(output_dir=str(tmp_path)).run(
        [{"source_id": "source-0"}],
        [selected],
    )
    assert outputs[0]["source_id"] == "source-0"
    assert Path(outputs[0]["output_path"]).is_file()


def test_selector_rejects_incomplete_teacher_candidates():
    with pytest.raises(ValueError, match="exactly four"):
        SelectPandaCaption().run(
            [{"source_id": "source-0", "clip_index": 0, "path": "video.mp4"}],
            [[{"role": "opening_action", "caption": "caption"}]],
        )
