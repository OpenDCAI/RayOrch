"""视频 caption summary 的快速测试。"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from rayorch.experimental.multigrain_v3.benchmark.video.workload import (
    FrameCaption,
    FrameCaptioner,
    summarize_captions,
)


def test_caption_summary_preserves_order() -> None:
    """有序 captions 应生成稳定 summary。"""

    result = summarize_captions(
        [
            FrameCaption(0, "baby crawling"),
            FrameCaption(5, "baby on floor"),
        ]
    )

    assert result["source_indices"] == (0, 5)
    assert result["captions"] == ("baby crawling", "baby on floor")


def test_caption_summary_rejects_reordering() -> None:
    """乱序 captions 必须由 Reduce gate 拒绝。"""

    with pytest.raises(ValueError, match="ordered"):
        summarize_captions(
            [
                FrameCaption(5, "later"),
                FrameCaption(0, "earlier"),
            ]
        )


def test_frame_captioner_uses_left_padding_for_decoder_only_batches(
    monkeypatch,
) -> None:
    tokenizer = SimpleNamespace(padding_side="right")
    processor = SimpleNamespace(
        tokenizer=tokenizer,
        apply_chat_template=lambda *args, **kwargs: "prompt",
    )
    model = SimpleNamespace(eval=lambda: model)
    fake_transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: processor
        ),
        AutoModelForImageTextToText=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model
        ),
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float16="float16",
        float32="float32",
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    FrameCaptioner("model")

    assert tokenizer.padding_side == "left"
