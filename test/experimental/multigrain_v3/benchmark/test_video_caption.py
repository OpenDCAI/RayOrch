"""视频 caption summary 的快速测试。"""

from __future__ import annotations

import pytest

from rayorch.experimental.multigrain_v3.benchmark.video.workload import (
    FrameCaption,
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
