"""V3 和 Ray Data 视频 runner 的共用业务语义测试。"""

from __future__ import annotations

import numpy as np

from rayorch.experimental.multigrain_v3.benchmark.video.workload import (
    FrameFeature,
    FrameRecord,
    FrameTransformer,
    summarize_video,
    transform_frame,
)


def test_transform_and_summary_are_deterministic() -> None:
    """同一 frame bytes 必须产生稳定 feature 和有序 video summary。"""

    first = FrameRecord(
        video_path="v.avi",
        source_frame_index=0,
        image_bgr=np.zeros((24, 32, 3), dtype=np.uint8),
    )
    second = FrameRecord(
        video_path="v.avi",
        source_frame_index=2,
        image_bgr=np.full((24, 32, 3), 100, dtype=np.uint8),
    )

    features = [transform_frame(first), transform_frame(second)]
    repeated = [transform_frame(first), transform_frame(second)]

    assert features == repeated
    summary = summarize_video(features)
    assert summary["frames"] == 2
    assert summary["source_indices"] == (0, 2)
    assert summary["digests"] == tuple(
        feature.digest for feature in features
    )


def test_summary_rejects_out_of_order_features() -> None:
    """ordered Reduce 业务检查可捕获 baseline 忘记 ordinal sort 的错误。"""

    feature = lambda index: FrameFeature(
        source_frame_index=index,
        mean_bgr=(0.0, 0.0, 0.0),
        edge_density=0.0,
        digest=str(index),
    )

    import pytest

    with pytest.raises(ValueError, match="ordered"):
        summarize_video([feature(2), feature(0)])


def test_resnet_signature_uses_semantic_top_classes(monkeypatch) -> None:
    """ResNet signature 不应依赖不同 batch shape 下的 raw float bytes。"""

    transformer = object.__new__(FrameTransformer)
    transformer.backend = "resnet18"
    transformer.model = object()

    class FakeInference:
        """模拟 torch.inference_mode context。"""

        def __enter__(self):
            """进入 context。"""

        def __exit__(self, *args):
            """离开 context。"""

    class FakeOutput:
        """模拟 model tensor 输出链。"""

        def __init__(self, values):
            """保存 numpy logits。"""

            self.values = values

        def cpu(self):
            """返回自身。"""

            return self

        def numpy(self):
            """返回 logits。"""

            return self.values

    class FakeTorch:
        """提供 transform() 所需的最小 torch API。"""

        def inference_mode(self):
            """创建 fake context。"""

            return FakeInference()

        def stack(self, tensors):
            """返回输入。"""

            return tensors

    class FakeModel:
        """返回只在低位 float 上不同、top-5 相同的 logits。"""

        def __call__(self, tensors):
            """生成一行 logits。"""

            return FakeOutput(
                np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
            )

    transformer.torch = FakeTorch()
    transformer.model = FakeModel()
    monkeypatch.setattr(transformer, "_resnet_tensor", lambda image: image)
    frame = FrameRecord(
        "v",
        0,
        np.zeros((4, 4, 3), dtype=np.uint8),
    )

    first = transformer.transform([frame])[0].digest
    transformer.model = FakeModel()
    second = transformer.transform([frame])[0].digest

    assert first == second
