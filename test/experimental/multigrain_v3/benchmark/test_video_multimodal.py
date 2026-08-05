"""视频双分支 Pipeline 的 compile contract 测试。"""

from __future__ import annotations

from rayorch.experimental.multigrain_v3.benchmark.video.multimodal_v3 import (
    VideoMultimodalV3Pipeline,
)


def test_multimodal_pipeline_compiles_two_expand_reduce_branches() -> None:
    """Audio/frame 两个 fan-out scope 应在 document scope aligned merge。"""

    compiled = VideoMultimodalV3Pipeline(
        whisper_model_path="/tmp/whisper",
        vit_model_path="/tmp/vit",
    ).compile()

    assert [stage.kind.value for stage in compiled.dag.stages] == [
        "source",
        "expand",
        "map",
        "reduce",
        "expand",
        "map",
        "reduce",
        "map",
    ]
