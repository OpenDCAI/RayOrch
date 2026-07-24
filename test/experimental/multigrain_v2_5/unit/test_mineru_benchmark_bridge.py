"""CPU-only contract tests for the real MinerU V2.5 benchmark bridge."""

from __future__ import annotations

from rayorch.experimental.multigrain_v2_5.benchmark.mineru import (
    MinerUV25Pipeline,
    build_parser,
)
from rayorch.experimental.multigrain_v2_5.graph import Primitive


def test_mineru_pipeline_compiles_expand_gpu_map_and_aligned_reduce():
    """The real bridge remains a native Pipeline with ordered page side input."""

    pipeline = MinerUV25Pipeline(
        output_dir="/tmp/mineru-v2-5-test",
        mode="elastic",
        model="/tmp/model",
        replicas=4,
        batch_size=16,
        max_batch_wait_ms=5.0,
        gpu_memory_utilization=0.9,
        render_replicas=4,
        reduce_replicas=4,
        runtime_env={},
    )
    compiled = pipeline.compile()
    kinds = tuple(node.kind for node in compiled.graph.nodes)
    assert kinds == (
        Primitive.SOURCE,
        Primitive.EXPAND,
        Primitive.MAP,
        Primitive.REDUCE,
    )
    ocr = compiled.graph.nodes[2]
    assert ocr.execution is not None
    assert ocr.execution.replicas == 4
    assert dict(ocr.execution.options)["num_gpus"] == 1.0
    reduce = compiled.graph.nodes[3]
    assert tuple(binding.role for binding in reduce.inputs) == (
        "anchor",
        "members",
        "pages",
    )


def test_mineru_cli_defaults_to_safe_four_document_smoke():
    """Manual CLI defaults avoid accidentally launching the 368-PDF run."""

    args = build_parser().parse_args([])
    assert args.limit == 4
    assert args.replicas == 4
    assert args.batch_size == 16
    assert args.mode == "elastic"
