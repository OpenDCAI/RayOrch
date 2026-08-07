"""MinerU v3.6 Pipeline 的结构、UDF 复用和 CLI gate 测试。"""

from __future__ import annotations

from rayorch.experimental.multigrain_v3.benchmark.mineru import (
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    PdfMetadata,
)
from rayorch.experimental.multigrain_v3_6.benchmark.mineru import (
    MinerUV36Pipeline,
    build_parser,
)
from rayorch.experimental.multigrain_v3_6.logical import ExpandOrigin, GroupOrigin


def _pipeline(mode: str = "elastic") -> MinerUV36Pipeline:
    return MinerUV36Pipeline(
        output_dir="/tmp/v36-mineru-test",
        mode=mode,
        model="model",
        replicas=4,
        batch_size=64,
        gpu_memory_utilization=0.8,
        render_replicas=4,
        reduce_replicas=4,
        runtime_env={},
    )


def test_mineru_v36_has_four_calls_and_no_structural_pools():
    compiled = _pipeline().compile()
    targets = [spec.kernel.target for spec in compiled.logical.calls.values()]

    assert targets == [
        MinerUPdfToPages,
        MinerUVlmOcrPage,
        PdfMetadata,
        MinerUAssembleDoc,
    ]
    assert len(compiled.runtime.pools_by_call) == 4
    assert len(compiled.logical.domains) == 2
    assert sum(
        isinstance(spec.origin, ExpandOrigin)
        for spec in compiled.logical.ports.values()
    ) == 1
    assert sum(
        isinstance(spec.origin, GroupOrigin)
        for spec in compiled.logical.ports.values()
    ) == 2


def test_mineru_parent_and_elastic_only_change_ocr_pool_option():
    elastic = _pipeline("elastic").compile()
    parent = _pipeline("parent_bound").compile()

    assert elastic.logical.calls == parent.logical.calls
    elastic_options = list(elastic.runtime.pools_by_call.values())
    parent_options = list(parent.runtime.pools_by_call.values())
    changed = [
        (left, right)
        for left, right in zip(elastic_options, parent_options)
        if left != right
    ]
    assert len(changed) == 1
    assert changed[0][0].batch_scope == "elastic"
    assert changed[0][1].batch_scope == "parent_bound"


def test_mineru_cli_defaults_to_four_pdf_correctness_gate():
    args = build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--artifact-dir",
            "/tmp/artifacts",
            "--result-jsonl",
            "/tmp/results.jsonl",
        ]
    )

    assert args.limit == 4
    assert args.batch_size == 64
    assert args.max_inflight_arenas == 3
    assert args.num_cpus == 32
    assert args.object_store_gb == 100
