"""MinerU v3.3 Pipeline 的静态结构、业务 UDF 复用与 CLI gate 测试。"""

from __future__ import annotations

from rayorch.experimental.multigrain_v3.benchmark.mineru import (
    MinerUAssembleDoc,
    MinerUPdfToPages,
    MinerUVlmOcrPage,
    PdfMetadata,
)
from rayorch.experimental.multigrain_v3_3.benchmark.mineru import (
    MinerUV33Pipeline,
    build_parser,
)
from rayorch.experimental.multigrain_v3_3.program import (
    ExpandOrigin,
    GroupOrigin,
)


def _pipeline(mode: str = "elastic") -> MinerUV33Pipeline:
    """构造不初始化模型/actor 的符号 Pipeline。"""

    return MinerUV33Pipeline(
        output_dir="/tmp/v33-mineru-test",
        mode=mode,
        model="model",
        replicas=4,
        batch_size=64,
        gpu_memory_utilization=0.8,
        render_replicas=4,
        reduce_replicas=4,
        runtime_env={},
    )


def test_mineru_v33_has_four_calls_and_zero_structural_pools():
    compiled = _pipeline().compile()
    program = compiled.program

    targets = [spec.kernel.target for spec in program.calls.values()]
    assert targets == [
        MinerUPdfToPages,
        MinerUVlmOcrPage,
        PdfMetadata,
        MinerUAssembleDoc,
    ]
    assert len(compiled.execution.pools) == 4
    assert len(program.domains) == 2
    assert sum(
        isinstance(spec.origin, ExpandOrigin) for spec in program.ports.values()
    ) == 1
    assert sum(
        isinstance(spec.origin, GroupOrigin) for spec in program.ports.values()
    ) == 2


def test_mineru_parent_and_elastic_only_change_execution_option():
    elastic = _pipeline("elastic").compile()
    parent = _pipeline("parent_bound").compile()

    assert elastic.program.calls == parent.program.calls
    elastic_options = [dict(pool.options) for pool in elastic.execution.pools.values()]
    parent_options = [dict(pool.options) for pool in parent.execution.pools.values()]
    assert elastic_options[1]["batch_scope"] == "elastic"
    assert parent_options[1]["batch_scope"] == "parent_bound"


def test_mineru_cli_defaults_to_four_pdf_correctness_gate():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--output-dir", "/tmp/out",
            "--artifact-dir", "/tmp/artifacts",
            "--result-jsonl", "/tmp/results.jsonl",
        ]
    )

    assert args.limit == 4
    assert args.batch_size == 64
    assert args.max_inflight_arenas == 3
