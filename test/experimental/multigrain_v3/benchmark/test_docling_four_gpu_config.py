"""Docling V3 四卡 split-stage 配置与 GPU 采样器测试。"""

from __future__ import annotations

from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_v3 import (
    DoclingCoreV3Pipeline,
    four_gpu_split_stage_options,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.gpu_monitor import (
    GpuMonitor,
)


def _stage_options(pipeline: DoclingCoreV3Pipeline) -> dict[str, dict]:
    """按 UDF 类型名提取编译 DAG 的 execution 参数。"""

    return {
        stage.udf.target.__name__: {
            "replicas": stage.execution.replicas,
            "batch_scope": stage.execution.batch_scope,
            **dict(stage.execution.ray_options),
        }
        for stage in pipeline.compile().dag.stages
        if stage.execution is not None and stage.udf is not None
    }


def test_four_gpu_split_stage_compiles_static_four_gpu_plan() -> None:
    """四卡预设必须为 Layout/Table 分配合计四张静态 GPU。"""

    options = four_gpu_split_stage_options()
    compiled_options = _stage_options(DoclingCoreV3Pipeline(**options))

    assert compiled_options["DoclingLayoutPages"]["replicas"] == 1
    assert compiled_options["DoclingLayoutPages"]["num_gpus"] == 1.0
    assert compiled_options["DoclingLayoutPages"]["max_concurrency"] == 1
    assert compiled_options["DoclingTableCore"]["replicas"] == 3
    assert compiled_options["DoclingTableCore"]["num_gpus"] == 1.0
    assert compiled_options["DoclingTableCore"]["max_concurrency"] == 1
    assert compiled_options["ExpandDoclingTableJobs"].get("num_gpus", 0.0) == 0.0
    assert compiled_options["DoclingOcrPages"]["replicas"] == 4
    assert compiled_options["ExpandDoclingPages"]["replicas"] == 1
    assert compiled_options["ReduceDoclingDocument"]["replicas"] == 4
    assert sum(
        options["replicas"] * options.get("num_gpus", 0.0)
        for options in compiled_options.values()
    ) == 4.0


def test_four_gpu_split_stage_keeps_batch_scope_local_to_model_stages() -> None:
    """parent_bound 与 elastic 仅应改变模型 stage 的 batch scope。"""

    options = four_gpu_split_stage_options()
    parent = _stage_options(
        DoclingCoreV3Pipeline(**options, batch_scope="parent_bound")
    )
    elastic = _stage_options(
        DoclingCoreV3Pipeline(**options, batch_scope="elastic")
    )

    assert parent.keys() == elastic.keys()
    for name in parent:
        if name in {
            "DoclingLayoutPages",
            "DoclingOcrPages",
            "DoclingPostprocessPages",
            "ExpandDoclingTableJobs",
            "DoclingTableCore",
        }:
            assert parent[name]["batch_scope"] == "parent_bound"
            assert elastic[name]["batch_scope"] == "elastic"
            parent_options = {
                k: v for k, v in parent[name].items() if k != "batch_scope"
            }
            elastic_options = {
                k: v for k, v in elastic[name].items() if k != "batch_scope"
            }
            assert parent_options == elastic_options
        else:
            assert parent[name] == elastic[name]


def test_gpu_monitor_returns_empty_samples_without_pynvml(monkeypatch) -> None:
    """NVML 缺失时采样器必须无异常返回空样本并说明不可用。"""

    def missing_pynvml():
        raise ModuleNotFoundError("No module named 'pynvml'")

    monkeypatch.setattr(
        "rayorch.experimental.multigrain_v3.benchmark.document_docling."
        "gpu_monitor._load_pynvml",
        missing_pynvml,
    )

    monitor = GpuMonitor(interval_s=0.01).start()

    assert monitor.available is False
    assert "ModuleNotFoundError" in (monitor.unavailable_reason or "")
    assert monitor.stop() == ()
