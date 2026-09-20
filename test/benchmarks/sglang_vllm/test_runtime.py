"""Dependency-free end-to-end execution of the SGLang-vLLM graph."""

from __future__ import annotations

from pathlib import Path

from rayorch import RayModule
from rayorch.benchmark import SglangVllmBench
from rayorch.benchmarks.sglang_vllm import benchmark as sglang_vllm
from rayorch.benchmarks.sglang_vllm.pipeline import SglangVllmPipeline
from rayorch.benchmarks.sglang_vllm.udfs import (
    BuildHandoffPrompts,
    BuildResults,
)


class _Sglang:
    def run(self, prompts):
        return [f"S:{prompt}" for prompt in prompts]


class _Vllm:
    def run(self, prompts):
        return [f"V:{prompt}" for prompt in prompts]


class _SyntheticPipeline(SglangVllmPipeline):
    def __init__(self):
        self.sglang = RayModule(_Sglang).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )
        self.handoff = RayModule(BuildHandoffPrompts).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )
        self.vllm = RayModule(_Vllm).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )
        self.result = RayModule(BuildResults).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )


class _SyntheticBench(SglangVllmBench):
    def _pipeline(self):
        return _SyntheticPipeline()


def test_sglang_vllm_benchmark_runs_end_to_end(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        sglang_vllm,
        "_load_prompts",
        lambda path, limit: ("hello", "world"),
    )
    monkeypatch.setattr(sglang_vllm, "_validate", lambda benchmark: None)

    report = _SyntheticBench(
        input_path=tmp_path / "prompts.txt",
        output_dir=tmp_path / "output",
        model=tmp_path / "model",
        sglang_env="sglang-env",
        vllm_env="vllm-env",
        batch_size=2,
        input_batch_size=2,
    ).run(
        ray_init_kwargs={"include_dashboard": False, "num_gpus": 0},
        profile=False,
        run_id="synthetic-sglang-vllm",
    )

    assert report.metrics["prompts"] == 2
    assert [item["sglang"] for item in report.outputs] == [
        "S:hello",
        "S:world",
    ]
    assert all(item["vllm"].startswith("V:Review") for item in report.outputs)
