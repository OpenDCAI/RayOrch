"""Dependency-free end-to-end execution of the dual-vLLM graph."""

from __future__ import annotations

from pathlib import Path

from rayorch import RayModule
from rayorch.benchmark import DualVllmBench
from rayorch.benchmarks.dual_vllm import benchmark as dual_vllm
from rayorch.benchmarks.dual_vllm.pipeline import DualVllmPipeline
from rayorch.benchmarks.dual_vllm.udfs import (
    BuildRefinementPrompts,
    BuildResults,
)


class _ModelA:
    def run(self, prompts):
        return [f"A:{prompt}" for prompt in prompts]


class _ModelB:
    def run(self, prompts):
        return [f"B:{prompt}" for prompt in prompts]


class _SyntheticPipeline(DualVllmPipeline):
    def __init__(self):
        self.model_a = RayModule(_ModelA).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )
        self.prompt = RayModule(BuildRefinementPrompts).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )
        self.model_b = RayModule(_ModelB).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )
        self.result = RayModule(BuildResults).ray_options(
            replicas=1, batch_size=2, num_cpus=0
        )


class _SyntheticBench(DualVllmBench):
    def _pipeline(self):
        return _SyntheticPipeline()


def test_dual_vllm_benchmark_runs_end_to_end(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        dual_vllm,
        "_load_prompts",
        lambda path, limit: ("hello", "world"),
    )
    monkeypatch.setattr(dual_vllm, "_validate", lambda benchmark: None)

    report = _SyntheticBench(
        input_path=tmp_path / "prompts.txt",
        output_dir=tmp_path / "output",
        model_a=tmp_path / "model-a",
        model_b=tmp_path / "model-b",
        batch_size=2,
        input_batch_size=2,
    ).run(
        ray_init_kwargs={"include_dashboard": False, "num_gpus": 0},
        profile=False,
        run_id="synthetic-dual-vllm",
    )

    assert report.metrics["prompts"] == 2
    assert [item["model_a"] for item in report.outputs] == [
        "A:hello",
        "A:world",
    ]
    assert all(item["model_b"].startswith("B:Refine") for item in report.outputs)
