"""Input and configuration checks for the dual-vLLM Benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from rayorch.benchmark import DualVllmBench, benchmark_config
from rayorch.benchmarks.dual_vllm.benchmark import _load_prompts


def test_dual_vllm_configuration_round_trips(tmp_path: Path):
    benchmark = DualVllmBench(
        input_path=tmp_path / "prompts.jsonl",
        output_dir=tmp_path / "output",
        model_a=tmp_path / "model-a",
        model_b=tmp_path / "model-b",
        stage_options={"model_b": {"resources": {"node_b": 0.001}}},
    )

    assert DualVllmBench(**benchmark_config(benchmark)) == benchmark


def test_load_prompts_accepts_text_and_jsonl(tmp_path: Path):
    text = tmp_path / "prompts.txt"
    text.write_text("first\n\nsecond\n", encoding="utf-8")
    assert _load_prompts(text, 1) == ("first",)

    jsonl = tmp_path / "prompts.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "alpha"}),
                json.dumps("beta"),
            ]
        ),
        encoding="utf-8",
    )
    assert _load_prompts(jsonl, None) == ("alpha", "beta")
