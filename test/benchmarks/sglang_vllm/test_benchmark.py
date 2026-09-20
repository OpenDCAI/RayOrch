"""Input and configuration checks for the SGLang-vLLM Benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from rayorch.benchmark import SglangVllmBench, benchmark_config
from rayorch.benchmarks.sglang_vllm.benchmark import _load_prompts


def test_sglang_vllm_configuration_round_trips(tmp_path: Path):
    benchmark = SglangVllmBench(
        input_path=tmp_path / "prompts.jsonl",
        output_dir=tmp_path / "output",
        model=tmp_path / "model",
        sglang_env="sglang-env",
        vllm_env="vllm-env",
        stage_options={"vllm": {"resources": {"vllm_node": 0.001}}},
    )

    assert SglangVllmBench(**benchmark_config(benchmark)) == benchmark


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
