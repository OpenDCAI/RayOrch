"""Business UDFs for an SGLang -> vLLM inference workflow."""

from __future__ import annotations

import asyncio
from typing import Any


class SglangGenerate:
    def __init__(
        self,
        model: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        import sglang as sgl  # pyright: ignore[reportMissingImports]

        self.engine = sgl.Engine(
            model_path=model,
            tp_size=tensor_parallel_size,
            mem_fraction_static=gpu_memory_utilization,
        )
        self.sampling = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }

    def run(self, prompts: list[str]) -> list[str]:
        _ensure_event_loop()
        outputs = self.engine.generate(prompts, self.sampling)
        return [_generated_text(output, backend="SGLang") for output in outputs]


class BuildHandoffPrompts:
    def run(self, answers: list[str]) -> list[str]:
        return [
            "Review the following draft, correct any errors, and return a "
            f"concise final answer:\n\n{answer}"
            for answer in answers
        ]


class VllmGenerate:
    def __init__(
        self,
        model: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        from vllm import LLM, SamplingParams  # pyright: ignore[reportMissingImports]

        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def run(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(prompts, self.sampling, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]


class BuildResults:
    def run(
        self,
        prompts: list[str],
        sglang_answers: list[str],
        vllm_answers: list[str],
    ) -> list[dict[str, Any]]:
        return [
            {
                "prompt": prompt,
                "sglang": sglang_answer,
                "vllm": vllm_answer,
            }
            for prompt, sglang_answer, vllm_answer in zip(
                prompts,
                sglang_answers,
                vllm_answers,
                strict=True,
            )
        ]


def _generated_text(output: Any, *, backend: str) -> str:
    if isinstance(output, dict) and isinstance(output.get("text"), str):
        return output["text"]
    text = getattr(output, "text", None)
    if isinstance(text, str):
        return text
    raise TypeError(f"{backend} returned an unsupported generation result")


def _ensure_event_loop() -> None:
    """Provide the loop expected by SGLang's synchronous Engine API."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())


__all__ = [
    "BuildHandoffPrompts",
    "BuildResults",
    "SglangGenerate",
    "VllmGenerate",
]
