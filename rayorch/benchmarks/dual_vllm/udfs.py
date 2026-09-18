"""Business UDFs for a two-model vLLM refinement workflow."""

from __future__ import annotations

from typing import Any


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


class BuildRefinementPrompts:
    def run(self, answers: list[str]) -> list[str]:
        return [
            "Refine and improve the following answer. Preserve factual "
            f"content and make it concise:\n\n{answer}"
            for answer in answers
        ]


class BuildResults:
    def run(
        self,
        prompts: list[str],
        first_answers: list[str],
        final_answers: list[str],
    ) -> list[dict[str, Any]]:
        return [
            {
                "prompt": prompt,
                "model_a": first,
                "model_b": final,
            }
            for prompt, first, final in zip(
                prompts,
                first_answers,
                final_answers,
                strict=True,
            )
        ]


__all__ = ["BuildRefinementPrompts", "BuildResults", "VllmGenerate"]
