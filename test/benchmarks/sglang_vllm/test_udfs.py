"""Small compatibility checks for the SGLang-vLLM UDFs."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

from rayorch.benchmarks.sglang_vllm.udfs import SglangGenerate


class _Engine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, prompts, sampling):
        asyncio.get_event_loop()
        return [{"text": f"generated:{prompt}"} for prompt in prompts]


def test_sglang_udf_provides_event_loop_for_sync_engine(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=_Engine))
    udf = SglangGenerate("/models/model")
    asyncio.set_event_loop(None)

    assert udf.run(["hello"]) == ["generated:hello"]
