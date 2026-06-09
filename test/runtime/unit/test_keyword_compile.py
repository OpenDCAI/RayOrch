"""Compile-time contract for RuntimeRayModule keyword arguments."""
from __future__ import annotations

from rayorch import DagPipeline, RuntimeRayModule


class CombineOp:
    def run(self, first, second, third):
        return [
            f"{a}|{b}|{c}"
            for a, b, c in zip(first, second, third)
        ]


def test_compile_preserves_parameter_names_for_reordered_keywords() -> None:
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.combine = RuntimeRayModule(CombineOp)
            super().__init__()

        def forward(self, a: list[str], b: list[str], c: list[str]) -> list[str]:
            output = self.combine(third=c, first=a, second=b)
            return output

    pipe = Pipe().compile()
    spec = pipe._compiled.nodes["combine"]

    assert spec.args == ()
    assert tuple(spec.kw_args) == ("third", "first", "second")
    assert spec.input_names == ("first", "second", "third")
    assert pipe.combine.runtime_spec.inputs == ("first", "second", "third")


def test_compile_preserves_mixed_positional_and_keyword_parameters() -> None:
    class Pipe(DagPipeline):
        def __init__(self) -> None:
            self.combine = RuntimeRayModule(CombineOp)
            super().__init__()

        def forward(self, a: list[str], b: list[str], c: list[str]) -> list[str]:
            output = self.combine(a, third=c, second=b)
            return output

    spec = Pipe().compile()._compiled.nodes["combine"]

    assert len(spec.args) == 1
    assert tuple(spec.kw_args) == ("third", "second")
    assert spec.input_names == ("first", "second", "third")
