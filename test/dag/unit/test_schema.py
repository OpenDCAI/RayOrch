from __future__ import annotations

from typing import Generic, TypeVar

import pytest

import rayorch
from rayorch import DagPipeline, RayModule


class IntToText:
    def run(self, values: list[int]) -> list[str]:
        return [str(value) for value in values]


class AddOne:
    def run(self, values: list[int]) -> list[int]:
        return [value + 1 for value in values]


T = TypeVar("T")


class Identity(Generic[T]):
    def run(self, values: list[T]) -> list[T]:
        return values


def test_symbolic_ref_is_not_public_api() -> None:
    assert not hasattr(rayorch, "PipeRef")


def test_forward_uses_application_types_and_records_port_schema() -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.convert = RayModule(IntToText)
            super().__init__()

        def forward(self, values: list[int]) -> list[str]:
            return self.convert(values)

    graph = Pipe().compile()._compiled
    assert graph is not None
    assert graph.input_types["__input__values"] == list[int]
    assert graph.nodes["convert"].input_types == (list[int],)
    assert graph.nodes["convert"].output_types == (list[str],)
    assert graph.output_types == (list[str],)


def test_compile_infers_unannotated_source_type_from_operator() -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.add = RayModule(AddOne)
            super().__init__()

        def forward(self, values):
            return self.add(values)

    graph = Pipe().compile()._compiled
    assert graph is not None
    assert graph.input_types["__input__values"] == list[int]


def test_generic_operator_propagates_bound_typevar() -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.identity = RayModule(Identity)
            super().__init__()

        def forward(self, values: list[int]) -> list[int]:
            return self.identity(values)

    graph = Pipe().compile()._compiled
    assert graph is not None
    assert graph.nodes["identity"].output_types == (list[int],)


def test_compile_rejects_node_input_type_mismatch() -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.convert = RayModule(IntToText)
            self.add = RayModule(AddOne)
            super().__init__()

        def forward(self, values: list[int]) -> list[int]:
            text = self.convert(values)
            return self.add(text)

    with pytest.raises(TypeError, match="type mismatch at node 'add'"):
        Pipe().compile()


def test_compile_rejects_forward_output_type_mismatch() -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.convert = RayModule(IntToText)
            super().__init__()

        def forward(self, values: list[int]) -> list[int]:
            return self.convert(values)

    with pytest.raises(TypeError, match="pipeline output type mismatch"):
        Pipe().compile()


@pytest.mark.parametrize(
    ("body", "operation"),
    [
        (lambda value: bool(value), "bool\\(\\)/if"),
        (lambda value: len(value), "len\\(\\)"),
        (lambda value: value[0], "indexing"),
        (lambda value: list(value), "iteration"),
    ],
)
def test_symbolic_values_reject_data_dependent_python_operations(
    body,
    operation: str,
) -> None:
    class Pipe(DagPipeline):
        def __init__(self):
            self.add = RayModule(AddOne)
            super().__init__()

        def forward(self, values: list[int]) -> list[int]:
            body(values)
            return self.add(values)

    with pytest.raises(TypeError, match=operation):
        Pipe().compile()

