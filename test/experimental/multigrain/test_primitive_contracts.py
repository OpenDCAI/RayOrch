from __future__ import annotations

import pytest

from rayorch.experimental import multigrain as mg


class _Identity:
    def run(self, rows):
        return list(rows)


class _OneGroup:
    def run(self, rows):
        return [[row] for row in rows]


class _ReduceOne:
    def run(self, anchors, groups):
        return [len(group) for group in groups]


def test_compiled_primitive_rejects_live_operator_instance() -> None:
    class Pipe(mg.Pipeline):
        def __init__(self):
            super().__init__()
            self.op = mg.Map(_Identity())

        def forward(self, rows):
            return self.op(rows)

    with pytest.raises(TypeError, match="instances are eager-only"):
        Pipe().compile()

    rows = mg.source(["a"], name="rows")
    assert mg.Map(_Identity())(rows).values == ["a"]


def test_relate_rejects_undefined_multi_output_contract() -> None:
    with pytest.raises(ValueError, match="exactly one output"):
        mg.Relate(_Identity, num_outputs=2)


def test_compiled_relate_rejects_callable_join_key() -> None:
    class Pipe(mg.Pipeline):
        def __init__(self):
            super().__init__()
            self.join = mg.Relate(
                _Identity,
                on={"left": lambda value: value, "right": lambda value: value},
            )

        def forward(self, left, right):
            return self.join(left, right)

    with pytest.raises(TypeError, match="field names"):
        Pipe().compile()


def test_expand_validates_declared_output_arity() -> None:
    rows = mg.source(["a"], name="rows")
    with pytest.raises(ValueError, match="expected 2 outputs"):
        mg.Expand(_OneGroup, num_outputs=2)(rows)


def test_reduce_validates_declared_output_arity() -> None:
    anchors = mg.source(["a"], name="anchors")
    children = mg.Expand(_OneGroup)(anchors)
    with pytest.raises(ValueError, match="expected 2 outputs"):
        mg.Reduce(_ReduceOne, num_outputs=2)(mg.group_by(anchors, children))


@pytest.mark.parametrize("primitive", [mg.Map, mg.Expand, mg.Reduce])
def test_output_arity_must_be_positive_integer(primitive) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        primitive(_Identity, num_outputs=0)
    with pytest.raises(ValueError, match="positive integer"):
        primitive(_Identity, num_outputs=1.5)
