"""Small authoring-time binding shared by all primitive wrappers.

Purpose: own the repeated operator recipe/default/lazy-factory mechanics.
It deliberately does *not* own cardinality, lineage, execution, or recovery
algorithms; those remain explicit in each relation family.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Mapping

from ._utils import (
    LazyOp,
    op_name,
    op_ref,
    require_compilable_factory,
    validate_output_count,
)
from ..ir.model import (
    OperatorProperties,
    OperatorRecipe,
    PhysicalHints,
    RecoveryPolicy,
)


def load_recipe_object(ref: str) -> Any:
    module_name, _, attr = ref.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"invalid object reference: {ref}")
    target: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        target = getattr(target, part)
    return target


@dataclass
class PrimitiveBinding:
    op_cls: Any
    name: str
    num_outputs: int
    properties: OperatorProperties
    physical: PhysicalHints
    recovery: RecoveryPolicy
    recipe: OperatorRecipe
    lazy_op: LazyOp

    @classmethod
    def create(
        cls,
        op_cls: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        *,
        name: str | None,
        num_outputs: int,
        properties: OperatorProperties | None,
        physical: PhysicalHints | None,
        recovery: RecoveryPolicy | None,
        provenance: Mapping[str, Any] | None = None,
        default_physical: PhysicalHints | None = None,
    ) -> "PrimitiveBinding":
        init_kwargs = dict(kwargs)
        return cls(
            op_cls=op_cls,
            name=op_name(op_cls, name),
            num_outputs=validate_output_count(num_outputs),
            properties=properties or OperatorProperties(),
            physical=physical or default_physical or PhysicalHints(),
            recovery=recovery or RecoveryPolicy(),
            recipe=OperatorRecipe(
                cls_ref=op_ref(op_cls),
                args=tuple(args),
                kwargs=init_kwargs,
                provenance=dict(provenance or {}),
            ),
            lazy_op=LazyOp(op_cls, tuple(args), init_kwargs),
        )

    @property
    def op(self) -> Any:
        return self.lazy_op.get()

    def require_compilable(self, primitive: str) -> None:
        require_compilable_factory(self.op_cls, primitive)


class BoundPrimitive:
    """Property facade only; relation semantics must stay in concrete wrappers."""

    _binding: PrimitiveBinding

    @property
    def name(self) -> str:
        return self._binding.name

    @property
    def num_outputs(self) -> int:
        return self._binding.num_outputs

    @property
    def properties(self) -> OperatorProperties:
        return self._binding.properties

    @property
    def physical(self) -> PhysicalHints:
        return self._binding.physical

    @property
    def recovery(self) -> RecoveryPolicy:
        return self._binding.recovery

    @property
    def op_recipe(self) -> OperatorRecipe:
        return self._binding.recipe

    @property
    def op(self) -> Any:
        return self._binding.op
