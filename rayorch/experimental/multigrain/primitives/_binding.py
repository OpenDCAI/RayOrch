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
    factory_import_path,
    require_compilable_factory,
    validate_output_count,
)
from ..ir.operations import OperatorFactorySpec
from ..ir.policy import RecoveryPolicy, WorkerPoolSpec


def load_factory_object(ref: str) -> Any:
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
    workers: WorkerPoolSpec
    recovery: RecoveryPolicy
    factory: OperatorFactorySpec
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
        workers: WorkerPoolSpec | None,
        recovery: RecoveryPolicy | None,
    ) -> "PrimitiveBinding":
        init_kwargs = dict(kwargs)
        return cls(
            op_cls=op_cls,
            name=op_name(op_cls, name),
            num_outputs=validate_output_count(num_outputs),
            workers=workers or WorkerPoolSpec(),
            recovery=recovery or RecoveryPolicy(),
            factory=OperatorFactorySpec(
                import_path=factory_import_path(op_cls),
                args=tuple(args),
                kwargs=init_kwargs,
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
    def workers(self) -> WorkerPoolSpec:
        return self._binding.workers

    @property
    def recovery(self) -> RecoveryPolicy:
        return self._binding.recovery

    @property
    def factory_spec(self) -> OperatorFactorySpec:
        return self._binding.factory

    @property
    def op(self) -> Any:
        return self._binding.op
