"""Stable authoring facade for the backend-independent Multigrain V3 graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

from typing_extensions import ParamSpec

from .model.graph import (
    BatchPolicy,
    CompileError,
    FailurePolicy,
    OpaqueValue,
    ResourceSpec,
    freeze_config_value,
    freeze_constructor_arguments,
    freeze_runtime_env,
    thaw_config_value,
    trace_expand,
    trace_filter,
    trace_map_call,
    trace_reduce,
    validate_constructor_binding,
)
from .runtime.coordinator import ExecutorSession, RunStream


INITP = ParamSpec("INITP")
RUNP = ParamSpec("RUNP")
R = TypeVar("R")
S = TypeVar("S")


class RunOp(Protocol[INITP, RUNP, R]):
    """Structural protocol for a stateful class exposing a batched ``run``."""

    def __init__(self, *args: INITP.args, **kwargs: INITP.kwargs) -> None:
        """Initialize one persistent worker-side UDF instance."""

        ...

    def run(self, *args: RUNP.args, **kwargs: RUNP.kwargs) -> R:
        """Consume annotated batch columns and return annotated columns."""

        ...


class Pipeline(Generic[S]):
    """Base class whose single-source ``forward`` is traced into a graph."""

    def forward(self, source: Any) -> dict[str, Any]:
        """Describe a graph from one symbolic source to named root outputs."""

        raise NotImplementedError

    def compile(self):
        """Trace and return a verified immutable ``CompiledGraph``."""

        from .model.graph import GraphCompiler

        return GraphCompiler().compile(self)


@dataclass(frozen=True, slots=True, init=False)
class Map(Generic[INITP, RUNP, R]):
    """Immutable UDF constructor recipe and MAP execution configuration.

    Input roles and output leaves are intentionally absent from this class:
    they are inferred strictly from ``op_cls.run`` while tracing.
    """

    op_cls: type[RunOp[INITP, RUNP, R]]
    batch_policy: BatchPolicy
    resources: ResourceSpec
    failure_policy: FailurePolicy
    init_args: tuple[object, ...]
    init_kwargs: tuple[tuple[str, object], ...]

    def __init__(
        self,
        op_cls: type[RunOp[INITP, RUNP, R]],
        *,
        replicas: int = 1,
        batch_size: int = 1,
        max_batch_wait_ms: float = 2.0,
        num_cpus: float = 1.0,
        num_gpus: float = 0.0,
        failure_policy: FailurePolicy | None = None,
        runtime_env: Mapping[str, object] | None = None,
    ) -> None:
        """Capture a UDF class and validate immutable execution options."""

        if not isinstance(op_cls, type):
            raise TypeError("Map expects a UDF class, not an instance")
        if failure_policy is not None and not isinstance(
            failure_policy,
            FailurePolicy,
        ):
            raise TypeError("failure_policy must be a FailurePolicy")
        environment = freeze_runtime_env(runtime_env)
        object.__setattr__(self, "op_cls", op_cls)
        object.__setattr__(
            self,
            "batch_policy",
            BatchPolicy(batch_size, max_batch_wait_ms),
        )
        object.__setattr__(
            self,
            "resources",
            ResourceSpec(
                replicas,
                num_cpus,
                num_gpus,
                environment,
            ),
        )
        object.__setattr__(
            self,
            "failure_policy",
            failure_policy or FailurePolicy.raise_(),
        )
        object.__setattr__(self, "init_args", ())
        object.__setattr__(self, "init_kwargs", ())

    def pre_init(
        self,
        *args: INITP.args,
        **kwargs: INITP.kwargs,
    ) -> "Map[INITP, RUNP, R]":
        """Return a new Map with a constructor call validated and frozen."""

        raw_args = tuple(args)
        raw_kwargs = tuple(kwargs.items())
        validate_constructor_binding(
            self.op_cls,
            raw_args,
            raw_kwargs,
        )
        init_args, init_kwargs = freeze_constructor_arguments(
            raw_args,
            raw_kwargs,
        )
        return self._copy(
            init_args=init_args,
            init_kwargs=init_kwargs,
        )

    def with_options(self, **options: object) -> "Map[INITP, RUNP, R]":
        """Return a new Map after replacing selected execution options.

        Supported names are ``replicas``, ``batch_size``,
        ``max_batch_wait_ms``, ``num_cpus``, ``num_gpus``,
        ``runtime_env``, and ``failure_policy``.
        """

        allowed = {
            "replicas",
            "batch_size",
            "max_batch_wait_ms",
            "num_cpus",
            "num_gpus",
            "runtime_env",
            "failure_policy",
        }
        unknown = set(options).difference(allowed)
        if unknown:
            raise TypeError(
                "unsupported Map option(s): " + ", ".join(sorted(unknown))
            )
        batch = BatchPolicy(
            cast(
                int,
                options.get("batch_size", self.batch_policy.max_size),
            ),
            cast(
                float,
                options.get(
                    "max_batch_wait_ms",
                    self.batch_policy.max_wait_ms,
                ),
            ),
        )
        if "runtime_env" in options:
            runtime_env = options["runtime_env"]
            if runtime_env is not None and not isinstance(
                runtime_env,
                Mapping,
            ):
                raise TypeError("runtime_env must be a mapping or None")
            environment = freeze_runtime_env(
                cast(Mapping[str, object] | None, runtime_env)
            )
        else:
            environment = freeze_runtime_env(
                dict(self.resources.runtime_env)
            )
        resources = ResourceSpec(
            cast(int, options.get("replicas", self.resources.replicas)),
            cast(float, options.get("num_cpus", self.resources.num_cpus)),
            cast(float, options.get("num_gpus", self.resources.num_gpus)),
            environment,
        )
        failures = options.get("failure_policy", self.failure_policy)
        if not isinstance(failures, FailurePolicy):
            raise TypeError("failure_policy must be a FailurePolicy")
        return self._copy(
            batch_policy=batch,
            resources=resources,
            failure_policy=failures,
        )

    def __call__(
        self,
        *args: RUNP.args,
        **kwargs: RUNP.kwargs,
    ) -> Any:
        """Append one symbolic MAP call and mirror its annotated return."""

        return trace_map_call(self, tuple(args), dict(kwargs))

    def _copy(self, **changes: object) -> "Map[INITP, RUNP, R]":
        """Clone this frozen spec without routing through its public constructor."""

        clone = object.__new__(type(self))
        for name in (
            "op_cls",
            "batch_policy",
            "resources",
            "failure_policy",
            "init_args",
            "init_kwargs",
        ):
            object.__setattr__(
                clone,
                name,
                changes.get(name, getattr(self, name)),
            )
        return clone
def filter(mask: object, target: object, /):
    """Keep ``target`` where the aligned strict-bool ``mask`` is true."""

    return trace_filter(mask, target)


def expand(group: object, /):
    """Expose exactly one structural list layer and open a dynamic scope."""

    return trace_expand(group)


def reduce(items: object, /):
    """Close the path-local top scope and collect PRESENT items by ordinal."""

    return trace_reduce(items)


__all__ = [
    "CompileError",
    "ExecutorSession",
    "FailurePolicy",
    "Map",
    "OpaqueValue",
    "ParamSpec",
    "Pipeline",
    "RunOp",
    "RunStream",
    "expand",
    "filter",
    "freeze_config_value",
    "reduce",
    "thaw_config_value",
]
