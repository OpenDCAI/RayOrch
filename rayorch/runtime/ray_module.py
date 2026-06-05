"""RayModule adapter for the rowwise runtime."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from ..ray_module import RayModule, _infer_num_outputs
from .core import (
    LineageStore,
    MicroBatch,
    RuntimeNodeSpec,
    RuntimeResult,
    merge_runtime_results,
    run_rowwise,
)


def dispatch_microbatch_shard_contiguous(rm, *args, **kwargs):
    """Shard MicroBatch values by row and broadcast non-MicroBatch values."""
    replicas = rm._replicas

    def split_ranges(n: int) -> List[tuple[int, int]]:
        base, rem = divmod(n, replicas)
        start = 0
        ranges: List[tuple[int, int]] = []
        for rank in range(replicas):
            end = start + base + (1 if rank < rem else 0)
            ranges.append((start, end))
            start = end
        return ranges

    ranges: List[tuple[int, int]] | None = None
    for value in (*args, *kwargs.values()):
        if not isinstance(value, MicroBatch):
            continue
        current = split_ranges(len(value))
        if ranges is None:
            ranges = current
        elif current != ranges:
            raise ValueError("all MicroBatch inputs must have the same row count")

    def shard(value: Any) -> List[Any]:
        if isinstance(value, MicroBatch):
            return [value.slice(start, end) for start, end in ranges or []]
        return [value] * replicas

    return tuple(shard(arg) for arg in args), {
        key: shard(value) for key, value in kwargs.items()
    }


def collect_runtime_results(rm, outputs: Sequence[RuntimeResult]) -> RuntimeResult:
    """Collect replica-local RuntimeResult objects in rank order."""
    return merge_runtime_results(outputs)


class _RowwiseRuntimeActorOp:
    """Actor-local wrapper around one user op instance."""

    def __init__(
        self,
        op_cls: Type[Any],
        init_args: Tuple[Any, ...],
        init_kwargs: Dict[str, Any],
    ) -> None:
        self.op = op_cls(*init_args, **init_kwargs)

    def run(self, batch: MicroBatch, spec: RuntimeNodeSpec) -> RuntimeResult:
        lineage = LineageStore()
        out, bad = run_rowwise(
            self.op.run,
            batch,
            op=spec.node,
            inputs=spec.inputs,
            outputs=spec.outputs,
            lineage=lineage,
        )
        return lineage.result(out, bad)


class RuntimeRayModule(RayModule):
    """RayModule whose replicas run rowwise fault isolation inside each actor."""

    def __init__(
        self,
        op_cls: Type[Any],
        *,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
        op: Optional[str] = None,
        env: Optional[str] = None,
        replicas: int = 1,
        num_gpus_per_replica: float = 0.0,
        max_inflight: int = 1,
        num_outputs: int | None = None,
        dev_mode: bool = False,
    ) -> None:
        self._user_op_cls = op_cls
        self._op_name = op or op_cls.__name__
        self._runtime_spec: RuntimeNodeSpec | None = None
        if inputs is not None or outputs is not None:
            if inputs is None or outputs is None:
                raise ValueError("inputs and outputs must be provided together")
            self.bind_runtime_spec(
                RuntimeNodeSpec(
                    node=self._op_name,
                    inputs=tuple(inputs),
                    outputs=tuple(outputs),
                )
            )
        logical_num_outputs = (
            num_outputs
            if num_outputs is not None
            else len(outputs) if outputs is not None
            else _infer_num_outputs(op_cls)
        )
        super().__init__(
            _RowwiseRuntimeActorOp,
            env=env,
            replicas=replicas,
            num_gpus_per_replica=num_gpus_per_replica,
            dispatch_fn=dispatch_microbatch_shard_contiguous,
            collect_fn=collect_runtime_results,
            max_inflight=max_inflight,
            num_outputs=logical_num_outputs,
            dev_mode=dev_mode,
        )

    def pre_init(self, *args, **kwargs) -> "RuntimeRayModule":
        super().pre_init(
            self._user_op_cls,
            args,
            kwargs,
        )
        return self

    def bind_runtime_spec(self, spec: RuntimeNodeSpec) -> "RuntimeRayModule":
        """Bind DAG-inferred runtime ports.

        This is intentionally separate from ``pre_init()`` so a Pipeline compiler
        can fill the spec after tracing the DAG.
        """
        self._runtime_spec = spec
        return self

    def on_compile_node(self, spec: Any) -> None:
        """Receive a generic DAG compile hook and bind runtime ports."""
        self.bind_runtime_spec(
            RuntimeNodeSpec(
                node=spec.name,
                inputs=tuple(spec.input_names),
                outputs=tuple(spec.output_names),
            )
        )

    @property
    def runtime_spec(self) -> RuntimeNodeSpec | None:
        return self._runtime_spec

    def _require_runtime_spec(self) -> RuntimeNodeSpec:
        if self._runtime_spec is None:
            raise RuntimeError(
                "RuntimeRayModule requires a RuntimeNodeSpec. Pass inputs/outputs "
                "explicitly or call bind_runtime_spec() after DAG compile."
            )
        return self._runtime_spec

    def remote(self, batch: MicroBatch, spec: RuntimeNodeSpec | None = None):
        return super().remote(batch, spec or self._require_runtime_spec())
