"""Thin Ray adapter around the value-only Worker and BlockStore contracts."""

from __future__ import annotations

from typing import Any

from ..protocol import (
    BlockRef,
    CallInputLayout,
    CallOutputLayout,
    GrainPlan,
    RowBinding,
)
from .worker import Worker, WorkerSnapshot


class _RayBlockStore:
    """Map coarse Ray ObjectRefs to rows without interpreting payload values."""

    def __init__(self, ray_module: Any) -> None:
        self._ray = ray_module
        self._cache: dict[Any, tuple[Any, ...]] = {}

    def clear_cache(self) -> None:
        """Drop dereferenced payloads; the microbatch owns ObjectRef lifetime."""

        self._cache.clear()

    def put(self, values: tuple[Any, ...]) -> BlockRef:
        """Store one output column as a coarse object-store block."""

        return BlockRef(self._ray.put(tuple(values)))

    def get(self, binding: RowBinding) -> Any:
        """Resolve one row only inside a Worker or final materialization."""

        block = binding.block
        if not isinstance(block, BlockRef):
            raise TypeError(f"_RayBlockStore received non-Ray block: {block!r}")
        payload = block.handle
        # A batch's RowBindings normally share one coarse block; dereference it
        # once per RPC/materialization turn instead of once per logical Item.
        if isinstance(payload, self._ray.ObjectRef):
            if payload not in self._cache:
                self._cache[payload] = self._ray.get(payload)
            payload = self._cache[payload]
        return payload[binding.row]


class _RayWorkerActor:
    """Persist one Ray-free Worker behind the exact same DTO ABI."""

    def __init__(
        self,
        target: Any,
        init_args: tuple[Any, ...],
        init_kwargs: tuple[tuple[str, Any], ...],
        input_layout: CallInputLayout,
    ) -> None:
        import ray  # pyright: ignore[reportMissingImports]

        self._store = _RayBlockStore(ray)
        self._worker = Worker(
            target,
            init_args,
            init_kwargs,
            input_layout=input_layout,
        )

    def ready(self) -> bool:
        """Respond only after UDF construction has completed."""

        return True

    def execute(
        self,
        grain_plans: tuple[GrainPlan, ...],
        layouts: tuple[CallOutputLayout, ...],
    ):
        """Execute one batch without reading Program or RuntimeState."""

        self._store.clear_cache()
        try:
            return self._worker.execute(grain_plans, layouts, self._store)
        finally:
            # Input blocks must not leak across actor RPCs. Output bindings are
            # returned to the driver and then owned by a microbatch lifecycle.
            self._store.clear_cache()

    def observe(self) -> WorkerSnapshot:
        """Return an observation-only snapshot without leaking the UDF."""

        return self._worker.observe()


__all__ = ["_RayBlockStore", "_RayWorkerActor"]
