from __future__ import annotations

"""
Overlapped (graphless) microbatch pipeline.

We intentionally avoid an extra ``@ray.remote``/join task for each stage boundary.
``__call__()`` tracks ``ObjectRef``s from the **final** future per work item and uses
``ray.wait`` until all of those are ready, then :meth:`RayModule.RayModuleFuture.gather`
once (applying ``collect_fn``).

During ``forward``, each ``RayModule`` attribute is temporarily replaced by that
module's ``remote`` bound method (same as ``actor.method.remote``). Arguments that
are upstream :class:`RayModule.RayModuleFuture` values are unwrapped inside
:meth:`RayModule.remote` to a single ``ObjectRef`` (``completion_refs()[0]``) per
slot — see :meth:`RayModule.remote`. This matches ONE_TO_ALL-style identical shards;
otherwise prefer ``DagPipeline`` or explicit gathers.
"""

from collections import deque
from typing import Any, Deque, Dict, List, Sequence, Set, Tuple

import ray

from .ray_module import RayModule


class OverlappedPipeline:
    """
    Graphless overlapped runtime: each :meth:`submit_work` call submits one batch
    asynchronously; :meth:`__call__` drives a FIFO queue with an inflight cap.
    """

    def __init__(self, *, max_inflight: int = 8):
        self.max_inflight = max(1, int(max_inflight))

    def forward(self, work_item):
        raise NotImplementedError(
            "Subclass must implement forward(work_item) -> RayModuleFuture"
        )

    def submit_work(self, work_item: Any) -> RayModule.RayModuleFuture:
        originals: Dict[str, RayModule] = {}
        for name, value in list(self.__dict__.items()):
            if isinstance(value, RayModule):
                originals[name] = value
                setattr(self, name, value.remote)
        if not originals:
            raise ValueError("No attributes on pipeline object are RayModule instances")

        try:
            out = self.forward(work_item)
        finally:
            for name, value in originals.items():
                setattr(self, name, value)

        if not isinstance(out, RayModule.RayModuleFuture):
            raise TypeError(
                "forward must return a RayModuleFuture (e.g. tail module remote output)"
            )
        return out

    def __call__(self, inputs: Sequence[Any]) -> List[Any]:
        results: List[Any] = [None] * len(inputs)
        waiting: Deque[Tuple[int, Any]] = deque(enumerate(inputs))
        inflight_pending: Dict[int, RayModule.RayModuleFuture] = {}
        remaining_refs: Dict[int, int] = {}
        ref_to_idx: Dict[ray.ObjectRef, int] = {}
        outstanding: Set[ray.ObjectRef] = set()

        while waiting or inflight_pending:
            while waiting and len(inflight_pending) < self.max_inflight:
                idx, x = waiting.popleft()
                pending = self.submit_work(x)
                refs = pending.completion_refs()
                inflight_pending[idx] = pending
                remaining_refs[idx] = len(refs)
                for r in refs:
                    ref_to_idx[r] = idx
                    outstanding.add(r)

            if not outstanding:
                break

            done, _ = ray.wait(list(outstanding), num_returns=1)
            dr = done[0]
            outstanding.discard(dr)
            idx = ref_to_idx.pop(dr, None)
            if idx is None:
                continue
            remaining_refs[idx] -= 1
            if remaining_refs[idx] == 0:
                pending = inflight_pending.pop(idx)
                remaining_refs.pop(idx, None)
                results[idx] = pending.gather()

        return results
