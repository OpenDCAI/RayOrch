"""Executor event-loop progress and deadlock detection."""

from types import SimpleNamespace

import pytest

from rayorch._execution.executor import Executor


class _StalledEngine:
    """An incomplete Engine with no runnable or in-flight work."""

    def __init__(self) -> None:
        self.completion_checks = 0

    def is_complete(self) -> bool:
        self.completion_checks += 1
        if self.completion_checks > 1:
            raise AssertionError("Executor spun instead of reporting deadlock")
        return False

    def progress_summary(self) -> str:
        return "stalled"


def test_remaining_input_does_not_hide_deadlock_when_admission_is_full():
    engine = _StalledEngine()
    executor = object.__new__(Executor)
    executor._closed = False
    executor.plan = SimpleNamespace(source_ports=(object(),), calls=())
    executor.store = SimpleNamespace(clear_cache=lambda: None)
    executor._actors = {}
    executor._admit_input_batch = lambda columns: engine
    executor._dispatch_ready = lambda active, pending: False
    executor.close = lambda: None

    with pytest.raises(
        RuntimeError,
        match=r"RayOrch runtime deadlocked: input_batch\[0\] stalled",
    ):
        executor.run(
            [1, 2],
            input_batch_size=1,
            max_active_input_batches=1,
        )

    assert engine.completion_checks == 1
