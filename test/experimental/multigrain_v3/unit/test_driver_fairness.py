"""RunDriver Arena-major 提交顺序的 Ray-free 回归测试。"""

from __future__ import annotations

from types import SimpleNamespace

from rayorch.experimental.multigrain_v3.driver import RunDriver


class _FakeArena:
    def __init__(self, arena: int, calls: int) -> None:
        self.arena = arena
        self.calls = calls
        self.live_block_count = 0
        self.advanced = 0

    def advance(self, now: float) -> None:
        del now
        self.advanced += 1

    def reserve_dispatch(self, stage: int, now: float):
        del stage, now
        if self.calls == 0:
            return None
        self.calls -= 1
        return self.arena


class _FakeExecution:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.submitted: list[int] = []

    def can_submit(self, stage: int) -> bool:
        del stage
        return len(self.submitted) < self.capacity

    def submit(self, intent: int) -> bool:
        self.submitted.append(intent)
        return True


def _driver(capacity: int = 6) -> RunDriver:
    driver = object.__new__(RunDriver)
    driver.active = {
        index: _FakeArena(index, calls=3)
        for index in range(3)
    }
    driver.compiled = SimpleNamespace(
        dag=SimpleNamespace(
            stages=(SimpleNamespace(id=1, execution=object()),)
        )
    )
    driver.execution = _FakeExecution(capacity)
    driver.live_blocks_high_watermark = 0
    return driver


def test_submit_uses_actor_credits_in_arena_order() -> None:
    """较早 Arena 的 ready dispatch 先使用共享 stage credits。"""

    driver = _driver()

    assert driver._submit()
    assert driver.execution.submitted == [0, 0, 0, 1, 1, 1]
    assert all(arena.advanced == 1 for arena in driver.active.values())
