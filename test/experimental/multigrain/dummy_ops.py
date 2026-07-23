"""Importable dummy operators for multigrain Ray tests.

These live in a real importable module (not a pytest-collected test file) so that
Ray workers can reconstruct them from the ``OperatorRecipe.cls_ref`` stored in
the IR. Operators sleep per row so that parallelism shows up in wall-clock time.
"""
from __future__ import annotations

import time

from rayorch.runtime import BadRecordError

SLEEP = 0.2


class SlowEmbed:
    """Map op: sleep per row, then embed."""

    def run(self, chunks: list[str]) -> list[str]:
        for _ in chunks:
            time.sleep(SLEEP)
        return [f"emb:{chunk}" for chunk in chunks]


class SlowDrop:
    """Filter op: sleep per row, drop chunks containing 'x'."""

    def run(self, chunks: list[str]) -> list[bool]:
        for _ in chunks:
            time.sleep(SLEEP)
        return ["x" not in chunk for chunk in chunks]


class InitCountingEmbed:
    """Map op whose constructor reports actor-local initialization once."""

    def __init__(self, counter_name: str) -> None:
        import ray

        self._counter = ray.get_actor(counter_name)
        ray.get(self._counter.add.remote(1))

    def run(self, chunks: list[str]) -> list[str]:
        return [f"counted:{chunk}" for chunk in chunks]


class MergeColumns:
    """Map op used to exercise a two-branch DAG fan-in."""

    def run(self, left: list[str], right: list[str]) -> list[str]:
        return [f"{a}|{b}" for a, b in zip(left, right)]


class OpaquePoisonMap:
    """Fail an invocation without identifying the poison row."""

    def __init__(self, poison: str = "bad") -> None:
        self.poison = poison

    def run(self, rows: list[str]) -> list[str]:
        if self.poison in rows:
            raise RuntimeError(f"opaque poison: {self.poison}")
        return [f"ok:{row}" for row in rows]


class AlwaysOpaqueFail:
    def run(self, rows: list[str]) -> list[str]:
        raise RuntimeError(f"systemic opaque failure for {len(rows)} rows")


class RetryableOnceMap:
    def __init__(self) -> None:
        self.seen: set[str] = set()

    def run(self, rows: list[str]) -> list[str]:
        for index, row in enumerate(rows):
            if row.startswith("flaky") and row not in self.seen:
                self.seen.add(row)
                raise BadRecordError("temporary", index=index, retryable=True)
        return [f"ok:{row}" for row in rows]


class AlwaysRetryableBadMap:
    def run(self, rows: list[str]) -> list[str]:
        for index, row in enumerate(rows):
            if row.startswith("bad"):
                raise BadRecordError("still temporary", index=index, retryable=True)
        return [f"ok:{row}" for row in rows]


class TagSecondary:
    def run(self, rows: list[str]) -> list[str]:
        return [f"side:{row}" for row in rows]


class RetryableOnceMerge:
    def __init__(self) -> None:
        self.seen: set[str] = set()

    def run(self, rows: list[str], secondary: list[str]) -> list[str]:
        for index, row in enumerate(rows):
            if row.startswith("flaky") and row not in self.seen:
                self.seen.add(row)
                raise BadRecordError("temporary", index=index, retryable=True)
        return [
            f"ok:{row}|{side}"
            for row, side in zip(rows, secondary)
        ]
