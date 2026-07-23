"""Regression tests for low-cost batch-trigger observability."""

from __future__ import annotations

from .test_batch_trigger import _arena, _enqueue


def test_flush_reasons_fill_ratio_and_queue_high_watermark_are_exact():
    """Metrics must describe the physical RPC grouping without approximation."""

    _, arena, map_spec, sources = _arena(
        batch_size=4,
        wait_ms=100,
        count=6,
    )
    _enqueue(arena, map_spec, sources)
    full = arena.reserve_dispatch(1, admission_closed=False)
    tail = arena.reserve_dispatch(
        1,
        admission_closed=False,
        draining=True,
    )
    assert full is not None and tail is not None

    metrics = arena.metrics_snapshot()
    assert metrics["rpc_count"] == 2.0
    assert metrics["grains_per_rpc"] == 3.0
    assert metrics["batch_fill_ratio"] == 0.75
    assert metrics["ready_queue_high_watermark"] == 6.0
    assert metrics["flush_full"] == 1.0
    assert metrics["flush_arena_drain"] == 1.0
    assert metrics["flush_timeout"] == 0.0
    assert metrics["flush_isolation"] == 0.0
    assert metrics["tail_or_isolation_rpc_fraction"] == 0.5
