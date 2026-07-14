"""Deterministic (no-GPU) proof that LPT rebalancing eliminates the bubble.

The GPU benches (`bench_gpu_longtail.py`, `bench_gpu_complex.py`) show the
wall-clock effect; this locks the underlying scheduling guarantee into CI without
timing flakiness. Because the GPU op is perfectly linear in work units, the
per-shard *load* computed here is exactly what determines the GPU makespan.

Bubble = 1 - efficiency = 1 - sum(w)/(R*max_shard_load). We assert LPT drives it
to ~0 (meets Graham's 4/3 bound, >=95% efficiency) on a long-tailed workload,
while contiguous equal-count sharding leaves a real bubble.
"""
from __future__ import annotations

import random

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.ray import lpt_shard_planner
from rayorch.experimental.multigrain.ray.executor import _contiguous_ranges

R = 4


def _long_tail_weights(n: int, alpha: float, seed: int, cap: int = 45) -> list[int]:
    rng = random.Random(seed)
    return [min(cap, max(1, int((rng.paretovariate(alpha) - 1) * 16))) for _ in range(n)]


def _loads(parts, weights):
    return [sum(weights[i] for i in part) for part in parts]


def _opt(weights):
    return max(sum(weights) / R, max(weights))


def test_lpt_meets_graham_bound_and_kills_bubble() -> None:
    weights = _long_tail_weights(44, alpha=1.2, seed=11)
    batch = mg.source(weights, name="w")

    lpt = lpt_shard_planner(lambda w: w)(None, [batch], R)
    lpt_make = max(_loads(lpt, weights))
    opt = _opt(weights)

    # Graham 1969 bound and near-perfect efficiency (bubble < 5%)
    assert lpt_make <= (4.0 / 3.0 - 1.0 / (3 * R)) * opt
    efficiency = sum(weights) / (R * lpt_make)
    assert efficiency >= 0.95


def test_contiguous_leaves_a_bubble_that_lpt_removes() -> None:
    weights = _long_tail_weights(28, alpha=1.15, seed=7)  # few items/shard -> high variance
    batch = mg.source(weights, name="w")

    contiguous = [list(rng) for rng in _contiguous_ranges(len(weights), R)]
    lpt = lpt_shard_planner(lambda w: w)(None, [batch], R)

    cont_make = max(_loads(contiguous, weights))
    lpt_make = max(_loads(lpt, weights))

    cont_bubble = 1.0 - sum(weights) / (R * cont_make)
    lpt_bubble = 1.0 - sum(weights) / (R * lpt_make)

    assert cont_bubble > 0.15  # naive equal-count sharding wastes GPUs
    assert lpt_bubble < 0.05  # LPT rebalancing removes it
    assert lpt_make < cont_make


def test_lpt_balances_survivors_after_a_data_dependent_filter() -> None:
    """Filter-induced skew (bubble source #3): LPT rebalances the survivor set."""
    weights = _long_tail_weights(60, alpha=1.2, seed=3)
    survivors = [w for w in weights if w >= 6]  # data-dependent drop
    assert 0 < len(survivors) < len(weights)  # the filter really dropped rows

    batch = mg.source(survivors, name="w")
    lpt = lpt_shard_planner(lambda w: w)(None, [batch], R)
    efficiency = sum(survivors) / (R * max(_loads(lpt, survivors)))
    assert efficiency >= 0.95
