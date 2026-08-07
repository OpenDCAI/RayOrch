"""Small dependency-free statistics shared by paired benchmark runners."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Iterable


def sample_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    """Report explicit sample dispersion without inventing variance for n=1."""

    rows = tuple(float(value) for value in values)
    if not rows:
        raise ValueError("sample summary requires at least one value")
    mean = statistics.mean(rows)
    variance = statistics.variance(rows) if len(rows) > 1 else None
    stddev = variance**0.5 if variance is not None else None
    return {
        "samples": len(rows),
        "mean": mean,
        "median": statistics.median(rows),
        "sample_variance": variance,
        "sample_stddev": stddev,
        "minimum": min(rows),
        "maximum": max(rows),
    }


def paired_timing_summary(
    v3_walls: Iterable[float],
    v36_walls: Iterable[float],
    orders: Iterable[str],
) -> dict[str, object]:
    """Summarize arm samples and within-trial deltas, including order bias."""

    old = tuple(float(value) for value in v3_walls)
    new = tuple(float(value) for value in v36_walls)
    trial_orders = tuple(orders)
    if not old or len(old) != len(new) or len(old) != len(trial_orders):
        raise ValueError("paired timing samples and orders must be non-empty/aligned")
    if any(value <= 0 for value in old + new):
        raise ValueError("paired timing samples must be positive")

    deltas = tuple(right - left for left, right in zip(old, new))
    relative = tuple(delta / left for delta, left in zip(deltas, old))
    speedups = tuple(left / right for left, right in zip(old, new))
    by_order: dict[str, list[float]] = defaultdict(list)
    for order, value in zip(trial_orders, relative):
        by_order[order].append(value)
    return {
        "v3_wall_s": sample_summary(old),
        "v36_wall_s": sample_summary(new),
        "paired_v36_minus_v3_s": sample_summary(deltas),
        "paired_v36_relative_change": sample_summary(relative),
        "paired_v36_speedup": sample_summary(speedups),
        "v36_faster_trials": sum(delta < 0 for delta in deltas),
        "v3_faster_trials": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "relative_change_by_order": {
            order: sample_summary(values)
            for order, values in sorted(by_order.items())
        },
    }


__all__ = ["paired_timing_summary", "sample_summary"]
