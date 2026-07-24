"""Deterministic synthetic fan-out and service-time generation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChildWork:
    parent: str
    ordinal: int
    delay_s: float
    keep: bool


@dataclass(frozen=True, slots=True)
class ParentWork:
    name: str
    children: tuple[ChildWork, ...]


@dataclass(frozen=True, slots=True)
class SyntheticWorkload:
    seed: int
    fanout_mode: str
    service_mode: str
    fanout_scale: float
    mean_service_s: float
    drop_probability: float
    parents: tuple[ParentWork, ...]

    @property
    def total_children(self) -> int:
        return sum(len(parent.children) for parent in self.parents)

    @property
    def kept_children(self) -> int:
        return sum(
            child.keep
            for parent in self.parents
            for child in parent.children
        )


def _fanout(randomizer: random.Random, mode: str, scale: float) -> int:
    if mode == "constant":
        return max(0, int(scale))
    if mode == "uniform":
        return randomizer.randint(0, max(1, int(2 * scale)))
    if mode == "lognormal":
        return max(
            0,
            int(randomizer.lognormvariate(math.log(max(scale, 1)), 0.8)),
        )
    if mode == "pareto":
        return max(
            0,
            int(scale * randomizer.paretovariate(2.0)) - int(scale),
        )
    if mode == "zipf":
        rank = randomizer.randint(1, 16)
        return max(0, int(scale * 8 / rank))
    raise ValueError(f"unknown fanout mode: {mode}")


def _delay(randomizer: random.Random, mode: str, mean_s: float) -> float:
    if mode == "constant":
        return mean_s
    if mode == "uniform":
        return randomizer.uniform(0.25 * mean_s, 1.75 * mean_s)
    if mode == "lognormal":
        return randomizer.lognormvariate(
            math.log(max(mean_s, 1e-6)),
            0.7,
        )
    raise ValueError(f"unknown service mode: {mode}")


def generate_workload(
    *,
    seed: int,
    parent_count: int,
    fanout_mode: str,
    fanout_scale: float,
    service_mode: str,
    mean_service_s: float,
    drop_probability: float = 0.0,
) -> SyntheticWorkload:
    """Generate one fixed-seed workload shared by both batching modes."""

    if parent_count < 0:
        raise ValueError("parent_count must be non-negative")
    if not 0.0 <= drop_probability <= 1.0:
        raise ValueError("drop_probability must be in [0, 1]")
    randomizer = random.Random(seed)
    parents = []
    for parent_index in range(parent_count):
        name = f"parent-{parent_index}"
        count = _fanout(randomizer, fanout_mode, fanout_scale)
        children = tuple(
            ChildWork(
                parent=name,
                ordinal=ordinal,
                delay_s=_delay(randomizer, service_mode, mean_service_s),
                keep=randomizer.random() >= drop_probability,
            )
            for ordinal in range(count)
        )
        parents.append(ParentWork(name, children))
    return SyntheticWorkload(
        seed=seed,
        fanout_mode=fanout_mode,
        service_mode=service_mode,
        fanout_scale=fanout_scale,
        mean_service_s=mean_service_s,
        drop_probability=drop_probability,
        parents=tuple(parents),
    )
