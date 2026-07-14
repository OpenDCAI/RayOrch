"""Passive execution and recovery policies attached to graph nodes."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping


class RetryTiming(str, Enum):
    INLINE = "inline"
    DEFERRED = "deferred"


class ShardExhaustedAction(str, Enum):
    ABORT = "abort"
    DEGRADE = "degrade"


class IsolationExhaustedAction(str, Enum):
    QUARANTINE = "quarantine"
    ABORT = "abort"


class RecordRecoveryAction(str, Enum):
    RETRY = "retry_record"
    ISOLATE = "isolate_record"


class ShardRecoveryAction(str, Enum):
    RETRY = "retry_shard"
    DEGRADE = "degrade_shard"
    ABORT = "abort"


@dataclass(frozen=True)
class WorkerPoolSpec:
    """Worker replicas and per-worker GPU allocation consumed by Ray."""

    replicas: int = 1
    gpus_per_worker: float = 0.0

    def __post_init__(self) -> None:
        if self.replicas < 1:
            raise ValueError("replicas must be >= 1")
        gpus = float(self.gpus_per_worker)
        if not math.isfinite(gpus) or gpus < 0:
            raise ValueError("gpus_per_worker must be finite and >= 0")
        object.__setattr__(self, "gpus_per_worker", gpus)


@dataclass(frozen=True)
class IsolationBudget:
    """Hard bound for adaptive record localization after shard failure."""

    max_work_factor: float = 3.0
    max_calls: int = 64
    on_exhausted: IsolationExhaustedAction = IsolationExhaustedAction.QUARANTINE

    def __post_init__(self) -> None:
        factor = float(self.max_work_factor)
        if not math.isfinite(factor) or factor < 0:
            raise ValueError("max_work_factor must be finite and >= 0")
        if self.max_calls < 0:
            raise ValueError("max_calls must be >= 0")
        object.__setattr__(self, "max_work_factor", factor)
        object.__setattr__(
            self,
            "on_exhausted",
            IsolationExhaustedAction(self.on_exhausted),
        )


@dataclass(frozen=True)
class RecoveryPolicy:
    """Retry and isolation bounds for one graph node."""

    max_record_retries: int = 0
    retry_timing: RetryTiming = RetryTiming.INLINE
    max_shard_retries: int = 2
    on_shard_exhausted: ShardExhaustedAction = ShardExhaustedAction.ABORT
    isolation: IsolationBudget = field(default_factory=IsolationBudget)

    def __post_init__(self) -> None:
        if self.max_record_retries < 0:
            raise ValueError("max_record_retries must be >= 0")
        if self.max_shard_retries < 0:
            raise ValueError("max_shard_retries must be >= 0")
        object.__setattr__(self, "retry_timing", RetryTiming(self.retry_timing))
        object.__setattr__(
            self,
            "on_shard_exhausted",
            ShardExhaustedAction(self.on_shard_exhausted),
        )
        if isinstance(self.isolation, Mapping):
            object.__setattr__(self, "isolation", IsolationBudget(**self.isolation))
        elif not isinstance(self.isolation, IsolationBudget):
            raise TypeError("isolation must be an IsolationBudget or mapping")

    def decide_record(
        self,
        *,
        retryable: bool,
        attempt: int,
    ) -> RecordRecoveryAction:
        if retryable and attempt < self.max_record_retries:
            return RecordRecoveryAction.RETRY
        return RecordRecoveryAction.ISOLATE

    def decide_shard(self, *, attempt: int) -> ShardRecoveryAction:
        if attempt < self.max_shard_retries:
            return ShardRecoveryAction.RETRY
        if self.on_shard_exhausted is ShardExhaustedAction.DEGRADE:
            return ShardRecoveryAction.DEGRADE
        return ShardRecoveryAction.ABORT


__all__ = [
    "IsolationBudget",
    "IsolationExhaustedAction",
    "RecordRecoveryAction",
    "RecoveryPolicy",
    "RetryTiming",
    "ShardExhaustedAction",
    "ShardRecoveryAction",
    "WorkerPoolSpec",
]
