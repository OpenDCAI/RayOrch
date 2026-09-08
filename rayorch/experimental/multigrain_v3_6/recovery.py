"""Call-scoped recovery configuration and its pure decision algebra.

This module is deliberately Ray-free and runtime-free. It never stores an
attempt, DispatchBatch, queue, or generation; callers provide immutable facts
and receive one action from a closed set.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import assert_never


class UdfRecoveryMode(Enum):
    """Recovery modes available for an opaque whole-dispatch UDF failure."""

    ABORT = auto()
    RETRY_BATCH = auto()
    RETRY_TAIL = auto()
    ISOLATE_TAIL = auto()


class RecoveryAction(Enum):
    """Exhaustive actions produced for one opaque UDF dispatch failure."""

    ABORT = auto()
    RETRY_IMMEDIATE = auto()
    RETRY_TAIL = auto()
    SPLIT_TAIL = auto()
    FAIL_SINGLETON = auto()


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    """Immutable recovery contract compiled onto exactly one RayModule Call."""

    udf_mode: UdfRecoveryMode = UdfRecoveryMode.ABORT
    udf_attempts: int = 0
    infra_retries: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.udf_mode, UdfRecoveryMode):
            raise TypeError("udf_mode must be a UdfRecoveryMode")
        if type(self.infra_retries) is not int:
            raise TypeError("infra_retries must be an integer")
        if type(self.udf_attempts) is not int:
            raise TypeError("udf_attempts must be an integer")
        if self.infra_retries < 0:
            raise ValueError("infra_retries must be non-negative")
        if self.udf_mode is UdfRecoveryMode.ABORT:
            if self.udf_attempts != 0:
                raise ValueError("abort recovery does not accept UDF attempts")
        elif self.udf_mode in {
            UdfRecoveryMode.RETRY_BATCH,
            UdfRecoveryMode.RETRY_TAIL,
        }:
            if self.udf_attempts <= 0:
                raise ValueError("retry recovery requires positive attempts")
        elif self.udf_mode is UdfRecoveryMode.ISOLATE_TAIL:
            if self.udf_attempts != 1:
                raise ValueError(
                    "isolate_tail performs exactly one whole-DispatchBatch "
                    "deferred retry"
                )

    @classmethod
    def abort(cls, *, infra_retries: int = 1) -> "RecoveryPolicy":
        """Abort the run after any opaque UDF dispatch failure."""

        return cls(UdfRecoveryMode.ABORT, 0, infra_retries)

    @classmethod
    def retry_batch(
        cls,
        *,
        attempts: int = 1,
        infra_retries: int = 1,
    ) -> "RecoveryPolicy":
        """Retry the same DispatchBatch through the immediate-retry queue."""

        return cls(UdfRecoveryMode.RETRY_BATCH, attempts, infra_retries)

    @classmethod
    def retry_tail(
        cls,
        *,
        attempts: int = 1,
        infra_retries: int = 1,
    ) -> "RecoveryPolicy":
        """Retry the same DispatchBatch through the deferred-recovery queue."""

        return cls(UdfRecoveryMode.RETRY_TAIL, attempts, infra_retries)

    @classmethod
    def isolate_tail(cls, *, infra_retries: int = 1) -> "RecoveryPolicy":
        """Defer one retry, then split a still-failing DispatchBatch to singletons."""

        return cls(UdfRecoveryMode.ISOLATE_TAIL, 1, infra_retries)

    def decide_udf(
        self,
        *,
        completed_retries: int,
        grain_count: int,
    ) -> RecoveryAction:
        """Reduce immutable attempt facts to one side-effect-free action."""

        if type(completed_retries) is not int or completed_retries < 0:
            raise ValueError("completed_retries must be a non-negative integer")
        if type(grain_count) is not int or grain_count <= 0:
            raise ValueError("UDF recovery requires a non-empty DispatchBatch")
        match self.udf_mode:
            case UdfRecoveryMode.ABORT:
                return RecoveryAction.ABORT
            case UdfRecoveryMode.RETRY_BATCH | UdfRecoveryMode.RETRY_TAIL as mode:
                if completed_retries >= self.udf_attempts:
                    return RecoveryAction.ABORT
                return (
                    RecoveryAction.RETRY_IMMEDIATE
                    if mode is UdfRecoveryMode.RETRY_BATCH
                    else RecoveryAction.RETRY_TAIL
                )
            case UdfRecoveryMode.ISOLATE_TAIL:
                if completed_retries == 0:
                    return RecoveryAction.RETRY_TAIL
                if grain_count == 1:
                    return RecoveryAction.FAIL_SINGLETON
                return RecoveryAction.SPLIT_TAIL
            case unreachable:
                assert_never(unreachable)

    def allows_infrastructure_retry(
        self,
        completed_retries: tuple[int, ...],
    ) -> bool:
        """Return whether every Grain in one DispatchBatch retains retry budget."""

        if not completed_retries:
            raise ValueError(
                "infrastructure recovery requires a non-empty DispatchBatch"
            )
        if any(type(count) is not int or count < 0 for count in completed_retries):
            raise ValueError("infrastructure retry counts must be non-negative integers")
        return all(count < self.infra_retries for count in completed_retries)


DEFAULT_RECOVERY_POLICY = RecoveryPolicy.abort()


__all__ = [
    "DEFAULT_RECOVERY_POLICY",
    "RecoveryPolicy",
    "RecoveryAction",
    "UdfRecoveryMode",
]
