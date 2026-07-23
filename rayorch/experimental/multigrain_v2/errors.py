"""Public exception contracts for Multigrain V2.2."""
from __future__ import annotations

from typing import Protocol


class _ExecutionSummary(Protocol):
    code: str
    message: str
    causes: tuple[str, ...]


class MultigrainError(Exception):
    """Base class for all public Multigrain failures."""


class CompileError(MultigrainError):
    """A stable compilation failure with an optional authoring path."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path


class ExecutionError(MultigrainError):
    """A stable execution failure detached from runtime implementation types."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        batch_id: str | None = None,
        node_id: str | None = None,
        causes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.batch_id = batch_id
        self.node_id = node_id
        self.causes = tuple(causes)

    @classmethod
    def from_summary(
        cls,
        summary: _ExecutionSummary | None,
    ) -> "ExecutionError":
        if summary is None:
            return cls(
                "execution aborted without an error summary",
                code="UNKNOWN_EXECUTION_ERROR",
            )
        return cls(
            summary.message,
            code=summary.code,
            causes=tuple(summary.causes),
        )


class BadRecordError(MultigrainError):
    """An explicitly attributable bad semantic item."""

    def __init__(self, message: str, *, index: int | None = None) -> None:
        super().__init__(message)
        if index is not None and (type(index) is not int or index < 0):
            raise ValueError("BadRecordError index must be a non-negative int")
        self.index = index


__all__ = [
    "BadRecordError",
    "CompileError",
    "ExecutionError",
    "MultigrainError",
]
