"""Lightweight benchmark metadata shared by all benchmark groups."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


@dataclass(frozen=True, slots=True)
class BenchmarkPlugin:
    """Import-free description of one UDF group and its runnable entrypoint."""

    name: str
    runner_module: str
    runtime_package: str
    runtime_resource: str = "runtime_env.json"

    def load_runner(self) -> ModuleType:
        """Import the heavy runner only when execution is requested."""

        return import_module(self.runner_module)


__all__ = ["BenchmarkPlugin"]
