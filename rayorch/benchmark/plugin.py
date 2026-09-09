"""Lightweight benchmark metadata shared by all benchmark groups."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, util
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkPlugin:
    """Import-free description of one UDF group and its runnable entrypoint."""

    name: str
    runner_module: str
    runtime_package: str
    runtime_resource: str = "runtime_env.json"
    required_modules: tuple[str, ...] = ()

    def load_runner(self) -> Any:
        return import_module(self.runner_module)

    def missing_dependencies(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.required_modules if util.find_spec(name) is None
        )


__all__ = ["BenchmarkPlugin"]
