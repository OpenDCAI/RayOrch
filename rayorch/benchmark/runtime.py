"""Runtime-environment loading for independently deployable UDF groups."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Mapping

from .plugin import BenchmarkPlugin


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeEnv:
    plugin: str
    value: dict[str, Any]
    digest: str


def _expand(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        expanded = value
        for name, replacement in variables.items():
            expanded = expanded.replace("${" + name + "}", replacement)
        return os.path.expandvars(expanded)
    if isinstance(value, list):
        return [_expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, variables) for key, item in value.items()}
    return value


def load_runtime_env(
    plugin: BenchmarkPlugin,
    *,
    variables: Mapping[str, str] | None = None,
    py_modules: Iterable[str | Path] = (),
    extra_pip: Iterable[str] = (),
) -> ResolvedRuntimeEnv:
    """Load and deterministically extend one group-owned Ray runtime_env."""

    resource = files(plugin.runtime_package).joinpath(plugin.runtime_resource)
    value = json.loads(resource.read_text(encoding="utf-8"))
    if value.pop("schema_version", None) != 1:
        raise ValueError(f"unsupported runtime_env schema in {resource}")
    value = _expand(value, variables or {})
    unresolved = [
        item
        for item in json.dumps(value).split('"')
        if "${" in item
    ]
    if unresolved:
        raise ValueError(f"unresolved runtime_env variables: {unresolved}")

    modules = [str(Path(path).resolve()) for path in py_modules]
    if modules:
        value["py_modules"] = modules
    additions = list(extra_pip)
    if additions:
        pip = value.setdefault("pip", {"packages": [], "pip_check": False})
        if not isinstance(pip, dict) or not isinstance(pip.get("packages"), list):
            raise TypeError("runtime_env pip must use Ray's dictionary form")
        pip["packages"].extend(additions)

    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return ResolvedRuntimeEnv(
        plugin=plugin.name,
        value=value,
        digest=hashlib.sha256(canonical.encode()).hexdigest(),
    )


__all__ = ["ResolvedRuntimeEnv", "load_runtime_env"]
