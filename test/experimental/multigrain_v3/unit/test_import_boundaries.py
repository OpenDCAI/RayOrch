"""Architectural import boundaries for the four-component V3 model."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path("rayorch/experimental/multigrain_v3")


def _local_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[-1])
    return imports


def _absolute_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_dag_is_static_and_does_not_import_runtime_components():
    """CompiledDAG must remain independent of Arena, Driver, and Ray."""

    imports = _local_imports(ROOT / "dag.py")
    assert not imports & {"arena", "driver", "execution", "worker"}
    assert "ray" not in _absolute_import_roots(ROOT / "dag.py")


def test_execution_and_worker_do_not_import_arena_internals():
    """StageExecutor/worker communicate only through stable protocol DTOs."""

    assert "arena" not in _local_imports(ROOT / "execution.py")
    assert "arena" not in _local_imports(ROOT / "worker.py")
    assert "protocol" in _local_imports(ROOT / "execution.py")
    assert "protocol" in _local_imports(ROOT / "worker.py")


def test_arena_engine_does_not_import_ray_or_stage_executor():
    """ArenaEngine owns state transitions but never calls Ray."""

    engine = ROOT / "arena" / "engine.py"
    imports = _local_imports(engine)
    assert "execution" not in imports
    assert "ray" not in _absolute_import_roots(engine)


def test_driver_does_not_reach_into_arena_tables():
    """RunDriver may use ArenaEngine methods/properties, not internal tables."""

    tree = ast.parse((ROOT / "driver.py").read_text())
    forbidden = {
        "grains",
        "items",
        "values",
        "blocks",
        "queues",
        "reduce_accumulators",
        "pending_invocations",
        "leases",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr not in forbidden:
            continue
        if isinstance(node.value, ast.Name) and node.value.id == "arena":
            raise AssertionError(f"RunDriver reads arena.{node.attr}")
