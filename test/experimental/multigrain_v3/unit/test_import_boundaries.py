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
    assert "api" not in imports


def test_reduce_accumulator_is_pure_and_ray_free():
    """Hierarchical shape logic owns no Arena tables, Driver, or Ray."""

    reduce_module = ROOT / "arena" / "reduce.py"
    imports = _local_imports(reduce_module)
    assert not imports & {"engine", "driver", "execution", "worker"}
    assert "ray" not in _absolute_import_roots(reduce_module)


def test_lower_layers_depend_on_contracts_not_authoring_api():
    """Arena/Execution/Worker 共享底层合同，但不能反向依赖 authoring API。"""

    lower_layers = (
        ROOT / "arena" / "state.py",
        ROOT / "arena" / "engine.py",
        ROOT / "execution.py",
        ROOT / "worker.py",
    )
    for path in lower_layers:
        assert "api" not in _local_imports(path), path


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


def test_v3_classes_and_functions_have_chinese_docstrings():
    """V3 新增类/函数必须留下中文功能与设计说明，避免维护语义漂移。"""

    missing = []
    english_only = []
    for path in ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue
            if node.name.startswith("__") and node.name not in {
                "__init__",
                "__post_init__",
            }:
                continue
            docstring = ast.get_docstring(node)
            label = f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}"
            if not docstring:
                missing.append(label)
            elif not any("\u4e00" <= char <= "\u9fff" for char in docstring):
                english_only.append(label)
    assert missing == []
    assert english_only == []
