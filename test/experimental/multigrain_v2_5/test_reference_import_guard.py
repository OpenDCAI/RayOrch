from __future__ import annotations

import ast
import importlib
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = Path(__file__).with_name("reference_semantics")
PRODUCTION = ROOT / "rayorch" / "experimental" / "multigrain_v2_5"


def _imports(path: Path) -> set[str]:
    imported: set[str] = set()
    for source in path.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    return imported


def test_reference_source_has_no_production_imports():
    assert not any(
        name.startswith("rayorch.experimental.multigrain")
        for name in _imports(REFERENCE)
    )


def test_production_source_has_no_test_or_reference_imports():
    if not PRODUCTION.exists():
        return
    assert not any(
        name.startswith("test.")
        or "reference_semantics" in name
        for name in _imports(PRODUCTION)
    )


class _ProductionBlocker(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("rayorch.experimental.multigrain"):
            raise ImportError(f"blocked production import: {fullname}")
        return None


class _ReferenceBlocker(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if "multigrain_v2_5.reference_semantics" in fullname:
            raise ImportError(f"blocked reference import: {fullname}")
        return None


def test_reference_package_loads_when_all_production_imports_are_blocked():
    prefix = "test.experimental.multigrain_v2_5.reference_semantics"
    saved = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == prefix or name.startswith(prefix + ".")
    }
    for name in saved:
        sys.modules.pop(name, None)

    blocker = _ProductionBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        loaded = importlib.import_module(prefix)
        assert loaded.ReferenceInterpreter is not None
    finally:
        sys.meta_path.remove(blocker)
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_production_package_loads_when_reference_imports_are_blocked():
    if not PRODUCTION.exists():
        return
    prefix = "rayorch.experimental.multigrain_v2_5"
    saved = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == prefix or name.startswith(prefix + ".")
    }
    for name in saved:
        sys.modules.pop(name, None)

    blocker = _ReferenceBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        loaded = importlib.import_module(prefix)
        assert loaded.GrainRecord is not None
    finally:
        sys.meta_path.remove(blocker)
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
