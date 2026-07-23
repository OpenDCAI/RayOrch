from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


REFERENCE_ROOT = Path(__file__).parent / "reference_semantics"
FORBIDDEN = (
    "rayorch.experimental.multigrain",
    "rayorch.experimental.multigrain_v2",
)


def test_reference_oracle_has_no_production_imports() -> None:
    for path in REFERENCE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            name.startswith(FORBIDDEN) for name in imported
        ), f"{path} imports production code: {imported}"


def test_reference_oracle_imports_with_production_modules_blocked() -> None:
    script = r'''
import importlib.abc
import sys

class RejectProduction(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "rayorch.experimental.multigrain"
            or fullname.startswith("rayorch.experimental.multigrain.")
            or fullname == "rayorch.experimental.multigrain_v2"
            or fullname.startswith("rayorch.experimental.multigrain_v2.")
        ):
            raise ImportError("production Multigrain import rejected: " + fullname)
        return None

sys.meta_path.insert(0, RejectProduction())
from test.experimental.multigrain_v2.reference_semantics import identity_oracle
from test.experimental.multigrain_v2.reference_semantics import graph_oracle
from test.experimental.multigrain_v2.reference_semantics import interpreter
from test.experimental.multigrain_v2.reference_semantics import semantic_cases
assert identity_oracle.vectors()
assert graph_oracle.project_relation(
    {"kind": "map", "roles": ("x",), "inputs": ("p",)}, 0
)["kind"] == "same_as"
assert interpreter.interpret_expand("x", ("p",), (1,)).entities
assert semantic_cases.CASES
assert not any(
    name == "rayorch.experimental.multigrain"
    or name.startswith("rayorch.experimental.multigrain.")
    or name == "rayorch.experimental.multigrain_v2"
    or name.startswith("rayorch.experimental.multigrain_v2.")
    for name in sys.modules
)
'''
    subprocess.run([sys.executable, "-c", script], check=True)
