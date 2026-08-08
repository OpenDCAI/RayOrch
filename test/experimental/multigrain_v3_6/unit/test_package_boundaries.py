"""Keep the V3.6 directory layout as an enforced dependency architecture."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = (
    Path(__file__).parents[4]
    / "rayorch"
    / "experimental"
    / "multigrain_v3_6"
)
PACKAGE_NAME = ("rayorch", "experimental", "multigrain_v3_6")


def _local_imports(path: Path) -> tuple[tuple[str, ...], ...]:
    """Resolve local absolute/relative imports to paths below multigrain_v3_6."""

    relative = path.relative_to(PACKAGE).with_suffix("")
    package = (*PACKAGE_NAME, *relative.parts[:-1])
    targets: list[tuple[str, ...]] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = (alias.name.split(".") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = package[: len(package) - node.level + 1]
                names = ((*prefix, *(node.module or "").split(".")),)
            else:
                names = ((node.module or "").split("."),)
        else:
            continue
        for name in names:
            parts = tuple(part for part in name if part)
            if parts[: len(PACKAGE_NAME)] == PACKAGE_NAME:
                targets.append(parts[len(PACKAGE_NAME) :])
    return tuple(targets)


def test_v36_package_dependency_boundaries() -> None:
    violations: list[str] = []
    foundation = {"model.py", "protocol.py", "recovery.py"}
    upper_layers = {"api", "functional", "program", "runtime", "execution"}

    for path in PACKAGE.rglob("*.py"):
        if "benchmark" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PACKAGE)
        source_layer = relative.parts[0]
        for target in _local_imports(path):
            if not target:
                continue
            target_layer = target[0]
            reason: str | None = None

            if relative.name in foundation and target_layer in upper_layers:
                reason = "foundation contract imports an upper layer"
            elif source_layer == "program" and target_layer in {
                "api",
                "functional",
                "runtime",
                "execution",
            }:
                reason = "static Program imports an authoring/runtime/execution layer"
            elif source_layer == "runtime":
                if target_layer in {"api", "functional", "execution"}:
                    reason = "runtime imports authoring or physical execution"
                elif target_layer == "program" and target[:2] != ("program", "plan"):
                    reason = "runtime may consume RuntimePlan, not compiler internals"
            elif relative.as_posix() == "execution/worker.py" and target_layer not in {
                "model",
                "protocol",
            }:
                reason = "value-only Worker imports Program, runtime, or Ray driver state"
            elif relative.name in {"api.py", "functional.py"} and target_layer in {
                "runtime",
                "execution",
            }:
                reason = "authoring API imports dynamic or physical execution state"

            if reason is not None:
                violations.append(f"{relative} -> {'.'.join(target)}: {reason}")

    assert not violations, "\n".join(violations)


def test_program_phase_exports_are_deliberate() -> None:
    """Compiler is the facade; verifier has one path; lowering stays internal."""

    from rayorch.experimental.multigrain_v3_6.program import compiler, lowering, verify

    assert compiler.__all__ == ["compile_logical"]
    assert not hasattr(compiler, "verify_logical")
    assert not hasattr(compiler, "verify_runtime_plan")
    assert verify.__all__ == ["verify_logical", "verify_runtime_plan"]
    assert lowering.__all__ == []
