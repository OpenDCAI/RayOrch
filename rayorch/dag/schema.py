"""Internal type-schema helpers for DAG compilation."""
from __future__ import annotations

import inspect
from typing import Any, Dict, Mapping, Tuple, get_args, get_origin, get_type_hints

from ..ray_module import RayModule
from .graph import PipeRef


def safe_type_hints(value: Any) -> Dict[str, Any]:
    try:
        return get_type_hints(value)
    except Exception:
        return {}


def input_types(
    module: RayModule,
    args: Tuple[PipeRef, ...],
    kw_args: Dict[str, PipeRef],
) -> Tuple[Any, ...]:
    op_cls = getattr(module, "_user_op_cls", getattr(module, "_op_cls", None))
    run_fn = getattr(op_cls, "run", None)
    if run_fn is None:
        return (Any,) * (len(args) + len(kw_args))

    hints = safe_type_hints(run_fn)
    try:
        params = list(inspect.signature(run_fn).parameters.values())
        if params and params[0].name == "self":
            params = params[1:]
        positional = [
            param
            for param in params
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        types = [hints.get(param.name, Any) for param in positional[:len(args)]]
        types.extend(hints.get(name, Any) for name in kw_args)
        return tuple(types)
    except (TypeError, ValueError):
        return (Any,) * (len(args) + len(kw_args))


def return_types(value: Any, count: int) -> Tuple[Any, ...]:
    hint = safe_type_hints(value).get("return", Any)
    origin = get_origin(hint)
    if origin in (tuple, Tuple):
        parts = get_args(hint)
        if len(parts) == 2 and parts[1] is Ellipsis:
            return (parts[0],) * count
        if len(parts) != count:
            raise TypeError(
                f"{value.__qualname__} return annotation declares "
                f"{len(parts)} outputs, but the DAG node has {count}"
            )
        return tuple(parts)
    return (hint,) if count == 1 else (Any,) * count


def type_name(value: Any) -> str:
    return getattr(value, "__name__", str(value).replace("typing.", ""))


def types_compatible(actual: Any, expected: Any) -> bool:
    if _is_typevar(actual) or _is_typevar(expected):
        return True
    if actual is Any or expected is Any or actual == expected:
        return True
    actual_origin = get_origin(actual)
    expected_origin = get_origin(expected)
    if actual_origin != expected_origin or actual_origin is None:
        return False
    actual_args = get_args(actual)
    expected_args = get_args(expected)
    return len(actual_args) == len(expected_args) and all(
        types_compatible(left, right)
        for left, right in zip(actual_args, expected_args)
    )


def bind_typevars(template: Any, actual: Any, bindings: Dict[Any, Any]) -> None:
    if _is_typevar(template):
        if actual is not Any:
            previous = bindings.get(template)
            if previous is not None and not types_compatible(actual, previous):
                raise TypeError(
                    f"generic type {type_name(template)} was bound to both "
                    f"{type_name(previous)} and {type_name(actual)}"
                )
            bindings[template] = actual
        return

    template_args = get_args(template)
    actual_args = get_args(actual)
    if (
        get_origin(template) == get_origin(actual)
        and len(template_args) == len(actual_args)
    ):
        for nested_template, nested_actual in zip(template_args, actual_args):
            bind_typevars(nested_template, nested_actual, bindings)


def substitute_typevars(value: Any, bindings: Mapping[Any, Any]) -> Any:
    if _is_typevar(value):
        return bindings.get(value, value)
    origin = get_origin(value)
    args = get_args(value)
    if origin is None or not args:
        return value
    resolved = tuple(substitute_typevars(arg, bindings) for arg in args)
    try:
        return origin[resolved[0] if len(resolved) == 1 else resolved]
    except TypeError:
        return value


def _is_typevar(value: Any) -> bool:
    return type(value).__name__ == "TypeVar"
