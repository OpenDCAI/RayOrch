"""Pipeline forward() tracing and DAG compilation helpers."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
import textwrap
from typing import Any, Dict, List, Mapping, Tuple

from ..ray_module import RayModule
from .graph import CompiledGraph, NodeSpec, PipeRef
from .schema import (
    bind_typevars,
    input_types,
    return_types,
    substitute_typevars,
    type_name,
    types_compatible,
)


@dataclass(frozen=True)
class _CallHint:
    """AST-derived output names for one ``self.<module>(...)`` call."""

    output_names: Tuple[str, ...] = ()


def _expect_ref(value: Any) -> PipeRef:
    if isinstance(value, PipeRef):
        return value
    raise TypeError(
        "pipeline operators can only consume symbolic values produced by "
        "forward() inputs or upstream operators"
    )


def _validate_ref(
    ref: PipeRef,
    context: str,
    input_keys: set[str],
    nodes: Dict[str, NodeSpec],
) -> None:
    if ref.index < 0:
        raise ValueError(f"'{context}': negative slot index in {ref}")
    if ref.node in input_keys:
        return
    if ref.node not in nodes:
        raise ValueError(f"'{context}': references unknown node '{ref.node}'")
    if ref.index >= nodes[ref.node].num_outputs:
        raise ValueError(
            f"'{context}': reads slot {ref.index} from '{ref.node}', "
            f"which only has {nodes[ref.node].num_outputs} outputs"
        )


def _simple_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _target_names(target: ast.AST) -> Tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        names = tuple(_simple_name(elt) for elt in target.elts)
        if all(names):
            return tuple(str(name) for name in names)
    return ()


def _call_attr_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
    ):
        return func.attr
    return None


class _ForwardHintVisitor(ast.NodeVisitor):
    """Collect ``self.op(...)`` calls in the same order Python executes them."""

    def __init__(self) -> None:
        self.hints: Dict[tuple[str, int], _CallHint] = {}
        self._seen: Dict[str, int] = {}
        self._assign_outputs: Dict[ast.Call, Tuple[str, ...]] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.value, ast.Call):
            outputs = _target_names(node.targets[0])
            if outputs:
                self._assign_outputs[node.value] = outputs
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

        attr = _call_attr_name(node)
        if attr is not None:
            ordinal = self._seen.get(attr, 0)
            self._seen[attr] = ordinal + 1
            self.hints[(attr, ordinal)] = _CallHint(
                output_names=self._assign_outputs.get(node, ()),
            )


def forward_call_hints(forward: Any) -> Dict[tuple[str, int], _CallHint]:
    """Parse simple naming hints from ``forward()`` source.

    Dynamic tracing remains the source of truth for topology.  AST is used only
    to attach stable, node-local names to ports when source is available.
    """
    try:
        source = inspect.getsource(forward)
    except (OSError, TypeError):
        return {}

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return {}

    visitor = _ForwardHintVisitor()
    visitor.visit(tree)
    return visitor.hints


def _input_names(
    module: RayModule,
    args: Tuple[PipeRef, ...],
    kw_args: Dict[str, PipeRef],
) -> Tuple[str, ...]:
    op_cls = getattr(module, "_user_op_cls", getattr(module, "_op_cls", None))
    run_fn = getattr(op_cls, "run", None)
    if run_fn is not None:
        try:
            params = list(inspect.signature(run_fn).parameters.values())
            if params and params[0].name == "self":
                params = params[1:]
            bound = inspect.Signature(params).bind_partial(*args, **kw_args)
            return tuple(bound.arguments.keys())
        except (TypeError, ValueError):
            pass
    return tuple([f"arg{i}" for i in range(len(args))] + list(kw_args.keys()))


def _notify_compile_hook(spec: NodeSpec) -> None:
    hook = getattr(spec.module, "on_compile_node", None)
    if hook is not None:
        hook(spec)


class GraphTracer:
    """Records node declarations during ``forward()``."""

    def __init__(
        self,
        call_hints: Mapping[tuple[str, int], _CallHint] | None = None,
    ) -> None:
        self._tape: List[NodeSpec] = []
        self._name_count: Dict[str, int] = {}
        self._call_hints = dict(call_hints or {})
        self._call_count: Dict[str, int] = {}
        self._inferred_input_types: Dict[str, Any] = {}

    def _unique_name(self, base: str) -> str:
        n = self._name_count.get(base, 0)
        self._name_count[base] = n + 1
        return base if n == 0 else f"{base}_{n}"

    def add_node(
        self,
        base_name: str,
        module: RayModule,
        args: Tuple[PipeRef, ...],
        kw_args: Dict[str, PipeRef],
    ) -> tuple[str, int]:
        name = self._unique_name(base_name)
        ordinal = self._call_count.get(base_name, 0)
        self._call_count[base_name] = ordinal + 1

        hint = self._call_hints.get((base_name, ordinal), _CallHint())
        input_names = _input_names(module, args, kw_args)
        output_names = hint.output_names or tuple(
            f"{name}.out{i}" for i in range(module.num_outputs)
        )
        num_outputs = len(output_names) if hint.output_names else module.num_outputs
        if hint.output_names and module.num_outputs != 1 and module.num_outputs != num_outputs:
            raise ValueError(
                f"node '{name}' assigns {num_outputs} outputs in forward(), "
                f"but module declares num_outputs={module.num_outputs}"
            )
        node_input_types = input_types(module, args, kw_args)
        all_refs = args + tuple(kw_args.values())
        type_bindings: Dict[Any, Any] = {}
        for input_name, ref, expected_type in zip(
            input_names, all_refs, node_input_types
        ):
            actual_type = ref.value_type
            if ref.node.startswith("__input__") and actual_type is Any:
                inferred = self._inferred_input_types.get(ref.node, Any)
                if inferred is Any:
                    self._inferred_input_types[ref.node] = expected_type
                    actual_type = expected_type
                else:
                    actual_type = inferred
            bind_typevars(expected_type, actual_type, type_bindings)
            if not types_compatible(actual_type, expected_type):
                raise TypeError(
                    f"type mismatch at node '{name}', input '{input_name}': "
                    f"expected {type_name(expected_type)}, got "
                    f"{type_name(actual_type)} from '{ref.node}'"
                )

        op_cls = getattr(module, "_user_op_cls", getattr(module, "_op_cls", None))
        run_fn = getattr(op_cls, "run", None)
        output_types = (
            return_types(run_fn, num_outputs)
            if run_fn is not None
            else (Any,) * num_outputs
        )
        output_types = tuple(
            substitute_typevars(value, type_bindings)
            for value in output_types
        )

        self._tape.append(NodeSpec(
            name=name,
            module=module,
            args=args,
            kw_args=dict(kw_args),
            max_inflight=module.max_inflight,
            num_outputs=num_outputs,
            input_names=input_names,
            output_names=output_names,
            input_types=node_input_types,
            output_types=output_types,
        ))
        return name, num_outputs

    def make_refs(
        self,
        node_name: str,
        num_outputs: int,
    ) -> PipeRef | Tuple[PipeRef, ...]:
        output_types = self._tape[-1].output_types
        if num_outputs <= 1:
            return PipeRef(node_name, value_type=output_types[0])
        return tuple(
            PipeRef(node_name, i, output_types[i])
            for i in range(num_outputs)
        )

    def build(
        self,
        outputs: PipeRef | Tuple[PipeRef, ...],
        input_keys: Tuple[str, ...],
        input_types: Mapping[str, Any] | None = None,
        expected_output_types: Tuple[Any, ...] = (),
    ) -> CompiledGraph:
        graph_outputs = (outputs,) if isinstance(outputs, PipeRef) else tuple(outputs)
        if expected_output_types:
            if len(expected_output_types) != len(graph_outputs):
                raise TypeError(
                    "forward() return annotation does not match the number "
                    "of pipeline outputs"
                )
            for ref, expected_type in zip(graph_outputs, expected_output_types):
                if not types_compatible(ref.value_type, expected_type):
                    raise TypeError(
                        "pipeline output type mismatch: expected "
                        f"{type_name(expected_type)}, got "
                        f"{type_name(ref.value_type)} from '{ref.node}'"
                    )
        input_key_set = set(input_keys)
        by_name = {n.name: n for n in self._tape}
        topo_order = tuple(n.name for n in self._tape)

        deps: Dict[str, Tuple[str, ...]] = {}
        consumers: Dict[str, List[str]] = {n: [] for n in topo_order}

        for node in self._tape:
            all_refs = list(node.args) + list(node.kw_args.values())
            seen: set[str] = set()
            dep_list: List[str] = []
            for ref in all_refs:
                _validate_ref(ref, node.name, input_key_set, by_name)
                if ref.node in input_key_set or ref.node in seen:
                    continue
                seen.add(ref.node)
                dep_list.append(ref.node)
                consumers[ref.node].append(node.name)
            deps[node.name] = tuple(dep_list)

        for ref in graph_outputs:
            _validate_ref(ref, "<graph_output>", input_key_set, by_name)

        resolved_input_types = dict(input_types or {})
        for key, inferred_type in self._inferred_input_types.items():
            if resolved_input_types.get(key, Any) is Any:
                resolved_input_types[key] = inferred_type

        graph = CompiledGraph(
            nodes=by_name,
            topo_order=topo_order,
            deps=deps,
            consumers={k: tuple(v) for k, v in consumers.items()},
            graph_outputs=graph_outputs,
            input_keys=input_keys,
            input_types=resolved_input_types,
            output_types=tuple(ref.value_type for ref in graph_outputs),
        )
        for name in graph.topo_order:
            _notify_compile_hook(graph.nodes[name])
        return graph


class TraceProxy:
    """Stands in for a ``RayModule`` during ``forward()`` tracing."""

    def __init__(self, tracer: GraphTracer, attr_name: str, module: RayModule):
        self._tracer = tracer
        self._attr_name = attr_name
        self._module = module

    def __call__(self, *args: Any, **kwargs: Any) -> PipeRef | Tuple[PipeRef, ...]:
        pipe_args = tuple(_expect_ref(v) for v in args)
        pipe_kw = {k: _expect_ref(v) for k, v in kwargs.items()}
        name, num_outputs = self._tracer.add_node(
            self._attr_name, self._module, pipe_args, pipe_kw,
        )
        return self._tracer.make_refs(name, num_outputs)
