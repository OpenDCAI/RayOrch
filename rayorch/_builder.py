"""Symbolic trace builder behind the public authoring API."""

from __future__ import annotations

import contextvars
import inspect
import itertools
from typing import Any

from ._model import CallRef, DomainRef, InputMode, PortRef
from .api import OptionalInput, Pipeline, Port, RayModule
from .errors import CompileError
from ._program.compiler import compile_logical
from ._program.logical import (
    BroadcastOrigin,
    CallInputSpec,
    CallOutputOrigin,
    CallSpec,
    DomainSpec,
    ExpandOrigin,
    FilterOrigin,
    LogicalProgram,
    PortOrigin,
    PortSpec,
    ReduceOrigin,
    SourceOrigin,
    UdfSpec,
    freeze_mapping,
)
from ._program.plan import CompiledProgram


_TRACE_IDS = itertools.count()
_ACTIVE_TRACE: contextvars.ContextVar[ProgramBuilder | None] = contextvars.ContextVar(
    "rayorch_pipeline_trace",
    default=None,
)


class ProgramBuilder:
    """Capture authoring operations without deriving runtime facts or Effects."""

    def __init__(self, source_names: tuple[str, ...]) -> None:
        self.owner = next(_TRACE_IDS)
        self.next_port = 0
        self.next_call = 0
        self.next_domain = 1
        self.calls: dict[CallRef, CallSpec] = {}
        self.ports: dict[PortRef, PortSpec] = {}
        self.domains: dict[DomainRef, DomainSpec] = {
            DomainRef(0): DomainSpec(DomainRef(0), debug_name="root")
        }
        self.call_options: dict[CallRef, tuple[tuple[str, Any], ...]] = {}
        self._view_intern: dict[tuple[Any, ...], tuple[PortRef, ...]] = {}

        self.source_ports = tuple(
            self._new_port(DomainRef(0), SourceOrigin(index, name))
            for index, name in enumerate(source_names)
        )

    def public(self, ref: PortRef) -> Port:
        return Port(ref, self.owner)

    def spec(self, port: Port, label: str = "operation") -> PortSpec:
        if not isinstance(port, Port) or port._owner != self.owner:
            raise CompileError(f"{label} requires a Port from the active Pipeline")
        try:
            return self.ports[port.ref]
        except KeyError as error:
            raise CompileError(f"unknown Port: {port.ref}") from error

    def call(
        self,
        module: RayModule,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Port | tuple[Port, ...]:
        if not args and not kwargs:
            raise CompileError("RayModule requires at least one Port input")

        positional_inputs = [
            CallInputSpec(*self._input(value, f"arg_{index}"))
            for index, value in enumerate(args)
        ]
        keyword_inputs = [
            (name, CallInputSpec(*self._input(value, name)))
            for name, value in kwargs.items()
        ]
        inputs = (*positional_inputs, *(item for _, item in keyword_inputs))
        domains = {self.ports[item.port].domain for item in inputs}
        if len(domains) != 1:
            raise CompileError(
                "RayModule inputs belong to different Domains; use explicit "
                "broadcast/reduce or an aligned relation"
            )
        execution_domain = next(iter(domains))

        call = CallRef(self.next_call)
        self.next_call += 1
        self.calls[call] = CallSpec(
            call,
            UdfSpec(module.udf, module.init_args, tuple(module.init_kwargs.items())),
            execution_domain,
            tuple(positional_inputs),
            tuple(keyword_inputs),
        )
        self.call_options[call] = tuple(module.options.items())
        outputs = tuple(
            self._new_port(execution_domain, CallOutputOrigin(call, output_index))
            for output_index in range(module.num_outputs)
        )
        public = tuple(self.public(ref) for ref in outputs)
        return public[0] if len(public) == 1 else public

    def expand(self, ports: tuple[Port, ...]) -> tuple[Port, ...]:
        specs = self._ports(ports, "expand")
        if len({spec.domain for spec in specs}) != 1:
            raise CompileError("aligned expand inputs must share one parent Domain")
        if len({port.ref for port in ports}) != len(ports):
            raise CompileError("aligned expand inputs must be distinct Ports")

        signature = ("expand", tuple(port.ref for port in ports))
        existing = self._view_intern.get(signature)
        if existing is not None:
            return tuple(self.public(ref) for ref in existing)

        producers: list[CallOutputOrigin] = []
        for spec in specs:
            if not isinstance(spec.origin, CallOutputOrigin):
                raise CompileError("expand inputs must be outputs of one producer Call")
            producers.append(spec.origin)
        if len(ports) > 1 and len({origin.call for origin in producers}) != 1:
            raise CompileError(
                "aligned expand inputs must be outputs of one producer Call"
            )

        already_expanded = {
            spec.origin.group_port
            for spec in self.ports.values()
            if isinstance(spec.origin, ExpandOrigin)
        }
        if already_expanded.intersection(port.ref for port in ports):
            raise CompileError(
                "a group Port already belongs to another expand relation; "
                "reuse the existing expanded Port or declare alignment once"
            )

        parent = specs[0].domain
        child = DomainRef(self.next_domain)
        self.next_domain += 1
        self.domains[child] = DomainSpec(
            child,
            parent,
            debug_name=f"expand_{child.value}",
        )
        outputs = tuple(
            self._new_port(child, ExpandOrigin(port.ref)) for port in ports
        )
        self._view_intern[signature] = outputs
        return tuple(self.public(ref) for ref in outputs)

    def reduce(
        self,
        ports: tuple[Port, ...],
        members: Port | None,
    ) -> tuple[Port, ...]:
        specs = self._ports(ports, "reduce")
        if len({spec.domain for spec in specs}) != 1:
            raise CompileError("aligned reduce inputs must share one child Domain")
        child = specs[0].domain
        domain = self.domains[child]
        if domain.parent is None:
            raise CompileError("reduce input is already in a root Domain")

        member_port = ports[0] if members is None else members
        if self.spec(member_port, "reduce members").domain != child:
            raise CompileError("reduce members must share the values' child Domain")
        signature = ("reduce", tuple(port.ref for port in ports), member_port.ref)
        existing = self._view_intern.get(signature)
        if existing is not None:
            return tuple(self.public(ref) for ref in existing)

        outputs = tuple(
            self._new_port(domain.parent, ReduceOrigin(port.ref, member_port.ref))
            for port in ports
        )
        self._view_intern[signature] = outputs
        return tuple(self.public(ref) for ref in outputs)

    def broadcast(self, source: Port, like: Port) -> Port:
        source_spec = self.spec(source, "broadcast")
        target_spec = self.spec(like, "broadcast like")
        if source_spec.domain == target_spec.domain:
            return source
        if not self._is_ancestor(source_spec.domain, target_spec.domain):
            raise CompileError("broadcast source Domain must be a target ancestor")
        signature = ("broadcast", source.ref, target_spec.domain)
        existing = self._view_intern.get(signature)
        if existing is not None:
            return self.public(existing[0])
        output = self._new_port(target_spec.domain, BroadcastOrigin(source.ref))
        self._view_intern[signature] = (output,)
        return self.public(output)

    def filter(self, source: Port, mask: Port) -> Port:
        source_spec = self.spec(source, "filter")
        mask_spec = self.spec(mask, "filter mask")
        if source_spec.domain != mask_spec.domain:
            raise CompileError("filter source and mask must share one Domain")
        signature = ("filter", source.ref, mask.ref)
        existing = self._view_intern.get(signature)
        if existing is not None:
            return self.public(existing[0])
        output = self._new_port(source_spec.domain, FilterOrigin(source.ref, mask.ref))
        self._view_intern[signature] = (output,)
        return self.public(output)

    def normalize_outputs(self, value: Any) -> object:
        if isinstance(value, Port):
            self.spec(value, "Pipeline output")
            return value.ref
        if isinstance(value, tuple) and value:
            return tuple(self.normalize_outputs(item) for item in value)
        raise CompileError(
            "Pipeline.forward must return a Port or a non-empty nested tuple of Ports"
        )

    def build(self, output_tree: object, *, optimize: bool) -> CompiledProgram:
        logical = LogicalProgram(
            calls=freeze_mapping(self.calls),
            ports=freeze_mapping(self.ports),
            domains=freeze_mapping(self.domains),
            source_ports=self.source_ports,
            output_tree=output_tree,
        )
        return compile_logical(
            logical,
            freeze_mapping(self.call_options),
            optimize=optimize,
        )

    def _new_port(self, domain: DomainRef, origin: PortOrigin) -> PortRef:
        ref = PortRef(self.next_port)
        self.next_port += 1
        self.ports[ref] = PortSpec(ref, domain, origin)
        return ref

    def _ports(self, ports: tuple[Port, ...], label: str) -> tuple[PortSpec, ...]:
        if not ports:
            raise CompileError(f"{label} requires at least one Port")
        return tuple(self.spec(port, label) for port in ports)

    def _input(self, value: Any, name: str) -> tuple[PortRef, InputMode]:
        if isinstance(value, OptionalInput):
            self.spec(value.port, f"input {name}")
            return value.port.ref, InputMode.OPTIONAL
        if isinstance(value, Port):
            self.spec(value, f"input {name}")
            return value.ref, InputMode.REQUIRED
        raise CompileError(f"RayModule input {name!r} must be a Port")

    def _is_ancestor(self, ancestor: DomainRef, child: DomainRef) -> bool:
        cursor: DomainRef | None = child
        while cursor is not None:
            if cursor == ancestor:
                return True
            cursor = self.domains[cursor].parent
        return False


def active_builder(label: str = "Port operations") -> ProgramBuilder:
    """Return the active builder or reject authoring outside ``forward``."""

    builder = _ACTIVE_TRACE.get()
    if builder is None:
        raise CompileError(f"{label} are only valid in Pipeline.forward")
    return builder


def compile_pipeline(
    pipeline: Pipeline,
    *,
    optimize: bool,
) -> CompiledProgram:
    """Trace one Pipeline and delegate immutable graph compilation."""

    parameters = tuple(inspect.signature(pipeline.forward).parameters.values())
    allowed = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    if not parameters:
        raise CompileError("Pipeline.forward requires at least one source")
    if any(parameter.kind not in allowed for parameter in parameters):
        raise CompileError("Pipeline.forward supports positional sources only")

    builder = ProgramBuilder(tuple(parameter.name for parameter in parameters))
    token = _ACTIVE_TRACE.set(builder)
    try:
        sources = tuple(builder.public(port) for port in builder.source_ports)
        result = pipeline.forward(*sources)
    finally:
        _ACTIVE_TRACE.reset(token)
    return builder.build(builder.normalize_outputs(result), optimize=optimize)


__all__ = ["ProgramBuilder", "active_builder", "compile_pipeline"]
