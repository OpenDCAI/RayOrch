"""User-facing DAG pipeline API."""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..ray_module import RayModule
from .compiler import GraphTracer, TraceProxy, forward_call_hints
from .executor import Executor, SequentialExecutor
from .graph import CompiledGraph, PipeRef


class Pipeline:
    """Declarative DAG pipeline base class.

    Subclass, attach ``RayModule`` attributes, implement ``forward()`` with
    ``PipeRef`` wiring, then call the pipeline on batched inputs.
    """

    def __init__(self) -> None:
        self._compiled: Optional[CompiledGraph] = None
        self._param_names: Tuple[str, ...] = ()
        self._input_keys: Tuple[str, ...] = ()

    def forward(self, x: PipeRef) -> PipeRef | Tuple[PipeRef, ...]:
        raise NotImplementedError

    @staticmethod
    def _input_key(param: str) -> str:
        return f"__input__{param}"

    def _introspect_forward(
        self,
    ) -> Tuple[List[PipeRef], Dict[str, PipeRef], Tuple[str, ...], Tuple[str, ...]]:
        sig = inspect.signature(self.forward)
        params = list(sig.parameters.values())
        if not params:
            raise ValueError("forward() must declare at least one parameter")

        pos_refs: List[PipeRef] = []
        kw_refs: Dict[str, PipeRef] = {}
        names: List[str] = []
        keys: List[str] = []

        for param in params:
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise TypeError(
                    "forward() cannot use *args/**kwargs; use explicit parameters"
                )
            key = self._input_key(param.name)
            ref = PipeRef(key)
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                pos_refs.append(ref)
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                kw_refs[param.name] = ref
            names.append(param.name)
            keys.append(key)

        return pos_refs, kw_refs, tuple(names), tuple(keys)

    def compile(self) -> Pipeline:
        tracer = GraphTracer(forward_call_hints(self.forward))
        originals: Dict[str, RayModule] = {}
        for attr, value in list(self.__dict__.items()):
            if isinstance(value, RayModule):
                originals[attr] = value
                setattr(self, attr, TraceProxy(tracer, attr, value))
        if not originals:
            raise ValueError("no RayModule attributes found on pipeline")

        pos_refs, kw_refs, param_names, input_keys = self._introspect_forward()
        try:
            out = self.forward(*pos_refs, **kw_refs)
        finally:
            for attr, module in originals.items():
                setattr(self, attr, module)

        self._param_names = param_names
        self._input_keys = input_keys
        self._compiled = tracer.build(out, input_keys)
        return self

    def _resolve_inputs(
        self,
        args: Tuple[Sequence[Any], ...],
        kwargs: Mapping[str, Sequence[Any]],
    ) -> Dict[str, Sequence[Any]]:
        if not self._param_names:
            raise RuntimeError("pipeline is not compiled")
        if args and kwargs:
            raise ValueError("pass inputs positionally or by keyword, not both")

        if kwargs:
            if set(kwargs.keys()) != set(self._param_names):
                raise ValueError(
                    f"keyword mismatch: expected {self._param_names}, "
                    f"got {tuple(kwargs.keys())}"
                )
            return {self._input_key(name): kwargs[name] for name in self._param_names}

        if len(args) != len(self._param_names):
            raise ValueError(
                f"expected {len(self._param_names)} positional inputs, "
                f"got {len(args)}"
            )
        return dict(zip(self._input_keys, args))

    def __call__(
        self,
        *inputs: Sequence[Any],
        **named_inputs: Sequence[Any],
    ) -> List[Any]:
        return self.run(*inputs, **named_inputs)

    def run(
        self,
        *inputs: Sequence[Any],
        executor: Executor | None = None,
        **named_inputs: Sequence[Any],
    ) -> List[Any]:
        """Run pipeline. Defaults to serial; pass ``DagExecutor()`` for overlap."""
        if self._compiled is None:
            self.compile()
        columns = self._resolve_inputs(inputs, named_inputs)
        if executor is None:
            executor = SequentialExecutor()
        return executor.execute(self._compiled, columns)


DagPipeline = Pipeline
