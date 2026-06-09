"""User-facing DAG pipeline API."""
from __future__ import annotations

from functools import wraps
import inspect
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    get_args,
    get_origin,
    get_type_hints,
)

from ..ray_module import RayModule
from .compiler import GraphTracer, TraceProxy, forward_call_hints
from .graph import CompiledGraph, PipeRef


class Pipeline:
    """Declarative DAG pipeline base class.

    Subclass, attach ``RayModule`` attributes, and implement ``forward()`` with
    normal application-level type annotations. Calling the pipeline eagerly
    executes ``forward()``; executors compile and schedule the same definition.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        forward = cls.__dict__.get("forward")
        if forward is None:
            return

        if "run" not in cls.__dict__:
            @wraps(forward)
            def run(self, *args: Any, **kwargs: Any) -> Any:
                return self._run_eager(*args, **kwargs)

            setattr(cls, "run", run)

        if "__call__" not in cls.__dict__:
            @wraps(forward)
            def call(self, *args: Any, **kwargs: Any) -> Any:
                return self._run_eager(*args, **kwargs)

            setattr(cls, "__call__", call)

    def __init__(self) -> None:
        self._compiled: Optional[CompiledGraph] = None
        self._param_names: Tuple[str, ...] = ()
        self._input_keys: Tuple[str, ...] = ()

    def forward(self, x: Any) -> Any:
        raise NotImplementedError

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self._run_eager(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._run_eager(*args, **kwargs)

    def _run_eager(self, *args: Any, **kwargs: Any) -> Any:
        runtime_nodes = [
            name
            for name, value in self.__dict__.items()
            if getattr(value, "requires_compiled_executor", False)
        ]
        if runtime_nodes:
            raise RuntimeError(
                f"{type(self).__name__} contains RuntimeRayModule nodes "
                f"{runtime_nodes} and cannot run eagerly. Use "
                "RuntimeDagExecutor(pipeline, ...).run(...) instead."
            )
        return self.forward(*args, **kwargs)

    @staticmethod
    def _input_key(param: str) -> str:
        return f"__input__{param}"

    def _introspect_forward(
        self,
    ) -> Tuple[
        List[PipeRef],
        Dict[str, PipeRef],
        Tuple[str, ...],
        Tuple[str, ...],
        Dict[str, Any],
    ]:
        sig = inspect.signature(self.forward)
        try:
            hints = get_type_hints(self.forward)
        except Exception:
            hints = {}
        params = list(sig.parameters.values())
        if not params:
            raise ValueError("forward() must declare at least one parameter")

        pos_refs: List[PipeRef] = []
        kw_refs: Dict[str, PipeRef] = {}
        names: List[str] = []
        keys: List[str] = []
        input_types: Dict[str, Any] = {}

        for param in params:
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise TypeError(
                    "forward() cannot use *args/**kwargs; use explicit parameters"
                )
            key = self._input_key(param.name)
            value_type = hints.get(param.name, Any)
            if value_type is PipeRef:
                value_type = Any
            ref = PipeRef(key, value_type=value_type)
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                pos_refs.append(ref)
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                kw_refs[param.name] = ref
            names.append(param.name)
            keys.append(key)
            input_types[key] = value_type

        return pos_refs, kw_refs, tuple(names), tuple(keys), input_types

    def _forward_output_types(self, count: int) -> Tuple[Any, ...]:
        try:
            hint = get_type_hints(self.forward).get("return", Any)
        except Exception:
            hint = Any
        if hint is PipeRef:
            return ()
        origin = get_origin(hint)
        if origin in (tuple, Tuple):
            parts = get_args(hint)
            if len(parts) == 2 and parts[1] is Ellipsis:
                return (parts[0],) * count
            return tuple(parts)
        return (hint,) if count == 1 and hint is not Any else ()

    def compile(self) -> Pipeline:
        tracer = GraphTracer(forward_call_hints(self.forward))
        originals: Dict[str, RayModule] = {}
        for attr, value in list(self.__dict__.items()):
            if isinstance(value, RayModule):
                originals[attr] = value
                setattr(self, attr, TraceProxy(tracer, attr, value))
        if not originals:
            raise ValueError("no RayModule attributes found on pipeline")

        pos_refs, kw_refs, param_names, input_keys, input_types = (
            self._introspect_forward()
        )
        try:
            out = self.forward(*pos_refs, **kw_refs)
        finally:
            for attr, module in originals.items():
                setattr(self, attr, module)

        self._param_names = param_names
        self._input_keys = input_keys
        output_count = len(out) if isinstance(out, tuple) else 1
        self._compiled = tracer.build(
            out,
            input_keys,
            input_types=input_types,
            expected_output_types=self._forward_output_types(output_count),
        )
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

DagPipeline = Pipeline
