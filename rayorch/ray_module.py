from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Tuple, Type, TypeVar, cast
from typing_extensions import ParamSpec
from .nvtx_profiler import nvtx_range
import contextlib

INITP = ParamSpec("InitP")   # op_cls.__init__ 的参数
RUNP = ParamSpec("RunP")     # op_cls.run 的参数
R = TypeVar("R")
class RunOp(Protocol[INITP, RUNP, R]):
    def __init__(self, *args: INITP.args, **kwargs: INITP.kwargs) -> None: ...
    def run(self, *args: RUNP.args, **kwargs: RUNP.kwargs) -> R: ...

import ray

from .dispatch_mode import DispatchMode, ShardedObjectRef, get_predefined_dispatch_fn
from .env_registry import EnvRegistry

# from image_class import ImageLoadOp, ImageSaveOP
# from yolo_class import YOLODrawOp
# from sam_class import SAMOp, OutputSAMMasksOp


@ray.remote
def _join_and_collect_refs(collect_fn, module, refs):
    """Join a list of refs and apply module-level collect in a task."""
    outputs = ray.get(refs)
    return collect_fn(module, outputs)


_JOIN_COLLECT_SUBMIT_COUNT = 0


def _reset_join_collect_submit_count() -> None:
    global _JOIN_COLLECT_SUBMIT_COUNT
    _JOIN_COLLECT_SUBMIT_COUNT = 0


def _get_join_collect_submit_count() -> int:
    return _JOIN_COLLECT_SUBMIT_COUNT


# --------------------------
# Actor：运行时不要 Generic/ParamSpec, 否则会报serialize的错误
# --------------------------
@ray.remote
class RunnerActor:
    def __init__(
        self,
        op_cls: Type[Any],                 # 运行时用 Any
        init_args: Tuple[Any, ...],
        init_kwargs: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ):
        dev_nvtx_range = nvtx_range(f"op[{op_cls.__name__}].init | replica={meta.get('replica', 0)} | tag={meta.get('tag', '')}") if meta.get('dev', False) else contextlib.nullcontext()
        with dev_nvtx_range:
            self.op = op_cls(*init_args, **init_kwargs)

    def run(self, args, kwargs, meta=None):
        def _slice_for_replica(seq, shard_idx: int, num_shards: int):
            n = len(seq)
            base = n // num_shards
            rem = n % num_shards
            start = shard_idx * base + min(shard_idx, rem)
            end = start + base + (1 if shard_idx < rem else 0)
            return seq[start:end]

        def _resolve_refs(x):
            if isinstance(x, ShardedObjectRef):
                resolved = ray.get(x.ref)
                if isinstance(resolved, (list, tuple)):
                    return _slice_for_replica(resolved, x.shard_idx, x.num_shards)
                return resolved
            if isinstance(x, ray.ObjectRef):
                return ray.get(x)
            # Only recurse into built-in containers. Some third-party payload objects
            # (e.g. MinerU ContentBlock) behave like dict subclasses while exposing
            # attribute access (`block.type`). Rebuilding them as plain dict would
            # destroy their runtime protocol.
            if type(x) is tuple:
                return tuple(_resolve_refs(v) for v in x)
            if type(x) is list:
                return [_resolve_refs(v) for v in x]
            if type(x) is dict:
                return {k: _resolve_refs(v) for k, v in x.items()}
            return x

        dev_nvtx_range = nvtx_range(f"op[{self.op.__class__.__name__}].run | rreplica={meta.get('replica', 0)} | tag={meta.get('tag', '')}") if meta.get('dev', False) else contextlib.nullcontext()
        with dev_nvtx_range:
            resolved_args = _resolve_refs(args)
            resolved_kwargs = _resolve_refs(kwargs)
            return self.op.run(*resolved_args, **resolved_kwargs)

class RayModule(Generic[INITP, RUNP, R]):
    @dataclass
    class RayModuleFuture:
        module: "RayModule"
        refs: Any
        collect_fn: Optional[Callable[..., Any]]
        _dependency_ref_cache: Optional[ray.ObjectRef] = field(default=None, init=False, repr=False)

        def get(self) -> Any:
            """
            Block until all the refs are resolved and return the collected output.
            """
            if isinstance(self.refs, list):
                output = ray.get(self.refs)
                if self.collect_fn is None:
                    return output
                return self.collect_fn(self.module, output)
            return ray.get(self.refs)

        def completion_refs(self) -> List[ray.ObjectRef]:
            """``ObjectRef`` 列表，供 ``ray.wait`` 等按 ref 驱动调度（单 ref 时包成单元素列表）。"""
            r = self.refs
            return [r] if not isinstance(r, list) else list(r)

        def dependency_ref(self) -> ray.ObjectRef:
            """
            Return a single dependency ref for downstream stages.

            - single ref: pass-through
            - multi refs: asynchronously join+collect into one ref
            """
            if self._dependency_ref_cache is not None:
                return self._dependency_ref_cache

            refs = self.completion_refs()
            if len(refs) == 1:
                self._dependency_ref_cache = refs[0]
                return refs[0]
            if self.collect_fn is None:
                raise ValueError(
                    "Cannot build single dependency ref from multiple refs without collect_fn. "
                    "Please gather upstream explicitly before wiring downstream."
                )
            global _JOIN_COLLECT_SUBMIT_COUNT
            _JOIN_COLLECT_SUBMIT_COUNT += 1
            self._dependency_ref_cache = _join_and_collect_refs.remote(self.collect_fn, self.module, refs)
            return self._dependency_ref_cache

    def __init__(
        self,
        op_cls: Type[RunOp[INITP, RUNP, R]],
        *,
        env: Optional[str] = None,   # ✅ 允许 None
        # Map and Reduce
        replicas: int = 1,
        num_gpus_per_replica: float = 0.0,
        dispatch_mode: DispatchMode | None = None,
        dispatch_fn : Optional[Callable[..., Any]] = None,
        collect_fn : Optional[Callable[..., Any]] = None,
        # Tag and Log
        dev_mode: bool = False,
        init_tag_fn: Optional[Callable[[Tuple[Any, ...], Dict[str, Any], int, Dict[str, Any] | None], str]] = None,
        run_tag_fn: Optional[Callable[[Tuple[Any, ...], Dict[str, Any], int, Dict[str, Any] | None], str]] = None,
    ) -> None:
        self._op_cls = op_cls
        self._env = env
        self._replicas = replicas
        self._num_gpus_per_replica = num_gpus_per_replica
        self.actors = []
        self._itag_fn = init_tag_fn
        self._tag_fn = run_tag_fn
        assert not ((not dev_mode) and (self._itag_fn is not None or self._tag_fn is not None)), "dev_mode must be True if init_tag_fn or run_tag_fn is provided"
        self._is_dev_mode = dev_mode
        
        # 让 dispatch_fn/collect_fn 也能直接传入覆盖 registry
        if dispatch_fn is not None and collect_fn is not None:
            self._dispatch_fn = dispatch_fn
            self._collect_fn = collect_fn
        elif dispatch_mode is not None:
            spec = get_predefined_dispatch_fn(dispatch_mode)
            self._dispatch_fn = spec.dispatch_fn
            self._collect_fn = spec.collect_fn
        else:
            self._dispatch_fn = None
            self._collect_fn = None

    def pre_init(self, *args: INITP.args, **kwargs: INITP.kwargs) -> "RayModule[INITP, RUNP, R]":
        tags = [""] * self._replicas
        # tag_fn
        if self._itag_fn:
            tags = [self._itag_fn(args, kwargs, i) for i in range(self._replicas)]
        # metas only for for logging/profiling, passing info between actors
        metas = [{"replica": i, "tag": tags[i], "dev": self._is_dev_mode} for i in range(self._replicas)]
        # create actors, with replica
        self.actors = [
            RunnerActor.options(
                runtime_env=EnvRegistry.get_ray_style_env(self._env) if self._env is not None else None,
                num_gpus=self._num_gpus_per_replica,
            ).remote(self._op_cls, args, kwargs, meta=metas[i])
            for i in range(self._replicas)
        ]
        return self

    def _fanout_refs(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> List[ray.ObjectRef]:
        """
        Submit one run per replica and return the list of refs.

        Dispatch protocol (aligned with `rayorch.dispatch_mode`):
        - per_args: tuple, each slot is a sequence of length `replicas` (column-wise)
        - per_kwargs: dict, each value is a sequence of length `replicas` (column-wise)
        """
        per_args, per_kwargs = self._dispatch_fn(self, *args, **kwargs)
        if not isinstance(per_args, tuple):
            raise TypeError(f"dispatch_fn must return tuple for per_args. Got {type(per_args)}")

        refs: List[ray.ObjectRef] = []
        for i in range(self._replicas):
            try:
                args_i = tuple(per_args[j][i] for j in range(len(per_args)))
            except Exception as e:
                raise ValueError(
                    f"dispatch_fn returned invalid per_args structure for replicas={self._replicas}. "
                    f"Expected each per_args[j] to be indexable by replica i."
                ) from e

            try:
                kwargs_i = {k: v[i] for k, v in per_kwargs.items()}
            except Exception as e:
                raise ValueError(
                    f"dispatch_fn returned invalid per_kwargs structure for replicas={self._replicas}. "
                    f"Expected each per_kwargs[k] to be indexable by replica i."
                ) from e

            tag_i = self._tag_fn(args_i, kwargs_i, i) if self._tag_fn else ""
            meta_i = {"tag": tag_i, "replica": i, "dev": self._is_dev_mode}
            refs.append(self.actors[i].run.remote(args_i, kwargs_i, meta_i))
        return refs

    def __call__(self, *args: RUNP.args, **kwargs: RUNP.kwargs) -> R:
        # ---- single replica ----
        if self._dispatch_fn is None or self._collect_fn is None or self._replicas == 1:
            args_i = args
            kwargs_i = kwargs
            tag = self._tag_fn(args_i, kwargs_i, 0) if self._tag_fn else ""
            meta = {"tag": tag, "replica": 0, "dev": self._is_dev_mode}

            ref = self.actors[0].run.remote(args_i, kwargs_i, meta)
            return cast(R, ray.get(ref))

        # ---- multi replica ----
        refs = self._fanout_refs(args, kwargs)
        output = ray.get(refs)  # block until all refs are resolved
        return cast(R, self._collect_fn(self, output))

    def remote(self, *args: RUNP.args, **kwargs: RUNP.kwargs) -> "RayModule.RayModuleFuture":
        """
        Non-blocking submit, aligned with Ray's ``actor.method.remote()`` style: returns
        :class:`RayModuleFuture`. Use :meth:`gather` to block and apply ``collect_fn``, or
        read ``.refs`` for raw ``ObjectRef`` / list of refs.

        If a positional or keyword argument is another module's :class:`RayModuleFuture`
        (e.g. eager pipeline wiring), it is replaced by a single dependency ref:
        - single-ref upstream: pass-through
        - multi-ref upstream: join+collect in a lightweight Ray task.
        """
        Fut = RayModule.RayModuleFuture

        args = tuple(a.dependency_ref() if isinstance(a, Fut) else a for a in args)
        kwargs = {
            k: (v.dependency_ref() if isinstance(v, Fut) else v)
            for k, v in kwargs.items()
        }

        if self._dispatch_fn is None or self._collect_fn is None or self._replicas == 1:
            meta = {"tag": "", "replica": 0, "dev": self._is_dev_mode}
            ref = self.actors[0].run.remote(args, kwargs, meta)
            return RayModule.RayModuleFuture(module=self, refs=ref, collect_fn=None)

        refs = self._fanout_refs(args, kwargs)
        return RayModule.RayModuleFuture(module=self, refs=refs, collect_fn=self._collect_fn)

    submit = remote

    def gather(self, pending: "RayModule.RayModuleFuture") -> R:
        """
        Block for a task previously returned by :meth:`remote` and return the final output.
        """
        if pending.module is not self:
            raise ValueError("RayModuleFuture was not created by this RayModule instance")
        return cast(R, pending.get())

