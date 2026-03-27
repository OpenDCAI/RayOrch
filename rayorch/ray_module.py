from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Optional, Protocol, Tuple, Type, TypeVar, cast
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

from .dispatch_mode import DispatchMode, get_predefined_dispatch_fn
from .env_registry import EnvRegistry

# from image_class import ImageLoadOp, ImageSaveOP
# from yolo_class import YOLODrawOp
# from sam_class import SAMOp, OutputSAMMasksOp


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
        dev_nvtx_range = nvtx_range(f"op[{self.op.__class__.__name__}].run | rreplica={meta.get('replica', 0)} | tag={meta.get('tag', '')}") if meta.get('dev', False) else contextlib.nullcontext()
        with dev_nvtx_range:
            return self.op.run(*args, **kwargs)

class RayModule(Generic[INITP, RUNP, R]):
    @dataclass(frozen=True)
    class PendingResult:
        refs: Any
        collect_fn: Optional[Callable[..., Any]]

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
        if self._itag_fn:
            tags = [self._itag_fn(args, kwargs, i) for i in range(self._replicas)]
        metas = [{"replica": i, "tag": tags[i], "dev": self._is_dev_mode} for i in range(self._replicas)]
        self.actors = [
            RunnerActor.options(
                runtime_env=EnvRegistry.get_ray_style_env(self._env) if self._env is not None else None,
                num_gpus=self._num_gpus_per_replica,
            ).remote(self._op_cls, args, kwargs, meta=metas[i])
            for i in range(self._replicas)
        ]
        return self

    def _fanout(self, *args: RUNP.args, **kwargs: RUNP.kwargs):
        # ---- single replica ----
        if self._dispatch_fn is None or self._collect_fn is None or self._replicas == 1:
            args_i = args
            kwargs_i = kwargs
            tag = self._tag_fn(args_i, kwargs_i, 0) if self._tag_fn else ""
            meta = {"tag": tag, "replica": 0, "dev": self._is_dev_mode}

            ref = self.actors[0].run.remote(args_i, kwargs_i, meta)
            return ray.get(ref)

        # ---- multi replica ----
        per_args, per_kwargs = self._dispatch_fn(self, *args, **kwargs)

        # Check for correctness
        for j in range(len(per_args)):
            if len(per_args[j]) != self._replicas:
                raise ValueError(f"Dispatched args[{j}] len ({len(per_args[j])}) != replicas ({self._replicas})")
        for k, v in per_kwargs.items():
            if len(v) != self._replicas:
                raise ValueError(f"Dispatched kwargs['{k}'] len ({len(v)}) != replicas ({self._replicas})")

        refs = []
        tags = [] 

        for i in range(self._replicas):
            args_i = tuple(per_args[j][i] for j in range(len(per_args)))
            kwargs_i = {k: v[i] for k, v in per_kwargs.items()}

            tag_i = self._tag_fn(args_i, kwargs_i, i) if self._tag_fn else ""
            tags.append(tag_i)

            meta_i = {"tag": tag_i, "replica": i, "dev": self._is_dev_mode}
            refs.append(self.actors[i].run.remote(args_i, kwargs_i, meta_i))

        output = ray.get(refs)
        return cast(R, self._collect_fn(self, output))
    
    def __call__(self, *args: RUNP.args, **kwargs: RUNP.kwargs) -> R:
        return cast(R, self._fanout(*args, **kwargs))

    def submit(self, *args: RUNP.args, **kwargs: RUNP.kwargs) -> "RayModule.PendingResult":
        """
        Submit task asynchronously and return a lightweight handle.
        """
        if self._dispatch_fn is None or self._collect_fn is None or self._replicas == 1:
            meta = {"tag": "", "replica": 0, "dev": self._is_dev_mode}
            ref = self.actors[0].run.remote(args, kwargs, meta)
            return RayModule.PendingResult(refs=ref, collect_fn=None)

        per_args, per_kwargs = self._dispatch_fn(self, *args, **kwargs)
        refs = []
        for i in range(self._replicas):
            args_i = tuple(per_args[j][i] for j in range(len(per_args)))
            kwargs_i = {k: v[i] for k, v in per_kwargs.items()}
            tag_i = self._tag_fn(args_i, kwargs_i, i) if self._tag_fn else ""
            meta_i = {"tag": tag_i, "replica": i, "dev": self._is_dev_mode}
            refs.append(self.actors[i].run.remote(args_i, kwargs_i, meta_i))
        return RayModule.PendingResult(refs=refs, collect_fn=self._collect_fn)

    def gather(self, pending: "RayModule.PendingResult") -> R:
        """
        Block for a previously submitted task and return the final output.
        """
        if isinstance(pending.refs, list):
            output = ray.get(pending.refs)
            if pending.collect_fn is None:
                return cast(R, output)
            return cast(R, pending.collect_fn(self, output))
        return cast(R, ray.get(pending.refs))

    def remote(self, *args: RUNP.args, **kwargs: RUNP.kwargs):
        # 简单版本：返回每个 replica 的 refs（或单个 ref）
        if self._dispatch_fn is None or self._collect_fn is None or self._replicas == 1:
            return self.actors[0].run.remote(args, kwargs)

        per_args, per_kwargs = self._dispatch_fn(self, *args, **kwargs)
        refs = []
        for i in range(self._replicas):
            args_i = tuple(per_args[j][i] for j in range(len(per_args)))
            kwargs_i = {k: v[i] for k, v in per_kwargs.items()}
            refs.append(self.actors[i].run.remote(args_i, kwargs_i))
        return refs

