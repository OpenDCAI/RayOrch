"""Public exceptions raised while compiling and executing RayOrch pipelines."""


class CompileError(ValueError):
    """The symbolic pipeline violates a compile-time invariant."""


class ExecutionError(RuntimeError):
    """A pipeline execution cannot continue under its compiled contract."""


__all__ = ["CompileError", "ExecutionError"]
