from functools import wraps
from contextlib import contextmanager

try:
    import torch
except ImportError:  # pragma: no cover - depends on optional runtime deps
    torch = None

g_trace_enabled = True

def is_trace_enabled():
    return g_trace_enabled and torch is not None and hasattr(torch, "cuda")

@contextmanager
def nvtx_range(msg):
    if not is_trace_enabled():
        yield
        return
    try:
        torch.cuda.nvtx.range_push(msg)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

def nvtx(msg: str):
    """通用的 NVTX 装饰器，可用于 Ray 任务 / Actor 方法"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with nvtx_range(msg):
                return func(*args, **kwargs)
        return wrapper
    return decorator
