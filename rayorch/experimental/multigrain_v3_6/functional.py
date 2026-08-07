"""Multigrain v3.6 的纯符号 Port 关系操作。"""

from __future__ import annotations

from .api import OptionalPort, Port, active_builder
from .model import CompileError


def expand(port: Port) -> Port:
    """把一个 group Port 展开到新建的 child Domain。"""

    return active_builder().expand((port,))[0]


def expand_aligned(*ports: Port) -> tuple[Port, ...]:
    """按共同 cardinality 把多个 group Port 展开到同一 child Domain。"""

    return active_builder().expand(tuple(ports))


def reduce(port: Port, *, members: Port | None = None) -> Port:
    """沿 Domain parent 回收一级，并保留有序成员关系。"""

    return active_builder().reduce((port,), members)[0]


def reduce_aligned(
    *ports: Port,
    members: Port | None = None,
) -> tuple[Port, ...]:
    """按同一成员集合回收多个值 Port。"""

    return active_builder().reduce(ports, members)


def broadcast(port: Port, *, like: Port) -> Port:
    """把祖先 Domain 的 Port 投影到 ``like`` 所在的后代 Domain。"""

    return active_builder().broadcast(port, like)


def filter(port: Port, mask: Port) -> Port:
    """以布尔 mask 改变成员状态，但不改变 source 的 Domain。"""

    return active_builder().filter(port, mask)


def optional(port: Port) -> OptionalPort:
    """把 Port 标为 Call 的 optional 输入；不创建结构节点。"""

    if not isinstance(port, Port):
        raise CompileError("optional requires a Port")
    return OptionalPort(port)


__all__ = [
    "broadcast",
    "expand",
    "expand_aligned",
    "filter",
    "optional",
    "reduce",
    "reduce_aligned",
]
