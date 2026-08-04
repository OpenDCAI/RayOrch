"""Multigrain V3.2 的 Port 级 grain transform。"""

from __future__ import annotations

from .api import Port, _expand_ports, _reduce_ports, optional


def expand(port: Port) -> Port:
    """把一个 Port 展开为新的 child-domain Port。"""

    return _expand_ports((port,))[0]


def expand_aligned(*ports: Port) -> tuple[Port, ...]:
    """用共享 child Entity domain 展开多个 group Port。"""

    return _expand_ports(tuple(ports))


def reduce(port: Port) -> Port:
    """把一个 child-domain Port 聚合回最近的 parent。"""

    return _reduce_ports((port,))[0]


def reduce_aligned(*ports: Port) -> tuple[Port, ...]:
    """用共享 canonical shape 聚合多个 aligned Port。"""

    return _reduce_ports(tuple(ports))


__all__ = [
    "expand",
    "expand_aligned",
    "optional",
    "reduce",
    "reduce_aligned",
]
