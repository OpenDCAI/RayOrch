"""Opt-in Ray execution backend for multigrain IR."""

from .executor import FaultSpec, MultigrainRayExecutor, lpt_shard_planner

__all__ = ["FaultSpec", "MultigrainRayExecutor", "lpt_shard_planner"]
