"""Public ArenaEngine surface.

Callers import only the engine, limits, and abort exception.  Internal state
records remain in :mod:`.state` for focused review and testing.
"""

from .engine import ArenaEngine
from .state import ArenaAbort, ArenaLimits

__all__ = ["ArenaAbort", "ArenaEngine", "ArenaLimits"]

