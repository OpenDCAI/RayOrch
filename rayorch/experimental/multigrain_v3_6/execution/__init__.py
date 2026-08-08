"""Physical execution boundary for Ray actors, RPCs, and value-only Workers.

Execution may consume Program and runtime contracts; neither lower layer may
import this package. The initializer deliberately stays empty so actor-side
``execution.worker`` imports do not eagerly load the driver or compiler.
"""
