"""Ray-free static Program pipeline: logical graph -> verified RuntimePlan.

This package must not import ``runtime`` or ``execution``. Its immutable output,
``RuntimePlan``, is the only static contract consumed by those outer layers.

The package initializer deliberately re-exports nothing: importing a lightweight
logical type must not eagerly load the full compiler pipeline.
"""
