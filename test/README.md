# Test layout

- `dag/unit/`: compiler and schema checks that do not start Ray.
- `dag/integration/`: compiled DAG execution and representative pipeline shapes.
- `ray_module/`: dispatch and collect behavior for the lightweight `RayModule`.
- `runtime/`: industrial runtime unit, integration, and opt-in performance tests.
- `legacy/`: archived tests for deprecated implementations; excluded from CI.
- `manual/`: hardware-dependent demos, profiling scripts, and intentional failures.

Run the regular suite with:

```bash
pytest test
```

Run long performance tests explicitly with:

```bash
pytest test --runslow
```

Files under `manual/` are excluded from pytest collection and should be run directly.
