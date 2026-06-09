# Runtime Test Layout

Runtime tests are separated by execution cost and responsibility.

## `unit/`

Fast tests with no Ray cluster or actor processes.

- `test_compile.py`: DAG compile, port naming, arity, and eager-call guards.
- `test_keyword_compile.py`: positional/keyword ref binding and operator
  parameter ordering.
- `test_dispatch.py`: deterministic MicroBatch sharding and broadcast rules.
- `test_rowwise.py`: bad-row isolation, retry, quarantine, and lineage semantics.

Run continuously during development:

```bash
pytest test/runtime/unit
```

## `integration/`

Tests that require real Ray actors. The suite shares one local Ray cluster to
avoid paying startup cost for every test.

- `test_lifecycle.py`: deferred configuration, start, close, and ownership.
- `test_executor.py`: replicas, column input, inflight error localization, and
  Flash-MinerU-style execution.
- `test_dag_semantics.py`: core empty-batch, reused-module, fanout, and fanin
  boundaries.
- `test_dag_topologies.py`: representative diamonds, multi-parent joins,
  branch filtering, graph outputs, and failures at different DAG depths.
- `test_keyword_arguments.py`: all-keyword, mixed positional/keyword, keyword
  error isolation, and executor source-column keyword calls.
- `test_overlap.py`: replica concurrency and pipeline overlap.

The topology expectations are documented in
[`../../docs/runtime_dag_topology_matrix.md`](../../docs/runtime_dag_topology_matrix.md).

Each test should cover one observable boundary and explicitly close every
executor or standalone RuntimeRayModule it starts.

```bash
pytest test/runtime/integration
```

## `performance/`

Long-running comparisons and stress workloads. These are skipped by default.

```bash
pytest --runslow -s test/runtime/performance
```

Correctness assertions belong in `unit/` or `integration/`; performance tests
should not become the only coverage for a behavior.
