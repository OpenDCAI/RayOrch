# Roadmap

Date: 2026-06-12

This document describes the four delivery phases that bring the current Runtime
MVP to a production-ready, multi-backend distributed pipeline system.

## Phase 1 — Architecture and Provider Abstractions

**Goal**: establish the infrastructure layering, provider interfaces, and state
model without requiring external services.

Deliverables:

- Document the control / execution / storage plane split
  ([`dag_new_pipeline_architecture.md`](dag_new_pipeline_architecture.md)).
- Define all five provider interfaces with no-op / in-memory defaults
  ([`provider_abstractions.md`](provider_abstractions.md)):
  - `ArtifactStoreProvider` → `InMemoryArtifactStore`
  - `MetadataStoreProvider` → `InMemoryMetadataStore`
  - `LineageSinkProvider` → `InMemoryLineageSink`
  - `MetricsProvider` → `NoopMetricsProvider`
  - `EventBusProvider` → `NoopEventBusProvider`
- Define the job / stage / block state machine
  ([`state_model.md`](state_model.md)).
- Wire provider injection into `RuntimeDagExecutor`.
- All existing tests must continue to pass with the in-memory providers as
  defaults.

Exit criteria: `pytest test/` passes; `RuntimeDagExecutor` accepts the five
provider kwargs.

## Phase 2 — IR Schema and RayBackend

**Goal**: expose an explicit JSON IR and the `RayBackend` code generator.

Deliverables:

- Implement `rayorch.compiler.emit_ir(pipeline)` using the existing AST +
  tracing path.
- Define the IR schema v1 ([`dsl_compiler.md`](dsl_compiler.md)).
- Implement `RayBackend.build(ir)` that reconstructs a `RuntimeDagExecutor`
  from a serialized IR without requiring the original Python source.
- Implement `rayorch.compiler.backends.registry` for named backend lookup.
- Add round-trip tests: compile pipeline → emit IR → build via `RayBackend` →
  run → compare results to direct pipeline execution.

Exit criteria: IR round-trip tests pass; IR can be serialized to JSON and
reloaded.

## Phase 3 — Object Store and RDBMS Integration

**Goal**: close the loop on durable persistence with real external services.

Deliverables:

- `MinioArtifactStore` / `S3ArtifactStore` (extra: `rayorch[minio]`,
  `rayorch[s3]`).
- `SQLMetadataStore` targeting PostgreSQL and MySQL via SQLAlchemy (extra:
  `rayorch[sql]`).
- `SQLLineageSink` writing path and quarantine records to SQL tables (extra:
  `rayorch[sql]`).
- Database schema migrations managed by Alembic.
- Integration tests using Docker Compose (MinIO + PostgreSQL) gated behind a
  `--integration` pytest marker.
- State consistency rules from [`state_model.md`](state_model.md) enforced in
  `SQLMetadataStore`.
- Recovery path: on driver restart, load stale `RUNNING` jobs and mark them
  `FAILED`.

Exit criteria: integration tests pass against real MinIO and PostgreSQL
containers; lineage persists across executor restart.

## Phase 4 — Observability and TorchBackend

**Goal**: production-grade observability and a second execution backend.

Deliverables:

- `PrometheusMetricsProvider` emitting the standard metric set defined in
  [`provider_abstractions.md`](provider_abstractions.md) (extra:
  `rayorch[prometheus]`).
- `KafkaEventBusProvider` for high-volume per-row event streaming (extra:
  `rayorch[kafka]`).
- Grafana dashboard template for the standard metric set.
- `TorchBackend.build(ir)` mapping IR nodes to `torch.fx.Graph` modules for
  GPU-only pure-tensor pipelines.
- Backend registry CLI: `rayorch compile --backend ray pipeline.py` and
  `rayorch compile --backend torch pipeline.py`.
- Documentation for adding custom backends.

Exit criteria: end-to-end benchmark with Prometheus scrape and Kafka topic
verified; `TorchBackend` passes a suite of tensor-operator round-trip tests.

## Non-Goals (explicitly deferred)

The following are out of scope for all four phases:

- Multi-tenant job isolation (separate Ray namespaces per tenant).
- Shuffle, partitioning, and dataset-wide state (see
  [`runtime_dag_topology_matrix.md`](runtime_dag_topology_matrix.md)).
- Arbitrary `N:M` cardinality changes (see
  [`todos/07-runtime-emit-api.md`](todos/07-runtime-emit-api.md)).
- StageSupervisor actor (see
  [`todos/04-stage-supervisor-future.md`](todos/04-stage-supervisor-future.md)).

These are tracked in their respective TODO documents and will be scoped into a
future phase when the core runtime stabilizes.
