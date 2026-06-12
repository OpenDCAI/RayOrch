# Provider Abstractions

Date: 2026-06-12

> **Status: Design draft — planned for Phase 1 and Phase 3 of the
> [roadmap](roadmap.md).**  The interfaces described here do not yet exist in
> the current codebase.  Concrete implementations backed by SQL, MinIO, and
> Kafka are Phase 3 deliverables; the no-op / in-memory stubs are Phase 1
> deliverables.

RayOrch separates *what to store* from *where to store it* through a set of
provider interfaces.  Each provider has a no-op or in-memory default so early
development does not require external services.  Production deployments replace
the defaults by injecting concrete implementations into `RuntimeDagExecutor`.

## Injection Point

```python
executor = RuntimeDagExecutor(
    pipeline,
    artifact_store=S3ArtifactStoreProvider(bucket="my-bucket"),
    metadata_store=PostgresMetadataStoreProvider(dsn="..."),
    lineage_sink=SQLLineageSink(dsn="..."),
    metrics=PrometheusMetricsProvider(gateway="..."),
    event_bus=KafkaEventBusProvider(brokers=["..."]),
)
```

Omitted providers default to their no-op or in-memory stubs.

## ArtifactStoreProvider

Manages raw binary artifacts: source PDFs, extracted images, and OCR
intermediates.  Operators receive and return lightweight reference strings;
the provider owns the bytes.

```python
class ArtifactStoreProvider:
    def put(self, key: str, data: bytes, content_type: str = "") -> str:
        """
        Upload `data` under `key` and return a canonical reference URI.
        The URI format is implementation-defined (e.g. s3://bucket/key).
        """
        ...

    def get(self, ref: str) -> bytes:
        """Download artifact identified by the reference URI."""
        ...

    def delete(self, ref: str) -> None:
        """Delete artifact.  Idempotent."""
        ...

    def exists(self, ref: str) -> bool: ...
```

Built-in implementations:

| Class | Backing store |
|---|---|
| `InMemoryArtifactStore` | Python dict (default, non-persistent) |
| `LocalFSArtifactStore` | Local filesystem path |
| `MinioArtifactStore` | MinIO / S3-compatible (extra: `rayorch[minio]`) |
| `S3ArtifactStore` | AWS S3 (extra: `rayorch[s3]`) |

## MetadataStoreProvider

Persists control-plane records: DAG definitions, job and stage state, block
metadata, image references, and OCR result indexes.

```python
class MetadataStoreProvider:
    # Job lifecycle
    def create_job(self, job: JobRecord) -> None: ...
    def update_job_state(self, job_id: str, state: JobState) -> None: ...
    def get_job(self, job_id: str) -> JobRecord | None: ...
    def list_jobs(self, state: JobState | None = None) -> list[JobRecord]: ...

    # Stage lifecycle
    def create_stage(self, stage: StageRecord) -> None: ...
    def update_stage_state(self, stage_id: str, state: StageState) -> None: ...

    # Block metadata
    def save_block(self, block: BlockRecord) -> None: ...
    def query_blocks(self, job_id: str) -> list[BlockRecord]: ...

    # Artifact references
    def save_image_ref(self, ref: ImageRef) -> None: ...
    def save_ocr_result(self, result: OcrResult) -> None: ...
    def query_ocr_results(self, block_id: str) -> list[OcrResult]: ...
```

Built-in implementations:

| Class | Backing store |
|---|---|
| `InMemoryMetadataStore` | Python dict (default) |
| `SQLMetadataStore` | PostgreSQL or MySQL via SQLAlchemy (extra: `rayorch[sql]`) |

State transitions follow the rules defined in [`state_model.md`](state_model.md).

## LineageSinkProvider

Commits execution traces (paths, quarantine records) produced by each
microbatch.  Full interface is documented in
[`todos/05-lineage-sink-backend.md`](todos/05-lineage-sink-backend.md).

```python
class LineageSinkProvider:
    def commit(self, delta: LineageDelta) -> None: ...
    def flush(self) -> None: ...
```

Built-in implementations:

| Class | Backing store |
|---|---|
| `InMemoryLineageSink` | Python dict (default) |
| `SQLLineageSink` | PostgreSQL / MySQL (extra: `rayorch[sql]`) |
| `ParquetLineageSink` | S3 / MinIO Parquet files (extra: `rayorch[parquet]`) |
| `RayActorLineageSink` | Ray actor buffer (extra: `rayorch[ray-sink]`) |

## MetricsProvider

Emits numeric metrics (counters, gauges, histograms) to an external monitoring
system.  Callsites in the executor use a stable method set; the backing system
is swappable.

```python
class MetricsProvider:
    def increment(self, name: str, value: float = 1.0, tags: dict | None = None) -> None:
        """Increment a counter."""
        ...

    def gauge(self, name: str, value: float, tags: dict | None = None) -> None:
        """Set a gauge to an absolute value."""
        ...

    def histogram(self, name: str, value: float, tags: dict | None = None) -> None:
        """Record a single observation for a histogram or summary."""
        ...
```

Built-in implementations:

| Class | Backing system |
|---|---|
| `NoopMetricsProvider` | Discards all calls (default) |
| `LoggingMetricsProvider` | Emits to Python `logging` |
| `PrometheusMetricsProvider` | Prometheus Pushgateway (extra: `rayorch[prometheus]`) |

Key metrics emitted by `RuntimeDagExecutor`:

```text
rayorch.job.started            counter   job_id
rayorch.job.completed          counter   job_id, status
rayorch.microbatch.dispatched  counter   job_id, stage
rayorch.microbatch.latency_s   histogram job_id, stage
rayorch.quarantine.count       counter   job_id, stage
rayorch.artifact.put_bytes     counter   bucket
```

## EventBusProvider

Publishes structured events to a streaming bus.  Intended for high-volume write
paths (per-row quarantine events, block completion signals) that would otherwise
place heavy write pressure on the RDBMS.

```python
class EventBusProvider:
    def publish(self, topic: str, event: dict) -> None:
        """
        Publish one event dict to `topic`.
        The call is best-effort; transient failures are logged but not raised.
        """
        ...

    def close(self) -> None:
        """Flush buffered events and close the connection."""
        ...
```

Built-in implementations:

| Class | Backing system |
|---|---|
| `NoopEventBusProvider` | Discards all calls (default) |
| `LoggingEventBusProvider` | Emits JSON to Python `logging` |
| `KafkaEventBusProvider` | Apache Kafka (extra: `rayorch[kafka]`) |

Standard event topics:

```text
rayorch.job.state_changed    {job_id, old_state, new_state, ts}
rayorch.stage.state_changed  {job_id, stage_id, old_state, new_state, ts}
rayorch.block.quarantined    {job_id, stage_id, row_id, error, ts}
rayorch.block.completed      {job_id, row_id, path_id, ts}
```

## Provider Lifecycle

Providers are not Ray actors.  They are plain Python objects created in the
driver process and passed to `RuntimeDagExecutor`.  Executors call `flush()`
or `close()` on providers that expose those methods during `executor.close()`.

Providers must be safe to call from the driver's main thread.  They must not
require Ray to be initialized.

## Summary Table

| Provider | Default impl | Production impl | Extra |
|---|---|---|---|
| `ArtifactStoreProvider` | `InMemoryArtifactStore` | `MinioArtifactStore` / `S3ArtifactStore` | `minio` / `s3` |
| `MetadataStoreProvider` | `InMemoryMetadataStore` | `SQLMetadataStore` | `sql` |
| `LineageSinkProvider` | `InMemoryLineageSink` | `SQLLineageSink` / `ParquetLineageSink` | `sql` / `parquet` |
| `MetricsProvider` | `NoopMetricsProvider` | `PrometheusMetricsProvider` | `prometheus` |
| `EventBusProvider` | `NoopEventBusProvider` | `KafkaEventBusProvider` | `kafka` |
