# TODO: Lineage Sink Backend

## Goal

Separate runtime lineage generation from lineage persistence.

## Direction

Runtime should produce deltas:

```text
paths
row_path updates
quarantined records
future parent_edges
future optional mutation events
```

Executor or supervisor should commit those deltas to a sink:

```text
InMemoryLineageSink      (MVP default)
RayActorLineageSink
SQLite/DuckDB sink
Parquet/object-store sink
Postgres sink
```

## LineageSinkProvider Interface

`LineageSinkProvider` is the canonical abstraction.  The executor receives it
at construction time and calls `commit(delta)` once per microbatch after all
stage outputs are collected.

```python
class LineageSinkProvider:
    def commit(self, delta: LineageDelta) -> None:
        """
        Persist one microbatch worth of lineage.

        delta contains:
          paths          – new PathRecord objects
          row_path       – row_id -> path_id mapping updates
          quarantined    – QuarantineRecord list
          parent_edges   – multi-parent join edges (future)
          mutations      – field-level mutation events (future)
        """
        ...

    def flush(self) -> None:
        """Force any buffered writes to the backing store."""
        ...
```

Concrete implementations ship as optional extras so the core runtime does not
depend on SQL drivers or object-store clients:

| Class | Backing store | Package extra |
|---|---|---|
| `InMemoryLineageSink` | Python dict | *(included)* |
| `SQLLineageSink` | PostgreSQL / MySQL | `rayorch[sql]` |
| `ParquetLineageSink` | S3 / MinIO via Parquet | `rayorch[parquet]` |
| `RayActorLineageSink` | Ray actor buffer | `rayorch[ray-sink]` |

## Rule

Do not synchronously write lineage per row from inside user op execution.

Prefer:

```text
local buffer
batch finalize
commit delta
```

## Payload Policy

Quarantine payload should become configurable:

```text
metadata
ref
full
```

MVP can keep full values for debugging, but production should default to
metadata or references.

## Relation to MetadataStoreProvider

`LineageSinkProvider` commits execution traces (paths, quarantines).
`MetadataStoreProvider` commits control-plane records (job state, block
metadata, OCR indexes).  Both live under the same provider injection point in
`RuntimeDagExecutor`, but remain separate interfaces so they can use different
backing stores (e.g. Parquet for lineage, PostgreSQL for metadata).
