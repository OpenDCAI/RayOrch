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
InMemoryLineageSink
RayLineageActorSink
SQLite/DuckDB sink
Parquet/object-store sink
Postgres sink
```

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

MVP can keep full values for debugging, but production should default to metadata or references.
