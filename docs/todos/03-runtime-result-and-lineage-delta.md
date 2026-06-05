# TODO: RuntimeResult and Lineage Delta

## Goal

Replace ad hoc `(MicroBatch, quarantined)` returns with a small structured result once DAG integration begins.

## Minimal Shape

```python
RuntimeResult:
    batch: MicroBatch
    quarantined: list[QuarantineRecord]
    lineage_delta: LineageDelta
```

`lineage_delta` should be append-only and cheap to merge.

## Why

DagExecutor needs one recognizable object that carries:

- healthy output rows;
- bad rows;
- lineage updates;
- mutation events;
- future parent edges for flatmap/emit.

## Non-Goal

Do not put full payload history into RuntimeResult. Payload storage should stay in MicroBatch or external object storage, not lineage metadata.
