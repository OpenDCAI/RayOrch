# TODO: MicroBatch Dispatch and Collect

## Goal

Make RayModule dispatch and collect understand MicroBatch without special-casing user code.

## Dispatch

For `Dispatch.SHARD_CONTIGUOUS` style execution, slicing must preserve all row-aligned state:

```text
columns
row_ids
path_ids
```

The dispatch function should call:

```python
microbatch.slice(start, end)
```

instead of slicing only the payload columns.

## Collect

Replica outputs should be merged in replica order:

```text
healthy MicroBatch:
  concat columns
  concat row_ids
  concat path_ids

quarantined:
  concat records

lineage delta:
  merge append-only updates
```

## Acceptance Test

Use 16 rows, 4 replicas, and one bad row in a shard:

- all healthy rows return exactly once;
- the bad row is quarantined exactly once;
- output order is deterministic;
- row IDs remain aligned with output rows.
