# State Model

Date: 2026-06-12

This document defines the authoritative state transitions for jobs, stages, and
blocks.  The rule is simple: **Ray owns in-flight execution state; the RDBMS
owns durable, queryable state**.  The two are never silently out of sync for
longer than one microbatch commit cycle.

## Responsibilities

| Concern | Owner |
|---|---|
| Whether a task is currently running | Ray (in-memory actor state) |
| Whether a job historically completed | RDBMS (`MetadataStoreProvider`) |
| Quarantine records, lineage paths | `LineageSinkProvider` |
| Large intermediate files | Object store (`ArtifactStoreProvider`) |

## Job State

A *job* is one end-to-end execution of a compiled DAG against a dataset.

```text
PENDING → RUNNING → SUCCEEDED
                  → FAILED
                  → CANCELLED
```

| State | Meaning |
|---|---|
| `PENDING` | Job record created; no actors started yet. |
| `RUNNING` | Executor has started at least one stage; microbatches in flight. |
| `SUCCEEDED` | All source microbatches produced a `RuntimeResult`; no unrecoverable errors. |
| `FAILED` | Executor encountered an unrecoverable error (actor crash, dependency failure). |
| `CANCELLED` | User or scheduler cancelled the job before it completed. |

Transitions are written to `MetadataStoreProvider` by `RuntimeDagExecutor`:

- `PENDING → RUNNING` — immediately after `executor.run()` begins dispatching.
- `RUNNING → SUCCEEDED / FAILED / CANCELLED` — when `executor.run()` returns or
  raises.

Ray actor handles are not written to the database.  If the driver crashes
mid-run, the job row remains `RUNNING`; a recovery agent can detect stale
`RUNNING` rows (no Ray actor heartbeat) and mark them `FAILED`.

## Stage State

A *stage* is one `RuntimeRayModule` node within a job.  Each stage has its own
row in the metadata store so that partial progress is visible.

```text
WAITING → ACTIVE → DONE
                 → FAILED
```

| State | Meaning |
|---|---|
| `WAITING` | Stage registered; waiting for upstream dependencies. |
| `ACTIVE` | At least one microbatch submitted to this stage's actors. |
| `DONE` | All microbatches for this stage collected successfully. |
| `FAILED` | Stage produced an unrecoverable error. |

Stage state is written by `RuntimeDagExecutor` as it dispatches and collects
each DAG node within the execution loop.

## Block State

A *block* is one document or record unit flowing through the pipeline.  Block
metadata is written to `MetadataStoreProvider`; block lineage is committed to
`LineageSinkProvider`.

```text
SUBMITTED → PROCESSED
          → QUARANTINED
```

| State | Meaning |
|---|---|
| `SUBMITTED` | Block ingested into the source microbatch. |
| `PROCESSED` | Block reached the final DAG output without errors. |
| `QUARANTINED` | Block failed at some stage; a `QuarantineRecord` was produced. |

A quarantined block does not move to `PROCESSED`.  Both states are terminal for
a single job run.  Reprocessing requires a new job with a filtered input set.

## Consistency Rules

1. Write the durable state transition to `MetadataStoreProvider` **before**
   releasing the microbatch result to downstream stages.
2. Do not write per-row state synchronously from inside user op execution.
   Buffer locally and commit via `LineageSinkProvider.commit(delta)` once per
   microbatch.
3. Ray actor handles are ephemeral.  The RDBMS row is the recovery checkpoint.
4. `ArtifactStoreProvider` puts must complete before the reference string is
   stored in `MetadataStoreProvider`.  Object-before-reference ordering ensures
   references are never dangling.

## Recovery

When the driver process restarts after a crash:

1. Load `RUNNING` jobs from `MetadataStoreProvider`.
2. Check whether the corresponding Ray actors still exist.
3. If actors are gone, mark job `FAILED` and surface the last known `ACTIVE`
   stage for debugging.
4. Future: replay from last committed lineage checkpoint if the executor
   supports partial replay (see
   [`todos/08-lineage-guided-partial-replay.md`](todos/08-lineage-guided-partial-replay.md)).
