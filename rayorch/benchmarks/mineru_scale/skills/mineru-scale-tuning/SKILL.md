---
name: mineru-scale-tuning
description: >-
  Configure and tune RayOrch MinerUScaleBench for local or multi-node GPU
  clusters. Use for parameter sizing, throughput tuning, OOM or pending-actor
  diagnosis, OCR batching, fractional GPU sharing, and runnable mineru_scale
  CLI or Python settings. Do not use for installing MinerU or changing RayOrch
  runtime internals.
---

# MinerU Scale Tuning

Produce a runnable parameter recommendation for `MinerUScaleBench`, grounded in
the repository's completed MinerU experiments and adjusted to the user's actual
cluster rather than blindly copying the 64-GPU defaults.

## Inspect the current contract

Before recommending or editing settings, read the current versions of:

- `../../benchmark.py` for exposed user parameters;
- `../../pipeline.py` for actor resources and batch defaults;
- `../../success_contract.json` for the frozen reference run;
- `../../README.md` when the user also needs launch or storage instructions.

Read [references/tuning-guide.md](references/tuning-guide.md) for historical
measurements, parameter mapping, symptom diagnosis, and the tuning sequence.

## Establish the workload facts

Use information already supplied by the user. When a missing fact materially
changes the recommendation, ask for only that fact. The important inputs are:

- physical GPU count, model, and memory per GPU;
- usable cluster CPUs and number of nodes;
- local/shared/HDFS input and output types;
- approximate PDF count and pages per PDF;
- whether the goal is smoke validation, maximum throughput, or conservative
  stability.

For a quick starting point, run:

```bash
python scripts/recommend.py --gpus <N> --profile conservative
python scripts/recommend.py --gpus <N> --profile shared-h20 --cluster-cpus <N>
```

`shared-h20` is specific to the validated 96-GB H20 layout. Do not select it
for an unknown GPU merely because fractional GPU scheduling is available.

## Preserve these invariants

- Keep `ocr_replicas * gpus_per_ocr_actor <= physical GPUs`. Equality is the
  normal full-cluster target.
- Budget one CPU for every render, OCR, assemble, and metadata actor. Reserve
  CPUs for Ray and the driver; do not produce a configuration whose actors can
  never be scheduled.
- Treat `gpu_memory_utilization` as per-vLLM-instance. Multiple actors sharing
  one GPU multiply model/runtime overhead as well as the configured fraction.
- Keep `render_dpi=200` when comparing against the historical baseline.
- Use a fresh output directory for every performance candidate so resume and
  committed-document reuse do not distort output-stage measurements.
- Current `mineru_scale` writes directly to shared POSIX/HDFS output. The old
  64-GPU run used node-local spool plus asynchronous Ceph upload, so storage
  concurrency does not transfer one-for-one.

## Tune in this order

1. Run a one- or two-PDF smoke test and verify output correctness.
2. Fit the OCR actor layout to GPU memory and the physical GPU count.
3. Sweep OCR `batch_size`, normally `32, 64, 96, 128`, while keeping the input
   corpus and other parameters fixed.
4. Increase `max_active_input_batches` until OCR packing and GPU activity stop
   improving; reduce it if object-store or process memory grows excessively.
5. Increase `render_replicas` only when OCR is starved by rendering.
6. Tune `assemble_replicas` against the real shared filesystem. Reduce it when
   storage contention rises even if more CPU actors are available.
7. Confirm the winning candidate on the full corpus with at least three runs.

Change one parameter family at a time. Compare `metrics.pages_per_s` and
`metrics.measured_wall_s`, not startup or end-to-end time alone. Require stable
document/page counts and no increase in `failed_documents`.

## Deliver the recommendation

State:

1. assumptions about hardware, CPU, storage, and workload;
2. the exact parameter values and requested CPU/GPU totals;
3. a complete CLI command or `MinerUScaleBench(...)` example;
4. the metrics to inspect, success threshold, and rollback condition;
5. which values are historically validated and which are new hypotheses.

Do not modify source, launch a paid cluster, or start a full-corpus experiment
unless the user explicitly requests that action.
