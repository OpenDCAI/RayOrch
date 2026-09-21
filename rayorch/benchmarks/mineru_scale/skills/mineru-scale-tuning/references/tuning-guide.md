# MinerU scale tuning reference

Use this reference when selecting parameters, explaining a recommendation, or
diagnosing throughput, memory, scheduling, and storage symptoms.

## Parameter mapping

| Historical name | Current `MinerUScaleBench` name | Meaning |
| --- | --- | --- |
| `microbatch_size` | `input_batch_size` | PDFs in one independently reclaimable input slice |
| `max_active_microbatches` | `max_active_input_batches` | Maximum active input slices |
| `replicas` / `ocr_replicas` | `ocr_replicas` | Persistent MinerU/vLLM actors |
| `reduce_replicas` | `assemble_replicas` | Ordered document assembly actors |
| `batch_size` | `batch_size` | Maximum page grains in one OCR RPC |
| `render_replicas` | `render_replicas` | PDF-render CPU actors |
| `gpus_per_ocr_actor` | same | Ray GPU reservation per OCR actor |
| `gpu_memory_utilization` | same | vLLM utilization setting per actor |

## Historical evidence

The completed 64-GPU run used 8 workers with 8 H20 GPUs each:

```text
PDFs                       3,690
pages                    174,744
render replicas              256
OCR replicas                 128
GPU per OCR actor            0.5
assemble replicas             64
OCR batch cap                 64 pages
input batch size              24 PDFs
active input batches          24
GPU memory utilization      0.32
measured wall           3,627.363 s
throughput                48.1738 pages/s
OCR RPCs                    3,171
average pages/RPC         55.1069
OCR fill ratio             86.10%
active mean GPU util       89.867%
failed documents                1
```

On 4 H20 GPUs, the 48-PDF OCR batch sweep measured:

| OCR batch cap | pages/s |
| ---: | ---: |
| 16 | 9.4675 |
| 32 | 11.4038 |
| 64 | 12.7204 |
| 128 | 13.3608 |
| 256 | 13.5697 |

The result established that small MinerU `batch_two_step_extract()` calls paid
substantial fixed overhead even when actor scheduling bubbles were low. Batch
sizes above 128 gave diminishing gains in that experiment.

With the same 368-PDF, 4-H20 workload and `batch_size=64`, cross-parent elastic
packing improved the physical OCR shape:

```text
                         parent-bound    elastic
OCR RPCs                    368             121
average pages/RPC          19.22           58.45
fill ratio                 30.0%           91.3%
throughput                  8.691          12.162 pages/s
```

Current RayOrch performs elastic page packing. Preserve enough active input
work for it to combine pages from different documents.

The historical sweep also explored OCR caps `32/48/64/96/128/192`, one to three
actors per GPU, and vLLM utilization values `0.20/0.26/0.32/0.40/0.80`. The
frozen large-scale configuration selected two actors per H20, `0.32` memory
utilization, and a 64-page cap. Treat this as an H20-specific stability and
throughput compromise, not a universal optimum.

## Starting profiles

### Smoke

Use only for correctness and environment validation:

```text
render_replicas            2
ocr_replicas               1
gpus_per_ocr_actor         1
assemble_replicas          1
batch_size                 8
input_batch_size           1
max_active_input_batches   1
gpu_memory_utilization     0.70
input_limit                1 or 2
```

Lower `gpu_memory_utilization` further if the model and vLLM overhead do not fit.

### Conservative production

Start with one OCR actor per physical GPU:

```text
ocr_replicas             = GPU count
gpus_per_ocr_actor       = 1
render_replicas          = about 2-4 x GPU count
assemble_replicas        = about 0.5-1 x GPU count
batch_size               = 64
input_batch_size         = 24
gpu_memory_utilization   = 0.75-0.80 on validated 96-GB H20
```

Do not copy the memory value to smaller GPUs without a smoke test.

### Shared H20 throughput

Use only for 96-GB H20 or a separately validated equivalent:

```text
ocr_replicas             = 2 x GPU count
gpus_per_ocr_actor       = 0.5
gpu_memory_utilization   = 0.32
render_replicas          = 4 x GPU count
assemble_replicas        = 1 x GPU count
batch_size               = 64
input_batch_size         = 24
```

The exact 64-GPU reference used 24 active input batches. For smaller clusters,
start lower and increase only when packing or GPU activity is insufficient.

## Capacity calculations

GPU reservation:

```text
reserved GPUs = ocr_replicas * gpus_per_ocr_actor
```

Default CPU actor demand:

```text
actor CPUs = render_replicas + ocr_replicas + assemble_replicas + 1
```

Leave additional CPUs for the Ray head, dashboard, driver, HDFS clients, and
operating system. When CPU capacity is short, scale render and assemble pools
down together before reducing a GPU-fitting OCR pool.

Approximate active document capacity:

```text
active PDFs = input_batch_size * max_active_input_batches
```

This is a capacity bound rather than guaranteed residency. Increasing it gives
elastic packing more page inventory but also increases images, object-store
values, and scheduler state.

## Tuning protocol

Use a deterministic subset of at least 256-512 representative PDFs, including
long documents. Keep input order, model, Ray version, output type, and node
shape fixed.

1. Use a new output and report directory for every candidate.
2. Record `startup_s`, but compare throughput using `measured_wall_s` and
   `pages_per_s`.
3. Sweep `batch_size` first. Reject OOM, actor restart, timeout, or correctness
   failures even when throughput improves.
4. Sweep active input batches, for example `3, 8, 16, 24`.
5. Change render and assemble replicas only after OCR packing is understood.
6. Run the best two candidates on the full corpus at least three times. Alternate
   candidate order when cluster load may drift.

For comparable quality, keep `render_dpi=200`. Changing DPI changes the image
workload and is not merely a scheduler optimization.

## Report interpretation

Read `<artifact-dir>/<run-id>/summary.json`.

Primary fields:

- `metrics.pages_per_s`: throughput;
- `metrics.measured_wall_s`: execution time after actor startup;
- `metrics.failed_documents`: business failures;
- `metrics.peak_active_input_batches`: achieved input concurrency;
- `metrics.calls`: per-stage RPC and batch data.

Find the OCR call by the suffix `MinerUScaleVlmOcrPage`, then calculate:

```python
average_batch = ocr["grain_dispatches"] / ocr["rpcs"]
fill_ratio = average_batch / summary["config"]["batch_size"]
```

Useful provisional gates are OCR fill at least 80%, no increase in failed
documents, and a repeatable throughput improvement. They are not correctness
guarantees and may need adjustment for short-document corpora.

The built-in profile samples GPUs visible to the driver, not the entire remote
cluster. Use Ray Dashboard, DCGM, or node-level NVML for cluster-wide GPU
utilization.

## Symptom guide

| Symptom | Likely cause | First adjustment |
| --- | --- | --- |
| Actors remain pending | GPU or CPU request exceeds cluster capacity | Check reservation formulas; reduce render/assemble pools |
| CUDA OOM during actor initialization | Too many vLLM actors per GPU or utilization too high | Use one actor/GPU or lower `gpu_memory_utilization` |
| OOM only on page inference | OCR batch/image sizes too large | Reduce `batch_size`; retain enough input concurrency |
| OCR fill below 80% and GPU bubbles | Insufficient ready pages | Increase active input batches, then render replicas |
| OCR fill high but GPU utilization low | CPU work inside two-step extraction or slow render | Inspect render/CPU and per-node utilization before adding actors |
| GPU busy but pages/s flat | Batch cap already saturated or output/storage bottleneck | Test one larger batch; inspect assemble/storage time |
| End-of-run tail dominates | Long documents or fragmented final batches | Keep elastic packing, use a representative corpus, compare batch histogram |
| Shared storage latency rises with assemble actors | Output contention | Reduce `assemble_replicas` |
| Driver/object-store memory grows | Too much active input | Reduce `max_active_input_batches` or `input_batch_size` |

## Current output caveat

The historical 64-GPU job assembled into node-local spool directories and used
eight asynchronous upload workers to publish to Ceph. The open-source
`mineru_scale` UDF writes directly to shared POSIX or HDFS storage. Consequently,
the historical `assemble_replicas=64` may be too aggressive for a public user's
filesystem. Validate output concurrency separately and favor correctness and
stable storage latency over matching the historical actor count.
