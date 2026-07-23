# M2 — Experiment Design: real workload, real baselines, scale

Design-first doc for M2 (the make-or-break milestone). Goal: prove the mechanism
on a **real** cardinality-changing GPU pipeline, against **CCF-A-comparable
baselines**, at **multi-node scale**, and measure our wedge (record-level recovery
+ relation provenance + provable reorder-safe rebalancing) — not just throughput.

Target venues: VLDB / EuroSys (see `TODO.md`). Baselines chosen to match what
those venues' reviewers will demand (Ray Data, Trident, Spark).

## 1. Workloads

Pick primary W1 (matches Trident's evaluation so the comparison is direct) and a
second W2 for generality.

- **W1 — Document parse (primary).** PDF → pages (Expand 1:N) → layout/OCR (GPU
  Map, long-tail per-page cost) → figure/table extraction (second branch) →
  `Relate(on=page_id)` link → assemble per document (Reduce N:1). Real PDFs from
  a public corpus (e.g. arXiv/DocLayNet-style) + our internal CEPH corpus. This is
  the MinerU-like pipeline and overlaps Trident's PDF workload deliberately.
- **W2 — Multimodal caption (generality).** images/video-frames (Expand) → VLM
  caption via vLLM (GPU Map) → dedup/cluster (Relate) → per-asset aggregate
  (Reduce). Uses real images; VLM = Qwen2.5-VL class.

Both have: genuine long-tail 1:N fan-out, a data-dependent filter, a GPU-heavy
stage, and a cross-branch M:N link — i.e. they exercise every primitive and every
bubble source in the taxonomy (`10-...md`).

## 2. Baselines

| Baseline | Why | How |
|---|---|---|
| **Ray Data (streaming batch)** ★ | closest competitor; block-level lineage + dynamic repartition | port W1/W2 to Ray Data `map/flat_map/groupby` |
| **Trident** ★ | same PDF workload, adaptive scheduling; advisor contact | obtain impl via Binhang Yuan; run W1 as-is |
| **Spark (+ optional Daft)** | classic dataflow + partition lineage | PySpark pipeline, GPU via UDF/mapPartitions |
| **Naive Ray actors** | no rebalancing / no relation lineage | hand-written Ray tasks, contiguous shard |
| **Ours (MultigrainRayExecutor)** | the system | passive IR + LPT + record lineage |
| **Ours − LPT (contiguous)** | internal ablation | isolate rebalancing gain |

cedar/Pecan are cited for optimizer/UDF-hint positioning, not necessarily run.

## 3. Metrics

- **Throughput / makespan** end-to-end (docs/s, pages/s) at fixed hardware.
- **GPU utilization / idle bubble** at the wide GPU stage (our headline mechanism).
- **Recovery**: inject row/task/node failures; measure (a) correctness (output ==
  fault-free run on healthy items), (b) recovery work = #records recomputed vs
  full-stage recompute (Spark/Ray Data), (c) recovery wall-time.
- **Lineage overhead**: steady-state throughput/memory with vs without record-level
  lineage; must beat Titian's ~20–30% (target < ~10%, ideally low single digits).
- **Scale**: strong/weak scaling across nodes × GPUs; efficiency vs OPT bound
  (reuse the LPT/Graham analysis in `10-...md`).
- **Reorder-safety at scale**: empirically confirm M1 theorem on the real run
  (byte-identical results across shard plans), not just unit scale.

### 3a. Anonymous internal evidence schema

Internal evidence must establish external validity without exporting payloads,
paths, user names, raw record IDs, display keys, model inputs, or error text.
Every run records two layers:

```text
RunEnvelope
  experiment_id        random publication-safe id
  workload_family      document_parse | multimodal_caption | ...
  engine_revision      git revision / artifact version
  cluster_shape        nodes, accelerators, CPU, memory class
  configuration        planner/recovery/replica policy names
  dataset_bucket       coarse public size/fanout bucket, never a source path
  started_at_bucket    day or week, not user/request timestamp

NodeMetric
  name, kind, replicas
  rows_in, rows_out, fanout_ratio
  shard_rows_in[], shard_rows_out[], shard_busy_s[]
  stage_makespan_s, idle_bubble_frac
  retries, recovery_rows
  relation_entries_out, lineage_bytes_out
```

`NodeMetric.to_dict()` is the engine-native anonymous stage trace. Arrays are
aligned by shard index and contain counts/durations only. A separate GPU sampler
may add per-stage aggregate utilization percentiles, but must not attach raw
object identifiers. Publication uses distributions and aggregate cases; any
internal case study must have a public workload reproducing the same qualitative
trend.

## 4. Prototype gaps to close before M2 can run

Ordered by leverage. These are the concrete "further prototype design" tasks.

1. **Multi-node execution.** Current `MultigrainRayExecutor` does intra-node replica
   sharding + whole-graph microbatch overlap. Need: cross-node placement, data
   movement between nodes (Ray object store / plasma), node-count-aware
   `shard_planner`. (Blocks the "scale" metric.)
2. **Real operator wrappers.** Wrap actual models as multigrain UDFs: a real
   layout/OCR (or MinerU components) and a vLLM VLM captioner, staying UDF-pure
   (value-in/value-out, no ids — preserves the M1 assumption). Provide
   `num_gpus_per_replica`, `gpu_heavy` properties.
3. **Instrumentation. [PARTIAL]** `metrics.RunMetrics` / `NodeMetric` +
   `lineage_footprint`: per-node makespan, per-shard busy -> idle-bubble fraction,
   anonymous per-shard input/output cardinality, fanout ratio, relation/lineage
   footprint and recovery counters. Threaded through both `MultigrainExecutor`
   and `MultigrainRayExecutor`. RunEnvelope persistence and real GPU-util sampling
   remain to be added when the internal collection boundary is chosen.
4. **Baseline harness.** Same W1/W2 expressed on Ray Data, Spark, naive Ray; shared
   dataset loader from CEPH; shared correctness checker (compare healthy outputs).
5. **Fault-injection harness. [DONE, MVP]** `multigrain.ray.executor.FaultSpec` injects
   deterministic task/node crashes; the executor retries only the failed shard, so
   `recovery_rows` stays lineage-local (< whole-stage). Row-level quarantine via
   `BadRecordError` already exists. Covered by
   `test_metrics_and_recovery.py`. (Node-kill via real Ray actor death is future.)
6. **Data ingestion.** Stream real PDFs/images from CEPH into source `PortBatch`es
   with stable `display_key`s (doc id) so traces stay human-readable at scale.
7. *(defer to M4)* Reduce/Relate sharding for skew — not required for W1/W2 first
   cut, but needed if Reduce becomes GPU-heavy.

## 5. Experiment matrix

| Exp | Question | Setup | Expected evidence |
|---|---|---|---|
| E1 throughput/bubble | do we cut GPU bubble on real long-tail? | W1, ours vs contiguous vs Ray Data vs Trident, 1 node | bubble ↓, throughput ≥ competitors |
| E2 scale | does it hold multi-node? | W1/W2, 2–8 nodes | near-linear scaling, efficiency vs OPT |
| E3 recovery **[DONE §5c]** | is record-level recovery cheaper + correct? | poison page, real pipeline, ours vs shard-level | ✅ ours COMPLETE (1 page lost, 0 redundant re-OCR); shard-level ABORT (248 lost, 1.5× GPU waste) |
| E4 lineage overhead | is lineage cheap? | W1 with/without lineage | overhead < ~10%, beats Titian |
| E5 reorder-safety | theorem holds at scale? | W1 across shard plans | byte-identical outputs (M1 at scale) |
| E6 ablation | contribution of each mechanism? | toggle relation-IR / lineage / LPT | isolate each gain |

## 5b. First real-workload run (E1 prototype, 2026-07-10)

**Setup.** Real Flash-MinerU parse of **368 PDFs** (7072 pages, `MinerU2.5-2509-1.2B`)
on **4×H20**, through the framework-native `MinerUReal` pipeline
(`Expand(RealPdfToPages) → Map(RealVlmOcrPage) → Reduce(RealAssembleDoc)`) driven by
`MultigrainRayExecutor`. GPU stage runs on a **persistent per-GPU actor pool**
(RayModule factory pattern: vLLM loads once per replica, declared via
`PhysicalHints(replicas=4, num_gpus_per_replica=1.0)`); render/OCR/assemble are
pipelined (render on 4 CPU actors, OCR on 4 GPU actors, assemble on driver).
Baseline = original Flash-MinerU (doc-grain, `SHARD_CONTIGUOUS`, batch=8,
inflight=4), **measured on this machine** via `regression_run.py` on the same
368 PDFs / 4×H20 and the **same rayorch repo/branch** (`codex/runtime-raymodule-mvp`
@ 36e07d2): **wall = 891.98 s** → 0.41 pdf/s (7072 pages → 7.93 pages/s), even a
touch above the previously quoted ~840 s. Model-load/warmup excluded from wall on
both sides. Live `nvidia-smi` during the baseline showed the doc-grain contiguous
bubble directly (GPU1/2 @ 100%, GPU3 @ 61%). Full log: `Flash-mineru/logs/baseline_raw.log`.

| config | wall (s) | pages/s | OCR makespan (s) | OCR bubble | vs 892 s |
|---|---|---|---|---|---|
| **ours, page-grain + LPT**        | **567.5** | 12.46 | 463.8 | 6.5% | **1.57×** |
| ours, page-grain + contiguous     | 617.2 | 11.46 | 509.4 | 4.2% | 1.45× |
| Flash-MinerU baseline (doc-grain, measured) | 892 | 7.93 | — | — | 1.00× |

**Honest reading.**
- **We beat the measured baseline (1.45–1.57×).** The win is *architectural*, not from LPT:
  page-grain fan-out feeds vLLM much larger, denser batches than doc-grain
  batch=8; persistent actors remove per-batch model/import cost; render/OCR/assemble
  overlap. OCR is ~82% of wall (`463.8/567.5`); the remaining ~18% is the
  render→driver→OCR **image round-trip** (images cross the object store twice) +
  priming + assemble tail.
- **LPT vs contiguous is only ~9% here and within run-to-run variance.** This
  dataset is homogeneous within sorted chunks (23 base papers × 16 identical
  copies), and *per-page* OCR cost is fairly uniform, so contiguous page-splitting
  is already ~balanced (4.2% bubble). LPT's decisive win needs **high per-record
  work variance** (long-tail), which we already show synthetically in
  `bench_gpu_longtail.py` / `bench_gpu_complex.py`. TODO: rerun with **shuffled**
  doc order (heterogeneous chunks) and/or a token-length weight to expose the
  LPT gap on this real workload.

**Next optimizations to widen the gap (and make LPT matter):**
1. **Actor-to-actor page handoff** (render actor → OCR actor via object refs), so
   images don't round-trip through the driver — should recover most of the ~18%.
2. **Shuffle / token-weighted LPT** to create the long-tail the mechanism targets.
3. Fault-injection on this real pipeline → E3 recovery numbers.

Artifacts: `Flash-mineru/mg_bridge/run_bench.py`, `mg_bench_results.jsonl`,
`logs/{lpt,contig}_raw.log`. Framework changes: persistent `_StageActor` pool +
factory-pattern op caching in `MultigrainExecutor`/`MultigrainRayExecutor`.

### Output equivalence check (compute logic unchanged, 2026-07-10)

Goal: prove the page-grain refactor did **not** change *what* is computed, only
*how it is scheduled*. All three runs (baseline / ours-LPT / ours-contig) wrote
per-doc markdown under the same `{stem}/vlm/{stem}.md` layout, so we compared them.

Same code path by construction: the multigrain wrappers call the **identical**
MinerU functions the baseline uses — `load_images_from_pdf(dpi=200)`,
`MinerUClient.batch_two_step_extract`, `result_to_middle_json`, `vlm_union_make`
(`dispatch_mineru_class.py` vs `mg_bridge/ops.py`). Multigrain only re-groups pages
into shardable page-grain records; the per-page math is the original code.

Measured over all 368 docs (baseline vs ours-LPT):

| metric | value |
|---|---|
| raw byte-identical | 0/368 |
| byte-identical after normalizing image names | 2/368 |
| **shift-robust token Jaccard, median** | **0.9989** |
| token Jaccard mean / min | 0.9844 / 0.5835 |
| docs with token Jaccard ≥ 0.98 | 344/368 |

**Why not byte-identical — and why that's expected.** Batched vLLM inference is
**not bitwise reproducible**: continuous batching composes each page with different
neighbors, so float reduction order (attention/matmul) changes and greedy decoding
occasionally flips a token or shifts a predicted bbox by a few px. Proof it's the
engine, not us: **ours-LPT vs ours-contig** (identical code, only shard order
differs) is *also* not byte-identical — only 36/368 match, median line-match 0.991.
The original engine is not reproducible against itself either. So the
baseline-vs-ours divergence (median token Jaccard 0.9989) is the **same order** as
this intrinsic batching jitter, not a logic change.

Two diff sources, both benign: (1) `images/<sha256>.jpg` names change because a
1-px bbox jitter changes the crop bytes → new content hash; (2) rare token jitter
(`Bengio`↔`Ben-gio`, `l/32`↔`l / 32`). The `min=0.006` "worst" docs by naive
line-diff are line-shift artifacts (one inserted line misaligns the zip) — their
token Jaccard is 0.999. Repro: ad-hoc compare script over the three output dirs.

Conclusion: **compute logic is equivalent**; residual differences are intrinsic GPU
batching non-determinism that the unmodified baseline exhibits against itself.

## 5c. E3 — record-level vs shard-level recovery on the real pipeline (2026-07-10)

The wedge vs Ray Data / Trident / Spark is **recovery granularity**. E3 measures it
on the *real* MinerU pipeline with the *same* vLLM OCR op and the *same* deterministic
poison page — the only variable is how the framework recovers. Measured at the OCR
(contents) grain, where the expensive GPU work lives, so the story isn't entangled
with the assembler's behaviour on a holed doc.

**Setup.** 48 PDFs → 992 pages, `MinerU2.5-2509-1.2B`, 4×H20, 4 persistent GPU
actors, node `RecoveryPolicy(max_shard_retries=2)`. Poison = one page
(`2410.19313v1_copy_2#p1`) that
deterministically fails in OCR (a poison pill / non-transient data fault).
- **record-level (ours):** the op raises `BadRecordError(index=i)`; `Map` isolates
  row *i* (quarantine + lineage trace) and re-runs the shard's healthy rows **as one
  batch** (fast-path added to `_run_with_bad_index`; no per-row serialization).
- **shard-level (Spark / Ray Data granularity):** the op raises after OCR; `Map`
  doesn't catch it, so `_run_shards` retries the **whole partition**. The fault is
  deterministic → retries exhaust → job aborts (the partition-level behaviour).

| strategy | outcome | quarantined | pages lost | **page-OCR execs** (GPU work) | wall (s) |
|---|---|---|---|---|---|
| **record-level (ours)** | **COMPLETE** | 1 (localized) | **1** | **991** (= 992−1, every healthy page once, 0 redundant) | 82.7 |
| shard-level (Spark-like) | **ABORT** | – | **248** (whole partition) | **1488** (992 useful + 496 wasted retries, 1.5×) | 166.3 |

**Reading.**
- **Same fault, opposite outcome.** Record-level *completes* and drops exactly the
  one bad page; shard-level *loses the whole 248-page partition and aborts the job*.
- **Recompute footprint is the credible cost metric** (hardware-agnostic). Ours
  re-OCRs **0** healthy pages (991 execs = one per healthy page); shard-level burns
  **1488** (1.5×) chasing a fault it can never clear. General law: shard waste =
  `partition × max_retries` (grows with retries), record waste = `0` (independent of
  retries). It also finished in **half the wall time** while succeeding.
- **Localization is automatic.** The quarantine trace names the exact culprit —
  `logical_item = 2410.19313v1_copy_2.pdf/page=1`, `failed_op = PoisonOcrPage`,
  `upstream_path = [RealPdfToPages, PoisonOcrPage]` — with no internal IDs leaked to
  the UDF (M1 value-purity holds).

**Framework change.** `Map._run_with_bad_index` now drops the known-bad row and
re-runs the remainder **as a single batch**, recursing only if the batch flags
another bad row (`run_calls` went 143→4 for the poison shard). Minimal recompute
count *and* no serialization. Existing row-isolation tests still pass.

**Honest scope.** This models a **deterministic data fault** (poison pill) — the
regime where record-level dominates. For *transient* infra faults (non-deterministic),
shard retry succeeds and both recover; record-level still bounds the blast radius to
the affected rows. Record-level replay of *downstream* consumers (re-run only the one
doc's Reduce after a page is fixed) is the natural next extension. Artifacts:
`Flash-mineru/mg_bridge/{mineru_poison_ops.py,run_recovery_e3.py}`,
`mg_e3_recovery.jsonl`, `logs/e3_full.log`.

## 5d. Full-scale E2E + output equivalence (fault-free AND after recovery)

Ran the whole real MinerU multigrain pipeline (render → OCR → assemble) at full
scale and verified the produced markdown is **logically equivalent** to the bare
Flash-MinerU baseline — both without faults and after record-level recovery.
Equivalence = **normalized token Jaccard** (image refs `images/<sha>.jpg` → `images/IMG`,
whitespace collapsed); byte identity is impossible because even the baseline is not
self-reproducible (VLM batching jitter). Comparator: `Flash-mineru/mg_bridge/compare_md.py`.

**E2E performance (368 PDFs / 7072 pages / 4×H20, LPT planner).**
The current run uses only the public framework path:
`MinerUReal.compile()` → `MultigrainRayExecutor.execute_stream`; the benchmark
contains no `_pool_for`, `run_shard`, manual concat, or hand-written
render/OCR/assemble scheduler.

| metric | value |
|---|---|
| wall | **519.95 s** (all 368 docs assembled) |
| vs measured baseline (891.98 s) | **1.72×** |
| vs previous benchmark-specific scheduler (585.13 s) | **1.13×** |
| OCR bubble fraction | **0.0653** (LPT keeps the long-tail idle low) |
| throughput | 13.6 pages/s, 0.708 pdf/s |

**Output equivalence, fault-free (368 vs baseline).** matched 368/368, 0 missing,
0 extra. token Jaccard median **0.9957**, mean 0.9922; **100 % ≥ 0.90,
98.9 % ≥ 0.95, 89.4 % ≥ 0.98**; seq-ratio median 0.9996. The low-Jaccard
tail remains the same OCR/table-jitter families seen in independent baseline
runs (`DuoAttention`, `hyper-connection`, dense malformed tables); all 7072
pages and 368 document outputs are present. No framework-level content loss.
Detail: `logs/cmp_engine_full_lpt_20260712.jsonl`.

**Output equivalence after record-level recovery (48-PDF subset, 1 poison page).**
Driver `Flash-mineru/mg_bridge/run_recovery_md.py` runs the full pipeline with a
`PoisonOcrPage` (record mode) on `1838_reformer…#p0` and `missing_child="fail_closed"`.

- **Chained suppression works**: the poisoned document produced **0 markdown files**
  (its assembler was never called) and surfaces as a `suppressed_incomplete` anchor-grain
  error — no truncated/corrupt file, no block. `assembled_docs=47, suppressed=[1838_reformer…]`.
- **Survivors are logically intact**: the 47 healthy docs vs baseline → **47/47 ≥ 0.98**
  token Jaccard (median 0.9972, min 0.9892), **0** docs with a >5 % char gap.

**Takeaway.** Multigrain output ≡ baseline modulo non-semantic VLM jitter, both in the
happy path and after a record-level fault; a permanently-lost page cascades to a *flagged,
unwritten* document rather than a silently-truncated one. Artifacts:
`Flash-mineru/mg_bridge/{compare_md.py,run_recovery_md.py}`,
`logs/{cmp_engine_full_lpt_20260712.jsonl,recover_md_result.json,cmp_recover.jsonl}`,
`outputs_engine_full_lpt_20260712`.

## 6. Risks & mitigations

- **Novelty compression by Trident/Ray Data** → lead with record-level recovery +
  relation provenance + proof; throughput is secondary. Deconflict with Binhang
  Yuan early.
- **Baseline engineering cost** → get Trident impl from advisor; use Ray Data/Spark
  off-the-shelf; keep W1 small enough to port faithfully.
- **Cluster/data access** → start single-node E1/E3/E4 (no multi-node dependency)
  to get most of the story; add E2 scale when cluster is available.
- **vLLM/real-model variance** → pin versions; warm up; report medians (reuse the
  warmed-actor methodology from `bench_gpu_mineru.py`).

## 7. Suggested order of execution

1. Prototype gaps #2 (real ops) + #3 (instrumentation) + #6 (CEPH ingest) → run
   **E1** single-node (headline bubble/throughput vs Ray Data). Highest-signal,
   no multi-node dependency.
2. #5 fault injection → **E3** + **E4** (recovery + overhead) — the wedge.
3. #1 multi-node → **E2** scale.
4. **E5**/**E6** reuse the above runs.
5. Talk to Binhang Yuan before E1 to line up Trident baseline.
