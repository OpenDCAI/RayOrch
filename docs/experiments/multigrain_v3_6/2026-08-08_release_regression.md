# v3.6 release regression

This report records the paired regression gate for the submitted v3.6
baseline. The Chinese source is retained as
[`2026-08-08_release_regression.zh.md`](2026-08-08_release_regression.zh.md).

## At a glance

### Scope

The gate compares v3.6 with the frozen v3/v3.5 workload runners while keeping
the business UDFs and input manifests unchanged. It checks both output
correctness and execution behavior; a speed result without identity and
lineage checks is not considered a pass.

The tested workloads include MinerU PDF parsing, Docling document processing,
video captioning, and video multimodal processing. The v3.6 implementation
under test is the package in
[`rayorch/experimental/multigrain_v3_6`](../../../rayorch/experimental/multigrain_v3_6).

### Gates

- static compiler and transition tests pass without starting Ray;
- optimized and unoptimized plans produce equivalent outcomes;
- empty input, repeated runs, persistent actors, and multiple microbatches
  complete without leaked state;
- actor construction failure and actor crash are recovered or reported
  according to policy;
- multi-output calls commit atomically;
- per-item failure and group-scoped suppression do not affect unrelated
  parents or calls;
- paired outputs preserve the expected item identities and ordering contract.

### Interpretation

The paired measurements are a regression guard for this source snapshot, not a
universal performance claim. New scheduler, worker, or model changes require a
fresh gate with the exact options and manifests recorded beside the results.

---

## MultiGrain V3.6 Real Workload Release Regression

> **Document niche: one re-checkable historical experiment snapshot.** This
> document proves the results of the commit and configuration described below; it
> is not a rolling guarantee for any later working tree. For the learning and
> specification document entry point, see the
> [`V3.6 documentation map`](../../multigrain_v3_6_documentation_map.md).

Date: 2026-08-08
Conclusion: pass. Under the frozen configuration, V3.6 shows no systematic
performance regression; the average relative change across the four paired gates
is all within ±2%, and the structure and identity contracts all pass.

### 1. Purpose and boundaries

V3.6 is a breaking architecture release branched from frozen V3.5 (`5f59566`).
It renames and re-partitions LogicalProgram, ProgramAnalysis, RuntimePlan,
RuntimeState, DispatchState, and the Worker DTOs, but should not change business
semantics or introduce new scheduling overhead.

This round validates the core implementation represented by commit `cdd01d0`,
together with the benchmark adapter fix discovered during testing. The gate
uniformly requires:

- two independent cold-start paired trials, with alternating execution order;
- reporting the mean, sample variance, and paired relative change;
- comparing document/video identity, structure, and business output;
- comparing RPC, average batch, actor, and release counts, to rule out silent
  structural inflation;
- running `ray.shutdown()` at the end and confirming that the machine has no Ray
  instance and no GPU compute process.

The machine uses 4×H20. Docling and the video workloads use outer wall — which
includes startup, materialization, and teardown — as the primary metric; MinerU
follows the historical runner's engine measured wall, and additionally records
startup/end-to-end.

### 2. Results summary

| Workload | Baseline | Data and configuration | Baseline mean / variance | V3.6 mean / variance | Paired average change | Correctness |
| --- | --- | --- | ---: | ---: | ---: | --- |
| MinerU | frozen V3.5 | 368 PDF / 7,072 pages; microbatch 24, active 3 | 603.889 s / 16.497 s² | 600.556 s / 8.545 s² | **-0.551%** | all 368 identities consistent; structure and OCR batch distribution consistent |
| Docling | V3 | 48 PDF / 992 pages / 512 tables; active 2 | 91.606 s / 5.103 s² | 92.742 s / 0.779 s² | **+1.260%** | identity, structure, and Markdown all exactly identical document by document; zero error/fallback |
| Video Caption | V3 | 256 videos; microbatch 32, active 4 | 86.854 s / 1.064 s² | 88.570 s / 0.717 s² | **+1.990%** | structure consistent; normalized text drift across the two rounds 0.391% / 0.293% (model non-determinism threshold <2%) |
| Video Multimodal | V3 | 256 videos; audio/frame sibling domains; 32×4 | 39.632 s / 0.054 s² | 39.827 s / 0.102 s² | **+0.492%** | output byte exact; structure, frame digest, and transcript all consistent |

A positive value means V3.6 is slower, a negative value means V3.6 is faster.
Two samples are not enough to estimate a long-term distribution, but the
alternating order, the structural diagnostics, and four different workloads show
no common direction and no extra overhead that grows with entity count.

### 3. MinerU full-368

Order and measured wall:

| Pair | Order | V3.5 | V3.6 | V3.6 - V3.5 |
| --- | --- | ---: | ---: | ---: |
| 1 | V3.5 → V3.6 | 606.761 s | 602.623 s | -4.138 s |
| 2 | V3.6 → V3.5 | 601.017 s | 598.489 s | -2.528 s |

The mean of the paired delta is -3.333 s, with sample variance 1.296 s². All
four executions produced 368 documents, 7,072 pages, and 121 OCR RPCs; the OCR
average batch was 58.446, and the complete histogram was also identical. V3.6's
total RPC was 640/642 and V3.5's was 641/640; the active high-watermark was 3 for
both, the actor count was 13 for both, and the release value was 16,352 for both.
Therefore no structural inflation caused by new Ref/Record or propagation events
was found.

The 368 Markdown path identities are exactly identical. The first round's
Jaccard median was 0.995893 with mean 0.992008; the second round's median was
0.995953 with mean 0.992861. The lowest value comes from
`hyper-connection_copy_7`; however, the Jaccard between V3.6's two own runs is
only 0.741061, lower than the 0.750666/0.919832 observed across runtimes in the
same round, which shows that it is business model non-determinism rather than an
Entity/Item mapping mismatch.

Raw artifacts: `/tmp/rayorch_v36_mineru_paired_20260808/results.jsonl` and the
four output directories in the same directory.

### 4. Docling 48-PDF

The outer wall samples are:

- V3: 93.202991 s, 90.008455 s;
- V3.6: 93.366572 s, 92.118057 s.

Both rounds produced 48 documents, 992 pages, and 512 tables; all identity,
structure, and Markdown were completely identical, and OCR/Table error and
fallback were both 0. V3 RPC was 598/604 and V3.6 was 588/628; this reflects
batch boundary changes under ready timing and did not cause any result or error
difference.

Pre-check exposed one real migration problem: the paired adapter still passed the
old parameter `max_in_fight` to V3.6, causing the V3.6 arm to fail only after the
V3 arm had already completed. It has been changed to `max_active_microbatches`,
and an adapter forwarding regression was added to prevent this kind of API
boundary flywire from appearing again.

Raw artifact: `/tmp/rayorch_v36_docling_20260808/paired-48.json`.

### 5. Video and the capacity window counterexample

The first Caption paired run used the CLI's original defaults, microbatch 4 and
active 2; V3 averaged about 123.5 s and V3.6 about 199.5 s. The frozen V3.5
control experiment with the same configuration was also about 194.9 s, and the
RPC of V3.5/V3.6 was 580/about 581 respectively, while V3 was about 390. This
shows that it is not a V3.6 refactor regression, but rather that the immediate
work-conserving scheduler adopted since V3.5 cannot form large batches under too
small an admission window.

The frozen performance configuration was microbatch 32, active 4 all along.
After the fix, Caption's V3/V3.6 RPC across both rounds is 390 for both, V3.6's
Caption stage is 67 RPC / 1,024 grains with an average batch of 15.284, and the
formal paired result is only 1.990% slower. This round also made two
reproducibility fixes:

- the video paired CLI default was changed to the frozen performance window
  32×4;
- the JSON artifact explicitly records `microbatch_size` and
  `max_active_microbatches`.

This counterexample should be kept: `microbatch_size` and
`max_active_microbatches` are capacity/throughput configuration, not pure
memory-safety knobs. Small windows are still supported, but they must not be used
to draw default performance conclusions against V3's timed-wait scheduler.

Multimodal covers two sibling Domains, audio and frame, each Expand/Map/Reduce'd
and then merged at root. V3.6's total RPC across two rounds was 1,105/1,107 and
V3's was 1,103/1,109; the outputs are completely identical, proving that this
complex lineage path produces no extra entities and omits none as a result of the
static/dynamic state split.

Raw artifacts:

- `/tmp/rayorch_v36_video_20260808/caption-256-mb32-i4-paired2.json`
- `/tmp/rayorch_v36_video_20260808/multimodal-256-mb32-i4-paired2.json`
- counterexamples: `caption-256.json` and `caption-v35-control-256.json`

### 6. Release judgment

The V3.6 version number is necessary: this round contains user-visible breaking
renames, the static-graph/runtime-state-table split, and the convergence of the
Worker ABI and Executor parameters, and is not suitable for writing back into
V3.5, whose regression is already complete. Real workloads prove that these
architecture changes bring no systematic performance regression; keeping V3.5 as
a frozen oracle also lets later optimizations continue to make paired judgments.

This round does not thereby claim that arbitrary workloads are automatically
equivalent. When adding a new primitive, scheduling policy, or admission default,
one still needs to separately verify result identity, structure, RPC/batch shape,
and end-to-end wall time.
