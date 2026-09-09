# v3.6 cumulative scheduler ablation and single-parent control

> Terminology note: this report preserves the pre-release command-line labels
> `--normal-queue-order` and `--mode elastic|parent_bound` so that the recorded
> commands remain reproducible. In the current source, their corresponding
> concepts are `ready_queue_order` and
> `batching_policy="any_parent"|"single_parent"`.

## Scope

This experiment separates the paper's **cumulative** scheduler progression
from an orthogonal parent-bound control.  The input is the frozen
`/tmp/mgv3-docling-368/manifest.json` corpus: 368 ArXiv/Docling PDFs, 7,072
pages.  The v3.6 runs use four local NVIDIA H20 GPUs, four OCR replicas, four
render replicas, four assemble replicas, OCR `batch_size=64`, source
`microbatch_size=24`, `max_active_microbatches=3`, render DPI 200, and the same
real render+VLM OCR implementation.  `metadata_only` skips only final
Markdown/layout serialization; it does not skip rendering or OCR.

The paper defines FIFO as one READY FIFO per Call.  Therefore the no-FIFO arm
does **not** fall back to the old shared FIFO (which would confound PR #9's
queue-partitioning optimization).  It keeps per-Call partitioning and replaces
the ready FIFO with an unordered per-Call ready index. `_facts`, recovery
queues, lineage, suppression barriers, and ordinal reconstruction are unchanged.

## Main cumulative arms

| Arm | Stream | 1:M page rebatching | FIFO | Measured wall (s) | Interpretation |
| --- | --- | --- | --- | ---: | --- |
| Coarse streaming baseline* | yes | no | no | 818.001 | document-grain native pipeline |
| Streaming + 1:M rebatching | yes | yes | no | 634.143 | page-grain elastic dispatch without FIFO |
| Streaming + 1:M rebatching + FIFO | yes | yes | yes | 579.347 | complete v3.6 path |

The command-level switches are `--normal-queue-order fifo|unordered` and
`--mode elastic|parent_bound`.  `unordered` is intentionally an
ablation-only mode; it does not change the logical output contract.

\* The coarse streaming number is the existing 368-PDF native Flash-MinerU
reference (`document` grain, four GPUs, measured wall 818.001 s; an independent
repeat was 822.476 s), not a new v3.6 runtime switch.  It is therefore an
architectural baseline, not a strict same-runtime causal arm.  A strict
same-runtime coarse arm would require
wrapping the complete per-PDF render→OCR→assemble path in one document UDF;
that would add benchmark-specific code and is not needed to interpret the two
internal v3.6 rows.

## Why the 810-second run is not the no-FIFO arm

The previously reported `810.923 s` run is `per-Call FIFO + parent_bound`.  It
still has page-level `F.expand/F.reduce` and it **does use FIFO**.  Its
reservation policy drains siblings of one parent at a time, so each PDF gets
one OCR RPC (368 RPCs, mean batch 19.22 pages, 30.03% fill).  The batch-shape
penalty therefore dominates, and changing FIFO cannot reveal a useful FIFO
effect.

The 96-PDF interaction check confirms this: FIFO+parent-bound was 172.160 s
versus unordered+parent-bound 169.088 s, a small opposite-direction noise
difference rather than evidence that FIFO is unnecessary.  The 810-second
point should not occupy the cumulative three-row paper table.  It is useful as
a separate sensitivity/control point for forbidding cross-parent packing.

## Full 368-PDF result for the cumulative table

| Arm | Measured wall (s) | End-to-end (s) | Pages/s | OCR RPCs | Mean OCR batch | Fill ratio | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Coarse streaming baseline* | 818.001 | 860.368 | 8.6455 | document-local | document-local | document-local | 368/368 docs |
| Streaming + 1:M rebatching, no FIFO | 634.143 | 675.170 | 11.1521 | 121 | 58.45 | 91.32% | 368/368 docs |
| Streaming + 1:M rebatching + FIFO | 579.347 | 621.235 | 12.2068 | 121 | 58.45 | 91.32% | 368/368 docs |

Relative to the complete v3.6 row, removing FIFO from the rebatching path
increases measured wall by **9.46%** (634.143 versus 579.347 s) while leaving
the OCR batch histogram unchanged.  This isolates the ready-order/supply
effect of FIFO.

Relative to the existing coarse document-grain baseline, enabling runtime-owned
1:M page rebatching and then FIFO reduces measured wall by 22.5% and 29.2%,
respectively.  Because the first number is a native reference rather than a
same-runtime toggle, these are architectural comparisons; the causal internal
claim is the 634.143→579.347 s FIFO step.

The two internal v3.6 runs consumed 7,072 OCR Grains and produced 368
successful document records.  No poison was injected;
`poison_report_contract_passed` remained true as the structural output check.

## 96-PDF repeat check

The first 96 entries (1,792 pages) were rerun to check run-to-run direction:

| Arm | Measured wall (s) | End-to-end (s) | OCR RPCs | Mean batch | Fill ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| FIFO + elastic | 130.456 | 172.680 | 33 | 54.30 | 84.85% |
| unordered + elastic | 142.076 | 183.165 | 33 | 54.30 | 84.85% |
| FIFO + parent-bound | 172.160 | 215.131 | 96 | 18.67 | 29.17% |
| unordered + parent-bound | 169.088 | 211.303 | 96 | 18.67 | 29.17% |

The direction is consistent: unordered elastic is +8.91% measured wall, while
parent-bound is +31.97% relative to FIFO elastic.  With parent-bound enabled,
changing FIFO has only a small opposite-direction difference (-1.78% on this
one run), showing that batch fragmentation dominates queue-order cost.  These
are one-run checks, not confidence intervals; paper numbers should use at
least three cyclic repetitions for the two internal rows.

## Recommended paper treatment

Use the three cumulative rows above, with the coarse native number explicitly
marked as an architectural baseline.  Do not replace the coarse row with the
810.923 s parent-bound number: those runs have different logical granularity
and the latter already uses FIFO.  If a strict same-runtime cumulative table is
required, run a dedicated document-grain v3.6 UDF; otherwise avoid adding
benchmark-only framework code for the sake of a single ablation row.

The 810.923 s parent-bound point may appear in a sentence or appendix as the
negative control for cross-parent packing, not as the no-FIFO arm.

Do not use `full` versus `metadata_only` as a framework ablation: it changes
the output serializer and storage workload.  Do not remove `_facts` FIFO or
lineage state for a performance arm; that changes fixed-point propagation and
the correctness contract rather than isolating a scheduling component.

## Reproduction

The two internal full-run summaries are under `/tmp/rayorch_ablation_20260829/`
on the local machine.  The runner records the selected mode in
`normal_queue_order` and `mode`, plus OCR batch histograms and output counts in
each `artifacts/summary.json`.  The native 818.001-second reference is recorded
in `docs/experiments/multigrain_v2_5/2026-07-24_native_flash_mineru_baseline.md`.
