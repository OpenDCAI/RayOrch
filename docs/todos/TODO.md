# Paper Roadmap (macro TODO)

Living roadmap that turns the multigrain prototype into a top-venue paper. Each
milestone says **why it matters for acceptance**, its **deliverables**, and its
**dependencies**. Detailed design lives in the numbered docs; this file is the
map. Keep statuses current.

> **How-to for users:** see [`../multigrain_user_guide.md`](../multigrain_user_guide.md)
> — the hands-on guide covering the UDF contract, primitives, Ray execution,
> and the fault/recovery (`BadRecordError`, `missing_child`) patterns.

## Thesis (the one sentence the paper defends)

> Fine-grained **lineage** is a single substrate that simultaneously drives
> **row-level recovery** and **long-tail rebalancing (bubble elimination)** in
> cardinality-changing (1:N / M:N) object pipelines — and reordering for
> performance provably preserves both results and lineage.

Wedge vs prior art: Spark has lineage but coarse (partition-level) and only for
recovery; Ray Data has GPU pipelining but no record-level lineage/recovery.
"Record-level lineage that serves recovery *and* scheduling in basis-changing
object pipelines" is the gap.

**Venue policy: target CCF-A.** MLSys fits the story but is *not* CCF-A, so it is
at most a fast preprint/backup, not the primary target. CCF-A candidates by
fit × hit-rate: **VLDB and EuroSys** (both A, best fit) > **USENIX ATC** (A,
data-prep friendly) > **OSDI / NSDI / SOSP** (A, reach; need M4 strong + beat Ray
Data/Trident at scale). See the "Related Work & Venue Strategy" section below.

## Status at a glance

| Milestone | Topic | Status | Blocks acceptance? |
|---|---|---|---|
| M0 | Prototype: relation IR + passive IR + primitives + passes + unified public Ray stream executor | done (generic DAG coordinator; persistent pools; bounded inflight; MinerU benchmark uses public API only) | prerequisite |
| M1 | Formalization: relation algebra + reordering-invariance theorem | **done** | yes (else "just engineering") |
| M2 | Real workload + real baselines + multi-node scale | **in progress** (E1 public-engine 368-PDF: 1.72× vs measured 892s baseline, output-equivalent; competitor baselines + multi-node TODO) | **yes (make-or-break)** |
| M3 | Fault-tolerance & overhead: row-level recovery vs full recompute; lineage cost | **in progress** (E3 done: record-level COMPLETE + 0 redundant re-OCR vs shard-level ABORT + 1.5× waste on real pipeline; lineage-overhead E4 TODO) | yes |
| M4 | New systems mechanism: kill residual bubbles (#4 stage barrier, #5 reduce skew) | not started | yes (differentiator) |
| M5 | Ablations + sensitivity + artifact + writing | not started | yes (polish) |

Minimal acceptable set: **M1 + M2 + M3 + (one of M4)**. M5 makes it competitive.

Primitive 内核已完成声明、输出 builder、handler registry、capability 与 structural
verifier 收敛；设计边界和重复矩阵见
[`18-multigrain-primitive-core-convergence.md`](18-multigrain-primitive-core-convergence.md)。
Expand mixed outputs、`mg.out.same/children` relation forest、现实 workload 覆盖和未来
Partition/global/window/async 扩展门已记录在
[`19-expand-mixed-output-relation-design.md`](19-expand-mixed-output-relation-design.md)；
该设计当前为 **deferred**，不阻塞框架理解、内核收敛或 M2/M3。

Deferred primitive authoring/compiler follow-ups（当前不优先）:
- [ ] **Expand mixed-output relation override**：保留现有 Expand 默认 shared-child
      语义，只在不可拆分的 corner case 中用 typed `mg.out.same/children` 表达
      parent metadata、独立 child cohort、sibling alignment 和 nested fanout。
      实现前重新审查 doc 19 的 R1–R3，禁止直接扩成万能 Transform/emit DSL。
- [ ] **AST-based UDF output-arity inference**：在不实例化或执行 UDF 的前提下，从
      importable `run()`/callable 源码推导 `num_outputs`。分析器返回
      `Exact(n) / Inconsistent(arities) / Unknown(reason)`：所有 reachable returns
      arity 一致时自动生成 IR output ports；分支不一致或存在隐式 fallthrough 时编译
      失败；动态 call、starred tuple、source unavailable 等未知情况要求用户显式声明。
      显式声明与可推导结果冲突时编译失败，运行时 `checked_output_lists()` 校验继续保留。
      实现应位于 authoring/compiler introspection 层，IR 仍只保存确定的整数 arity。

---

## M1 — Formalization  ✅ done

Goal: promote the empirical "reordering doesn't corrupt results/lineage" into a
theorem, so the work reads as research, not plumbing.

Deliverables (all landed):
- [x] Relation-algebra + physical-execution semantics grounded in code
      → `13-reordering-invariance-theorem.md`.
- [x] Reordering-Invariance theorem: part 1 (keyed/lineage `≈` everywhere),
      part 2 (byte-identical ordered equality at canonical Reduce outputs);
      corollaries: lineage-guided recovery is reorder-stable, LPT is safe.
- [x] Honest assumptions: WF co-ordered inputs, whole-batch Reduce/Relate,
      determinism, UDF value-purity (id/position independence).
- [x] Content-addressed relation ids (so key-join is strictly `≈`).
- [x] Machine-checked property test over random legal shard plans
      → `test/experimental/multigrain/test_reordering_invariance.py` (25×2, green).

Follow-ups (optional, do lazily): re-prove Lemma 2 under a sharded two-phase
Reduce (needed by M4 #5); lift WF via by-id shard partitioning.

---

## M2 — Real workload + real baselines + scale  ← NEXT, highest risk

Goal: prove the mechanism on a real pipeline against real systems at real scale.
This is what makes or breaks a systems submission.

Deliverables:
- [x] Slow real-artifact graph suite (real MinerU images, deterministic UDFs):
      nested Expand, diamond, true M:N, multi-root, Ray/LPT equivalence and
      measured CPU proxy bubble reduction. Findings:
      [`17-mineru-graph-integration-findings.md`](17-mineru-graph-integration-findings.md).
- [ ] Pick 1–2 real end-to-end workloads with genuine long-tail 1:N/M:N and GPU
      stages, e.g. real MinerU PDF→Markdown parse; and/or a multimodal prep
      pipeline (doc chunk→embed, image caption, video frame→caption). Real data.
- [ ] Baselines: Ray Data, Spark (or Daft/Dask), naive Ray-actor pipeline.
      Same workload, same hardware.
- [ ] Metrics: end-to-end throughput/latency, GPU utilization / idle bubble,
      cost. Show our speedup + equal correctness.
- [ ] **Multi-node** (≥2–4 nodes × GPUs). Single-node 4×H20 is not enough.
- [ ] Design doc first (workload spec, dataset, baseline configs, metric
      definitions, cluster plan) before spending GPU hours.

Dependencies: M0/M1. Risk: baseline engineering + cluster access + data. Start
the design doc now; it is cheap and de-risks the rest.

---

## M3 — Fault tolerance & overhead (the recovery half of the thesis)

Goal: demonstrate the "lineage → recovery" payoff and prove lineage is cheap.

Design spec for the recovery tier taxonomy (retry gradient: inline/deferred,
record/shard, degrade vs abort, stage-global drain): see
[`15-recovery-tiers-and-retry-scheduling.md`](15-recovery-tiers-and-retry-scheduling.md).
Batch-scoped UUID identity, arena lifetime, and low-rank/hash-consed lineage:
[`16-batch-arena-and-compact-lineage.md`](16-batch-arena-and-compact-lineage.md).

Deliverables:
- [x] Fault-injection harness: kill records / tasks / a node mid-run; includes
      bounded real-image integration coverage for inline/deferred retry,
      adaptive isolation, dense failure, fail-closed cascade, and actor death.
- [ ] Measure recovery cost & correctness: only affected records recomputed vs
      full recompute (Ray Data / Spark comparison where possible).
- [x] RecoveryPolicy + nested IsolationBudget passive IR/user interface;
      unsupported node-kind combinations rejected explicitly.
- [x] Recovery tiers for Map/shardable MVP (doc 15): inline/deferred record
      retry, immediate healthy-replica shard retry, bounded adaptive
      localization, actor replacement, and bounded stage-epoch drain. Remaining:
      wrapper-specific attributable units for Filter/Expand/Reduce/Relate.
- [ ] Overhead study: cost of record-level lineage tracking in steady state
      (must be small, e.g. < a few %); memory footprint of lineage; effect of
      `MaterializePolicy` boundaries.
- [ ] Compact-lineage substrate (doc 16): coordinator UUID/BatchArena → shared
      1:1 path table → columnar 1:N → Reduce/M:N compact relations → arena
      lifecycle/release.
- [ ] Tie back to M1 corollary: recovery localization is reorder-invariant
      (deferred retry is a legal reordering — no new proof needed).

Dependencies: M2 (needs the real pipeline to inject into).

---

## M4 — New systems mechanism (differentiator; do at least one)

Goal: contribute a scheduling/execution mechanism, not just "we applied LPT",
by killing a stated residual bubble from the taxonomy in
`10-multigrain-ir-mvp-plan.md` §"bubble taxonomy".

Options (pick ≥1, #5 preferred as it is the sharpest):
- [ ] **#5 Reduce fan-in skew**: sharded two-phase / bucketed / ring Reduce with
      hot-key splitting; requires a monoid (assoc.) reduce UDF; re-prove Lemma 2
      under sharded Reduce. Biggest research payoff.
- [ ] **#4 Inter-stage barrier**: per-stage streaming scheduler (replace
      whole-graph microbatch overlap) so stage N+1 starts on ready items.
- [ ] Adaptive re-optimization: re-plan shards from observed runtime weights.

Deliverables: mechanism + correctness argument (extends M1) + measured bubble
reduction on M2 workload vs the MVP whole-batch/contiguous version.

Dependencies: M1 (proof extension), M2 (to measure on real load).

---

## M5 — Ablations, sensitivity, artifact, writing

Goal: make the evaluation airtight and reproducible.

Deliverables:
- [ ] Per-mechanism ablation: relation IR / automatic lineage / LPT rebalance /
      M4 mechanism — isolate each one's contribution.
- [ ] Sensitivity: tail heaviness (Pareto α), #GPUs, #nodes, batch/microbatch,
      grain depth. Tie measured to the analytic bounds already in
      `10-multigrain-ir-mvp-plan.md`.
- [ ] Reproducible artifact (scripts, env, dataset pointers).
- [ ] Related-work positioning: Spark/RDD lineage, Ray Data/Datasets, Naiad/
      timely, Dryad/DryadLINQ, tf.data, lineage systems (Nectar), modern LLM
      data-prep stacks. Sharpen the wedge.
- [ ] Paper draft: intro → model/IR → theorem → mechanisms → eval → related.

Dependencies: M2–M4.

---

## Related Work & Venue Strategy (CCF-A focus)

### CCF-A venue map

| Venue | CCF | Fit | Note |
|---|---|---|---|
| VLDB | A | high | relation algebra + provenance + optimizer; primary target |
| EuroSys | A | high | mechanism + fault tolerance + scheduling + theorem; primary target |
| SIGMOD | A | high | DB framing alternative to VLDB |
| ICDE | A | med-high | pipelined fault tolerance lands here (e.g. Quokka) |
| USENIX ATC | A | high | data-prep friendly (Pecan); solid A backup |
| OSDI / NSDI / SOSP | A | reach | need M4 strong + multi-node + beat Ray Data/Trident |
| MLSys | **not CCF** | high | preprint/backup only, not primary (policy: CCF-A) |

### Comparable work (must cite; the ones marked ★ are must-beat baselines)

| Work | Venue | What | Relation to us |
|---|---|---|---|
| ★ Ray Data "Streaming Batch Model" | arXiv (NSDI/OSDI-tier) | partition-level lineage recovery + dynamic repartition, 3–8× | our closest competitor; **block-level** lineage, memory-based repartition, no proof, no record-level relations/recovery |
| ★ Trident | arXiv 2026 | adaptive scheduling for heterogeneous multimodal pipelines; **same PDF-parse+layout+LLM-OCR workload** | scheduling-focused; advisor 袁彬航 (Binhang Yuan) is our contact — use for exact baseline + deconflict |
| cedar | VLDB 2025 | unified ML input pipeline + optimizer + black-box UDF **hints** | closest to our "relation contract keeps UDF pure"; cite for optimizer/UDF-independence framing |
| Pecan | ATC 2024 | transformation reordering + hybrid placement | preprocessing scheduling; different mechanism |
| Youmu | MLSys 2025 | columnar page-level IO/shuffle for LLM training | data loading, not cardinality-changing relations |
| OVERLORD | arXiv | disaggregated dataloader, DGraph per-source lineage | source-level lineage + checkpointing |
| Titian / Newt / Smoke | VLDB/SoCC/SIGMOD | record-level provenance in Spark/DB | provenance without GPU rebalancing; ~20–30% overhead (our overhead must beat this) |
| Quokka | ICDE 2024 | write-ahead lineage for pipelined query recovery | recovery-focused, no relation/grain + no GPU bubble |

### The wedge (what none of them occupy)

**Record-level, relation-aware lineage (Expand/Reduce/Relate M:N) used as one
substrate for BOTH row-level recovery AND cardinality-aware GPU rebalancing, with
a reordering-invariance theorem.** Ray Data/Trident: block-level lineage +
memory/throughput repartition, no record-level relations, no proof. Provenance
systems (Titian/Smoke): record-level lineage but no GPU rebalancing. We sit in the
intersection.

Do **not** sell on throughput alone (Trident/Ray Data are already strong there).
Sell on record-level recovery + relation provenance + provable reorder-safety;
report competitive-or-better throughput as a bonus.

### Trident / advisor leverage (action item)

Binhang Yuan (Trident advisor) is a contact. Early conversation to: (a) draw the
scope boundary (ours = relation lineage/recovery/theorem; Trident = scheduling —
orthogonal, composable); (b) obtain Trident as an exact baseline or joint eval;
(c) get his venue read (VLDB/EuroSys/OSDI). This both de-risks M2 and raises
acceptance odds. Track in M2.

## Index of detailed docs

- `09-multi-grain-port-cardinality-api.md` — API / paradigm design.
- `10-multigrain-ir-mvp-plan.md` — IR MVP plan, passes, Ray exec, bubble taxonomy, GPU benches.
- `11-multigrain-primitive-api-ir-review.md` — per-primitive API/IR review.
- `12-relation-model-three-tiers.md` — relation model (Expand/Reduce | on= | adapter).
- `13-reordering-invariance-theorem.md` — M1 formalization + proofs.
- `14-m2-experiment-design.md` — M2 workloads, baselines, metrics, prototype gaps.
- `15-recovery-tiers-and-retry-scheduling.md` — recovery ladder and retry timing.
- `16-batch-arena-and-compact-lineage.md` — batch identity and compact lineage plan.
- `17-mineru-graph-integration-findings.md` — real-image graph/recovery/LPT coverage ledger.
- `18-multigrain-primitive-core-convergence.md` — primitive core ownership and convergence.
- `19-expand-mixed-output-relation-design.md` — deferred mixed-output API and future relation-space gates.
