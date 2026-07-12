# 恢复层级与重试调度（设计规范）

状态：**重试粒度已定稿；已实现 Map 记录重试、立即分片重试、有预算的自适应定位、actor 替换和有界 stage-epoch 延迟 drain。其他算子的记录单元仍在逐步实现。**
将 [`08-lineage-guided-partial-replay.md`](08-lineage-guided-partial-replay.md) 中“自适应隔离而非无条件二分”的要点落地。此处引用的重排不变性保证由
[`13-reordering-invariance-theorem.md`](13-reordering-invariance-theorem.md) 证明。

本文记录**完整的选项空间**（已选及未选）、**已锁定决策**、**抽象**（使所有选项都有稳定的接口归宿）和**增量实现计划**。这里没有飞线：每个层级都会汇入已存在的同一条 quarantine →（可选 drain）→ `missing_child` 级联。

---

## 1. 已锁定决策

| # | 决策 | 选择 | 理由 |
|---|---|---|---|
| D1 | 将 `retry_timing` 设为一等旋钮？ | **是** | 重试的*何时/何处*是真实维度，独立于*重试多少次*。 |
| D2 | 将 `placement` / `batching` 设为一等？ | **否——由 `deferred` 隐含** | 避免旋钮爆炸（YAGNI）。延迟天然会重新放置并批处理。 |
| D3 | `deferred` drain 的默认粒度 | **`stage_global`（逻辑目标）** | 只有 stage-global 能获得“重新放置以规避节点局部瞬态故障 + 对恢复批次 LPT-repack”。当前 executor 一次仅拥有一个已提交 chunk，因此在执行模型统一前，初始可实现范围是**chunk-global**（§6）。 |
| D4 | 分期 | **接口优先，增量实现/测试** | 现在定义完整 `RecoveryPolicy`，以使 IR 稳定；分阶段交付行为。若任何抽象变难/产生飞线，编码前停止并讨论。 |
| D5 | `retry_timing` 是否延迟 opaque shard retries？ | **否** | 其他分片已独立推进。立即在健康副本上重试 opaque 失败调用；延迟只会增加尾延迟。 |
| D6 | 分片耗尽后的定位 | **有预算的自适应切分** | 稀疏 poison 可精确定位；稠密/系统性失败有硬性工作/调用上限。整个分片 quarantine 不是默认。 |
| D7 | 定位预算耗尽 | **fail-closed 最小未解决子集** | 绝不产生不完整结果；保留所有已成功的兄弟子集。 |
| D8 | `stage_global` 的含义 | **有界 stage epoch** | 节点拥有的缓冲区跨越活跃 microbatches，而非整个数据集。分片大小阈值、stage 静止或输入关闭触发 drain。 |

---

## 2. 层级阶梯（唯一的由轻到重阶梯）

```
① record transient       → retry/defer logical record   max_record_retries + retry_timing
② record deterministic    → isolate logical record       BadRecordError(retryable=False)
③ opaque shard fault      → immediate healthy-replica retry  max_shard_retries
④ shard exhausted         → budgeted adaptive split      on_shard_exhausted="degrade"
⑤ budget exhausted        → quarantine unresolved leaves IsolationBudget
⑥ shard exhausted         → abort the job                on_shard_exhausted="abort"
⑦ missing consumer        → suppress / fail-open         missing_child (orthogonal)
```

`retry_timing` 适用于可归因记录恢复（①）以及被准入恢复池的定位工作（④），而非普通 opaque shard retry（③）。inline 记录重试现在运行；deferred 重试现在 quarantine，保持健康工作流动，然后将集合重新作为恢复批次运行。

逻辑记录单元遵循算子契约：

| 算子 | 可归因重试/隔离单元 |
|---|---|
| Map / Filter | 一个已对齐输入行 |
| Expand | 一个父行及其产生的全部子行 |
| Reduce | 一个 anchor 及其已分组的 descendants |
| Relate | 一个输出 relation/evidence 组 |

一次逻辑调用的所有输出 ports 原子恢复。

`recovery`（*产生*记录有多困难）与 `missing_child`（记录缺失时*消费者*如何处理）**正交**，且必须保持如此。

---

## 3. 完整选项空间（每个维度，已选或未选）

重试有四个子维度。我们仅将两个设为一等；其余在此记录，以使设计完整，且接口能够在不引起 churn 的情况下增长。

| 维度 | 选项 | 状态 | 说明 |
|---|---|---|---|
| **count** | `0`（立即隔离）… `k` | 一等（`max_record_retries`、`max_shard_retries`） | 0 record-retries = 今日行为。 |
| **timing** | `inline`、`deferred` | **一等（`retry_timing`）** [D1/D5] | 适用于可归因记录/定位，而非普通分片重试。 |
| **placement** | `same_replica`、`any_replica` | scheduler-owned [D2/D5] | timing 与 placement 分离。死亡 actor 总需替换；无用户旋钮。 |
| **batching** | `one_by_one`、`as_batch` | **由 `deferred` 隐含** [D2] | deferred drain 将 quarantine 集合作为一个（可 LPT-repack 的）批次重新运行。 |
| **drain scope** | `shard_local`、`stage_global` | 一等（`drain_scope`），默认 **`stage_global`** [D3/D8] | stage-global 是有界节点拥有 epoch，绝非无界的整个数据集 barrier。 |
| **shard-exhausted action** | `abort`、`degrade` | 一等（`on_shard_exhausted`），默认 `abort` | `degrade` = 层级 ④。 |
| **degrade localization** | 粗粒度 quarantine、singleton scan、自适应切分 | 初始固定为自适应切分 [D6] | 不是另一个公共策略旋钮；替代方案仍是实验基线。 |
| **localization budget** | 工作因子 + 调用次数 + 耗尽动作 | 嵌套一等 `IsolationBudget` | 默认值：3× 定位行工作量、64 次调用，然后 quarantine 未解决 leaves。 |

---

## 4. 抽象：一个策略、两个位置、一个 sink

### 4.1 `RecoveryPolicy`（声明式，镜像 `PhysicalHints`）

位于 `IRNode.recovery`，如同已有的 `physical`/`properties`（dataclass field，经由 `add_node` 传递；executor 持有默认值，node 可覆盖——与 `default_replicas` vs `physical.replicas` 相同模式）。可序列化、被动、无行为。

```python
@dataclass(frozen=True)
class IsolationBudget:
    max_work_factor: float = 3.0
    max_calls: int = 64
    on_exhausted: str = "quarantine"

@dataclass(frozen=True)
class RecoveryPolicy:
    # -- record level (enforced in the op wrapper) --
    max_record_retries: int = 0            # 0 = isolate immediately (current)
    retry_timing: str = "inline"           # "inline" | "deferred"
    # -- shard level (enforced in the executor scheduler) --
    max_shard_retries: int = 2
    on_shard_exhausted: str = "abort"      # "abort" | "degrade"
    isolation: IsolationBudget = IsolationBudget()
    # -- deferred drain (only when retry_timing == "deferred") --
    drain_scope: str = "stage_global"      # "shard_local" | "stage_global"
    # placement & batching are IMPLIED by "deferred" (see D2), not fields.
```

所有字段从第一天起即存在（稳定 IR）。行为分阶段交付（§7）；当 executor 遇到未实现的组合时，会抛出带有清晰消息的 `NotImplementedError`，而非悄然误行为。

### 4.2 故障分类位于异常上（而非散落的 ifs）

- `BadRecordError(index=i, retryable: bool = False)` ——记录可归因。
  `retryable=False`（默认）= 确定性 poison ⇒ 重试无意义（value-purity），隔离。`retryable=True` = 瞬态 ⇒ 可进入 ①。
- 任何其他异常 ⇒ 不可归因 ⇒ 分片级（③/④/⑤）。

决策逻辑集中，但应保持**作用域局部**，而非成为一个 god function：

- `policy.decide_record(fault, attempt)` → `{RETRY_RECORD, ISOLATE_RECORD}`；
- `policy.decide_shard(fault, attempt)` → `{RETRY_SHARD, DEGRADE_SHARD, ABORT}`。

两个函数共享一个被动 policy object，但仅暴露各自机制位置实际可以执行的动作。两处都不硬编码重试次数或 fallback 分支。

### 4.3 两个机制位置（物理上不可避免，决策集中）

- **Op-wrapper 位置**（符合条件 wrappers 共享的 isolation helper）：记录级。只有 wrapper 了解算子的逻辑记录单元，且能在该子集上重新调用 UDF。处理 ①/② 及 inline retry。
- **Scheduler 位置**（`ray_executor._run_shards` / stage loop）：分片级。只有 driver 能重提交/切分 shard 并选择健康 actor。处理 ③–⑥ 和 deferred **drain**。

### 4.4 共享 sink 是数据，而不是进程

运行时“sink”是被动 `PortBatch.errors: list[ErrorTrace]` 通道：error traces *内部*承载在与 values 相同的 `PortBatch` 中，经 `take`/`with_values` 复制，跨 shards 由 `concat` 合并，并由 `Reduce` 读取。**没有 collector actor、没有 queue、没有中央瓶颈**——这保留 no-center / no-fly-line 属性，并免费搭乘 Ray 的 object store。

**deferred drain pool** 不仅需要描述（它必须保留失败的*输入*以便重新运行）。它是 generic executor/coordinator 协调的节点拥有 driver metadata，经由现有 persistent actor pool drain。在 Ray 中，被保留的 payloads 仍是 ObjectRefs，故 driver 不复制大 values。
**绝不引入专用 sink/collector actor**——它会成为瓶颈和飞线。

### 4.5 统一数据路径（每个层级均在此结束）

```
run stage → healthy results ─────────────────────────────────────┐
          ↘ faults → quarantine (record row / degraded shard's rows)│
                        │  drain policy (inline? deferred? scope? k×)│
                        ▼                                           │
                   recovery drain  (re-run, may re-place; batched)   │
                        │ recovered → merge back by identity/lineage ┤
                        │ still-failed → ErrorTrace ─────────────────┘
                                                                    ▼
                                          missing_child cascade (final fate)
```

新层级是通往该路径的新*入口*，而不是新路径。degrade 先切分并挽救成功子集；只有未解决 leaves 进入 quarantine。deferred drain 在别处重试该集合；cascade 抑制所需 leaves 仍不可用的 documents。

---

## 5. 语义与保证

- **重排不变性对 deferred retry 成立。** 在另一副本上稍后重新运行一行，并按 record identity 重新附加其结果，在 UDF value-purity 下只是另一次合法物理重排 ⇒ 输出与血缘等价于串行基线（文档 13）。Deferred retry **不需要新证明**；它是现有定理的实例。
- **Degrade 保留血缘驱动的下游。** degraded shard 为每个未解决行构造一个带有 `ancestors` 的 `ErrorTrace`。成功 sibling subsets 保留正常输出，因此 cascade 仅影响实际由未解决 leaves 表示的 documents。
- **定位成本有界。** 普通 shard retries 不计入 isolation budget。每次 split invocation 前，收取一次调用及其输入行数。超过 `max_calls` 或 `max_work_factor × original_shard_rows` 前停止。
- **`abort` 语义不变。** `on_shard_exhausted="abort"` 仍会 raise；在 chunked driver 中，先前 chunks 的已写输出会持续存在（不存在全局 rollback——尚无 job-level checkpoint；见 §8）。

---

## 6. 成本 / 复杂度预算（如实）

- inline retry（①）是 wrapper-local。Adaptive degrade（④）在 shard scheduler 中增加有界 split worklist，但没有新进程或 side channel。它移除硬编码 retry decisions，并使 dense-failure cost 显式化。
- deferred drain（stage-global）是成本所在。它需要 (a) driver 上有界的 node-owned quarantine buffer，以及 (b) 一个 **drain dependency**：`Reduce` 的 `fail_closed` 决策必须等 drain 完成，否则它可能抑制即将恢复的 document。该 barrier 使受影响下游工作只在 drain 后 ready。epoch 在达到正常 shard target、健康工作静止或输入关闭时 drain；绝不等待无界 stream。

### 6.1 执行模型前提（初始收敛已落地）

先前 runtime 有三条执行路径：逐节点 `execute`、无状态全图 `execute_microbatches`（丢失 model/pool reuse）以及手动调用 `_pool_for`/`run_shard` 的 MinerU benchmark。这是 deferred recovery 的架构性阻碍。

recovery-tiers branch 现在有一条初始公共路径：

- `ExecutionCoordinator` 准入有界 microbatches，在其 input refs ready 时调度任意 IR nodes，支持独立分支/fan-in，并产出 ordered 或 completion-order results；
- `execute`、`execute_microbatches` 和 `execute_stream` 均委托给它；
- 节点执行仍复用 `_run_node`，所以 persistent actor pools、sharding、retry accounting、lineage 和 metrics 共用一个实现；
- completed-but-unconsumed outputs 计入 `max_inflight`（有界 backpressure）；
- `Flash-mineru/mg_bridge/run_bench.py` 现在只调用公共 `execute_stream`；不再 import private pools、actor methods、concat，或编写自己的 render/OCR/assemble scheduler。

目前验证：214 个默认 tests + 16 个 Ray parallelism/recovery tests；sparse opaque poison 挽救健康 siblings，dense failure 恰在 budget 处停止，dead actors 每次替换一个 replica，且三个 epoch triggers 均有覆盖。一次真实 4-PDF/48-page/4-GPU MinerU smoke 在 17.34 s 内产生 4/4 markdown（相对基线最小 token Jaccard 0.9953）。一次真实 poison-page run 执行 47 个健康 page OCR，quarantine 一个页面，恰好抑制其 document，并产生 3/3 逻辑一致的健康 markdown（最小 Jaccard 0.9916）。368-PDF performance regression 仍单独进行。

在**真正的 stage-global deferred** 之前尚余：coordinator 必须跨 microbatches 拥有 node recovery epoch，而非让每个 `_run_node` 独立完成 recovery。该 epoch 受 shard-size target 和 coordinator admission window 约束。还应测量 cheap CPU nodes 的 per-node Ray-task overhead，并在仅靠 `max_inflight` 不足时添加 per-stage queue limits。

### 6.2 重试策略外的其他扩展瓶颈

1. **每记录血缘分配。** `PortBatch` 当前每行承载若干 Python objects（`record_ids`、`ancestors`、`ancestor_display`、`ordinals`、`lineage`、`relations`），并在每阶段由 `take`/`with_values` 复制其中许多。数千页面尚可接受，但数百万记录时可能成为 GC/memory wall。所需优化：共享 1:1 lineage paths，使用紧凑/列式 ancestor 和 ordinal 存储。coordinator-owned UUID、`BatchArena`、hash-consed path table、relation encodings 与迁移顺序在 [`16-batch-arena-and-compact-lineage.md`](16-batch-arena-and-compact-lineage.md) 中指定。
2. **Driver-side merge/control。** `concat`、stage coordination 和部分 Reduce 工作在 driver-side。统一 scheduler 必须保持有界 buffering，且避免 driver 成为 payload-copy bottleneck。
3. **In-memory `Relate` join。** 当前 key join 在一个进程中构建所有 role indexes 并 materialize 每 key Cartesian products。它正确表示 M:N，但不是 distributed large-join 实现；partitioned join/spill 是后续工作。

---

## 7. 接口优先、增量计划（遵循 D4）

- **接口（完成）：** 带全部字段的 `RecoveryPolicy` 位于 `IRNode`（镜像 `physical`）；每个 user primitive 接受 `recovery=`；它经被动 IR 序列化，并在 executor wrapper reconstruction 后存续。`BadRecordError(..., retryable=)` 及 scope-local `decide_record` / `decide_shard` 已落地。非默认行为会抛出清晰的 `NotImplementedError`；默认值复现今日语义。
- **Phase 1（为 Map/shardable nodes 完成）：** inline Map record retry + budgeted adaptive shard localization、actor rotation/replacement、sparse 和 dense poison accounting。Expand/Filter/Reduce/Relate 的 attributable units 在各自 wrapper-specific contracts 落地前被明确拒绝。
- **Phase 1.5（执行统一；初始实现完成）：** 公共 `execute_stream` + generic DAG coordinator + persistent-pool reuse 已落地；`run_bench.py` 不再使用 private scheduling APIs。剩余 gate：重新运行 368 PDFs，保持实测约 585 s performance/backpressure。
- **Phase 2（Map 完成）：** node-owned recovery buffer + readiness dependency + re-placement/repacked recovery batch。target rows、stage quiescence 和 input close 均已测试。deferred inputs 合并时使用 invocation-unique temporary tokens，随后在 downstream execution 前恢复原 identity；不存在 collector actor 或 workload-specific path。
- **Phase 3（后续 / 可选）：** 精确 degrade localization（§8）、job-level checkpoint/resume、durable quarantine backend（文档 05）。

指导规则（D4）：若任一 phase 无法保持干净抽象（会散落 policy logic 或加入 side-channel），就暂停并重新审视设计，而非绕过它接线。

---

## 8. 开放风险 / 待重新审视

1. **Dense opaque failures。** Adaptive split 对 sparse poison 高效：一个 bad row 的定位 row-work 少于 `2n`。若 failures 占据 shard 的大部分，无限制切分趋近 `O(n log n)` row-work 和近 `2n` calls。`IsolationBudget` 限制它；未解决 leaves 被保守 quarantine。未来 strict-salvage mode 可有意以无界成本扫描到 singleton。
2. **无 job-level checkpoint。** `abort` 会重启 run；chunked drivers 保留已写 outputs，但不存在 resume-from-progress。Phase 3。
3. **Epoch drain vs. throughput。** 较大 epochs 改善 recovery batching，但延迟受影响 records。初始 trigger size 源自正常 shard size，而非新公共旋钮；在暴露调优前测量 latency/goodput。
4. **Single-index error reporting。** `BadRecordError(index=i)` 每次调用只能标识一个 bad row，因此许多 bad rows 会导致重复 peel-and-rerun。为兼容保留 `index`，但为 `BadRecordError(indices=[...])` 留出空间，以使能标识多个 failures 的 UDF 一次隔离它们。
5. **Value-purity 是显式范围边界。** 重排和 deferred retry 不覆盖全局有状态/sessionized operators、global sorting、cross-record deduplication、iterative/cyclic dataflow 或不安全 external side effects。除非未来 materialization/commit contract 为其提供显式语义，否则它们是非目标。

---

## 9. 测试计划（dummy-first）

- 复用 `test/experimental/multigrain/lineage_ops.py` poison hooks + 真实 `mg_bridge/mineru_poison_ops.py`。
- **本地测试**：断言 ① 一个 transient（`retryable=True`）行在 `k` 内恢复且不会被 quarantine；② deterministic 保持隔离。本地执行没有 shard，无法验证 ④/deferred scheduling。
- **Ray 测试（scheduler tiers 必需）：** 断言 ④ degraded shard 产生 per-row quarantine traces，且 `missing_child` cascade 恰好抑制受影响 documents（其他完整）。
- Phase 2 Ray tests：断言 deferred stage-/chunk-global drain 保持健康 throughput（无 head-of-line stall）、重新放置到不同 replica，并产生与 inline path 相同的 output + lineage（reordering-invariance property test）。
