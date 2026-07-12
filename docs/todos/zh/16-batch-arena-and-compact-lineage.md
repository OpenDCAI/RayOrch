# 批次作用域身份与紧凑血缘（设计规范）

状态：**设计方向已锁定；尚未实现。** 在统一 executor 稳定前，当前 `PortBatch` 仍是兼容数据模型。本规范是实现真正 stage-global deferred recovery 及百万记录血缘主张之前所需的基础。

相关文档：

- 恢复调度：[`15-recovery-tiers-and-retry-scheduling.md`](15-recovery-tiers-and-retry-scheduling.md)；
- 部分重放：[`08-lineage-guided-partial-replay.md`](08-lineage-guided-partial-replay.md)；
- 关系代数 / 重排安全：
  [`13-reordering-invariance-theorem.md`](13-reordering-invariance-theorem.md)。

---

## 1. 已锁定设计决策

1. **运行时身份以批次为作用域，而非永久全局。**
   coordinator 为每个准入的逻辑 microbatch 创建一个 UUID。
2. **一个 microbatch 的所有输入 ports 共享同一 `BatchId`。**
   `mg.source()` 不得独立生成 batch UUIDs。
3. **记录由紧凑 handle 寻址**：
   `RecordHandle = (BatchId, PortId, RowIndex)`。
4. **批次完成是内存回收边界。**
   一旦所有下游消费者、重试、输出提交及所请求的 trace materializations 均完成，整个 batch arena 即可释放。
5. **逻辑 lineage paths 将被 hash-consed/interned。**
   记录保存小型 `path_id`，而不是每阶段复制 Python tuple/dict。
6. **外部业务身份与之分离。**
   可选存储 source key/path 以供 checkpointing 或 cross-run queries，但它不是默认内部 runtime identity。

---

## 2. 身份模型

```text
BatchId      = UUID128
PortId       = compact integer assigned by compiled IR
RowIndex     = invocation-local integer
RecordHandle = (BatchId, PortId, RowIndex)
```

UUID 由 `ExecutionCoordinator` 在准入 microbatch 时生成，并经物理 sharding、LPT reordering、retries、deferred drain、fan-out 和 fan-in 保留。一次合法物理重排绝不创建新的 `BatchId`。

为何不用 `name:index`？

- indices 在每个物理 chunk 重启，并在活跃 batches 间冲突；
- stage-global quarantine/recovery 可能将失败行归因到错误 document；
- 全局唯一字符串 IDs 比 UUID 加紧凑整数分配和复制更多内存。

为何不要求稳定 source key？

- 许多 runtime objects 没有持久业务 key；
- cross-run 稳定性对 ephemeral scheduling 不必要；
- 它混淆 runtime identity 与 checkpoint/catalog identity。

对于持久化，在 source row 旁或 lineage backend 中存储可选 `ExternalKey`。恢复时，checkpoint 恢复原始 `BatchId`，或记录显式 old→new execution mapping。

---

## 3. `BatchArena`：所有权和生命周期

```text
coordinator admits input
  → BatchArena(UUID)
      values / Ray ObjectRefs
      compact record columns
      path table
      relation tables
      errors/quarantine handles
  → nodes consume/produce slices by RecordHandle
  → retry/deferred drain holds arena references
  → graph outputs committed/materialized
  → consumer + recovery reference count reaches zero
  → optional compact lineage snapshot
  → release entire arena
```

arena 是**逻辑所有权单元**，不是复制 payload 的 central actor：

- 大 values 保留在 Ray 的 object store；
- driver coordinator 保留 handles 和 compact metadata；
- actors 接收一次 invocation 所需的 slices/handles；
- 不引入全局 lineage collector actor。

以下情况下不得释放 arena：

- node invocation 或 downstream consumer 引用它；
- deferred quarantine item 可能重试；
- `Reduce`/`Relate` barrier 仍需要其 relation table；
- output commit 尚未完成；
- 用户请求 post-run trace materialization。

面向用户的 `ErrorTrace` 必须在 arena 释放前 materialize 它所需的小 display/error fields；它不得包含指向已释放 metadata 的悬垂指针。

---

## 4. 低秩血缘表示

当前 MVP 每行复制：

```text
ancestors: dict
ancestor_display: dict
ordinals: dict
lineage: tuple[str, ...]
relations: tuple[ParentRef, ...]
```

对于穿过同一 `Map A → Map B → Map C` 的一百万行，这会创建许多重复行构成的矩阵。其结构秩很低：operator path 是共享的，而记录主要在行/父坐标上不同。

### 4.1 Hash-consed path table

```text
LineagePathTable:
  path_id -> (parent_path_id, op_id, relation_kind)

intern(parent_path_id, op_id, relation_kind) -> path_id

RecordMeta:
  handle
  path_id
  relation_ref
  ordinal_ref
```

Interning 使用 structural hash + equality verification（hash collision 绝不可合并不相等的 paths）。复杂度由约 `O(records × 1:1 stages)` path objects 变为 `O(unique paths + records)` compact indices。

### 4.2 关系专用紧凑编码

| relation | 紧凑表示 |
|---|---|
| 1:1 Map | 共享 `path_id`；保留 row coordinate |
| Filter | survivor row-index vector / bitmap |
| 1:N Expand | `parent_row[]` + `child_ordinal[]`；每个 child 无 ancestor dict |
| N:1 Reduce | anchor row + descendant ranges/index vector |
| multi-output | 共享 invocation/path table；不同 `PortId` |
| M:N Relate | 每个 role 使用 CSR-style offsets + parent handle arrays |

M:N edge count 不可约（`O(edges)`），但 Python object overhead 不是：CSR/columnar arrays 取代每输出的 `ParentRef` objects tuples。

Display keys 是单独的可选列。它们对 scheduling correctness 非必需，可按 trace policy 惰性/materially 保留。

---

## 5. 正确性与语义

- **Value-purity 仍是定理假设。** UUID/handles 是 framework metadata，绝不进入 UDF inputs。
- **重排不变性：**物理 planners 置换 row vectors，但保留 `RecordHandle` 和 relation coordinates。
- **Deferred recovery：**quarantine 保存对仍存活 arena 的 handle；成功 retry 写回同一逻辑 handle。
- **批次独立性：**invocation-local 的 `Reduce`/`Relate` 不得悄然跨 arenas join。跨批操作需要显式 state/window/materialization contract。
- **持久化是 opt-in：**ephemeral runs 释放 arenas；checkpoint/lineage sinks 序列化 compact tables 和 execution IDs。

---

## 6. 兼容性表面

不要强制用户或 UDFs 采用 handles：

- `PortBatch.values` 保持 UDF-facing value column；
- 现有 `record_ids`、`ancestors`、`ordinals`、`lineage` 和 `relations`
  成为 arena tables 上的 compatibility views/materializations；
- tests 首先比较 old 与 compact representations 的相同 public values、trace views、grouping 和 errors；
- 仅在实现 parity 后，hot execution 才可停止急切构造 legacy Python objects。

这避免对 primitives、executor 和 MinerU 进行 flag-day rewrite。

---

## 7. 增量实现计划

1. **先插桩：**测量当前 lineage bytes/object counts 及 allocation time（E4 baseline）。
2. **Batch identity：**coordinator-owned UUID + `BatchContext`；不改变 UDF/API 行为。
3. **1:1 path interning：**实现 `LineagePathTable`；保留 legacy views。
4. **1:N columnar relation：**迁移 Expand parent/ordinal metadata。
5. **Reduce + error lookup：**grouping/cascade 直接消费 compact tables。
6. **M:N CSR：**迁移 `Relate` parent evidence。
7. **Arena refcount/release：**将生命周期绑定 coordinator completion、deferred drain 和 output commit。
8. **Optional persistence：**compact lineage sink/checkpoint format。

仅在步骤 2–7 后，恢复文档 15 才应将其 drain scope 称为跨多个 active microbatches 的真正 stage-global。

---

## 8. 必需测试与测量

- UUID 在同时活跃的 microbatches 间唯一；一个 microbatch 的所有 ports 共享它。
- sharding/reordering/retry 保留 handles。
- deferred item 或 fan-in consumer 存在时 arena 不被释放。
- completion 后 arena 被释放（weakref/resource-counter test）。
- 相同 1:1 paths intern 至同一 `path_id`。
- hash collision path 验证 structural equality。
- Map/Filter/Expand/Reduce/Relate 的 old vs compact lineage property tests。
- M:N CSR round-trip 及 multi-output coverage。
- E4：从 7k pages 到 million-record synthetic scale 的 bytes/record、Python object count、allocation CPU、serialization bytes 和 end-to-end throughput。

验收目标：lineage memory 随**records + unique paths + relation edges**增长，而不是随 `records × path length` Python objects 增长。
