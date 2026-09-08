# 04：逐段读懂 `runtime/dispatch.py`——Grain 的唯一物理状态机

主源码：[`runtime/dispatch.py`](../../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py)。

Engine 决定某个 Call × Entity 是否应该运行；`DispatchState` 则独占回答：它现在 READY、
IN_FLIGHT 还是 SEALED，在哪个 queue，哪个 generation 的报告仍有效。

---

## 1. 为什么 DispatchState 要从 Engine 分出来

Item/Expansion/Entity 是数据流语义；Grain phase、batch reservation、retry queue 和 stale report
fencing 是物理调度状态。如果全部混在 Engine：

- primitive outcome 与 retry queue 会交叉修改；
- Executor 可能直接改 GrainRecord；
- 同一 attempt 计数容易在 policy、Engine、Executor 多处重复。

V3.6 的拆分是：

```mermaid
flowchart LR
    Transition["grain_transition<br/>纯 phase 代数"]
    Policy["RecoveryPolicy<br/>纯动作决策"]
    Dispatch["DispatchState<br/>执行 phase/queue/generation 修改"]
    Engine["MicrobatchEngine<br/>决定语义传播"]
    Executor["Executor<br/>actor/RPC"]

    Transition --> Dispatch
    Policy --> Engine --> Dispatch
    Executor --> Engine
```

纯函数决定“允许什么”，DispatchState 是唯一执行物理状态变化的 owner。

---

## 2. 源码地图

| 源码段 | 作用 | 核心不变量 |
| --- | --- | --- |
| [`GrainRecord/Snapshot`](../../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py#L17-L32) | 私有可变记录与公开只读副本 | 可变 Record 不泄露 |
| [`DispatchBatch`](../../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py#L35-L48) | 一次精确 dispatch/recovery batch | 非空、同 Call |
| [`DispatchState.__init__`](../../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py#L58-L66) | 建 record table 与三个 queue | queue 只有一个 owner |
| [创建 READY/SEALED Grain](../../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py#L117-L135) | WAITING 离开 pending 后建档 | 一个 Grain 只创建一次 |
| [`priority/reserve_with_barriers`](../../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py#L137-L219) | 按优先级选择，并分开 live/barriered Grain | immediate-retry → ready → deferred-recovery |
| [`validate_in_flight/seal`](../../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py#L221-L237) | generation-fenced report commit | 旧 attempt 不可提交 |
| [UDF/infra recovery](../../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py#L239-L295) | 执行已批准的恢复动作 | policy 不在本文件 |
| [`_release/_transition`](../../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py#L297-L332) | DispatchBatch 预检后统一改 phase/generation | 不出现部分 batch 迁移 |

---

## 3. 三个核心 DTO

### 3.1 `GrainRecord`

```python
@dataclass(slots=True)
class GrainRecord:
    phase: GrainPhase
    generation: int = 0
    infra_failures: int = 0
    parent_anchor: EntityRef | None = None
```

这里只保存不能从 queue 安全推导的权威物理状态：

- `phase`：当前生命周期；
- `generation`：attempt fencing token；
- `infra_failures`：该 Grain 已完成的基础设施重试数。
- `parent_anchor`：首次 READY 时冻结的直接父级；packing、runnable queues 和 suppression barriers 共用。

它不保存 active actor、Item outcome、UDF retry mode 或 output。那些分别属于 Executor、Engine、
RecoveryPolicy 与 RuntimeState。

`GrainSnapshot` 复制这三个标量，供诊断和错误信息读取。调用者不能拿 snapshot 反向修改调度。

### 3.2 `DispatchBatch`

一个 batch 保存精确 Grain tuple 与 `udf_retries`：

```text
DispatchBatch(grains=(g1, g2, g3), udf_retries=1)
```

它必须非空且所有 Grain 属于同一 Call，因为一个 RPC 只能进入一个 actor pool/UDF。

`udf_retries` 属于这次恢复 DispatchBatch，而 `infra_failures` 属于每个 Grain：二者口径不同，不能合成
一个含糊的 attempts。

### 3.3 为什么删除 `_ReadyEntry`

旧 ready entry 另存一份预计算 `parent_anchor`，但 recovery queues 只有 GrainRef。现在 anchor 收口到
`GrainRecord`，ready queue 直接存 GrainRef；这样 ready、immediate-retry、deferred-recovery、
late commit 与失败恢复都读取同一份 scope 快照，不新增第二张 Grain→parent 索引。

---

## 4. 三个 queue 分别是什么

```python
self._ready_fifo_by_call = defaultdict(deque)  # CallRef -> deque[GrainRef]
self._immediate_retry_queue = deque()
self._deferred_recovery_queue = deque()
```

| queue | 放什么 | 使用场景 | 优先级 |
| --- | --- | --- | ---: |
| `_immediate_retry_queue` | 完整 `DispatchBatch` | `retry_batch`、基础设施重试 | 0 |
| `_ready_fifo_by_call` | 每个 Call 一条 `deque[GrainRef]` | 首次可执行 Grain，按 batch_size 聚批 | 1 |
| `_deferred_recovery_queue` | 完整 `DispatchBatch` | `retry_tail`、isolate/split 后的子 batch | 2 |

immediate-retry 在 READY work 之前处理失败 DispatchBatch；deferred-recovery 先让新鲜 READY work
推进，再回头处理可疑 batch，以降低 head-of-line blocking。

queue 只表示 runnable selection，不替代 `GrainRecord.phase`。record 是权威状态；reserve 时仍检查
每个 entry 是否 READY。

---

## 5. Grain 从 WAITING 离开时怎样建档

WAITING 不保存在 DispatchState。Call inputs 尚未决定时，Engine 使用
`RuntimeState.pending_grains`。

一旦输入代数作出决定：

```text
inputs_ready(grain, parent_anchor)
    _create(None + INPUTS_READY -> READY)
    freeze parent_anchor in GrainRecord
    append ready queue

inputs_terminal(grain)
    _create(None + INPUTS_TERMINAL -> SEALED)
    no queue
```

`_create()` 先拒绝重复 Grain，再通过唯一 `grain_transition()` 建立 phase。直接 SEALED 表示上游
已经决定无需 Worker，例如 required input DROPPED 或 upstream failure suppression。

---

## 6. `priority()` 与 `reserve_with_barriers()`：怎样选择下一批

`priority(call)` 只回答该 Call 当前最高可用层级，供 Executor 在多个 active microbatch 中比较。

Engine 把该 Call 当前 barriered 哈希视图传给
`reserve_with_barriers(call, max_size, pack_by_parent, barriered_anchors)`，它严格按：

```text
immediate-retry work
→ READY work
→ deferred-recovery work
```

live Grain 由 READY 原子转为 IN_FLIGHT；barriered Grain 走唯一新边
`READY + SUPPRESS -> SEALED` 并返回给 Engine 发布 `SUPPRESSED`。若本轮只有 barriered tombstone，
返回 cleanup-only `(None, suppressed)`，Executor 不发空 RPC。

### READY 聚批

`_reserve_ready()` 遍历当前 ready queue：

1. 只访问目标 Call 自己的 deque，并丢弃已非 READY 的 stale entry；
2. 对 `parent_anchor` 做期望 O(1) barrier membership 查询，命中即封闭且不回队；
3. 达到 `max_size` 后保留剩余项；
4. `pack_by_parent=True` 时只选择与第一个 live Grain 同 parent_anchor 的项；
5. 所有选中项统一走 `_reserve_exact()`；
6. 用 `remaining` 重建 queue，保持未选项相对顺序。

`pack_by_parent` 只影响一次物理 RPC 的 packing，不改变 Domain、Entity 或 reduce 语义。
any_parent 路径每个 entry 只弹出一次，隔离查询不会重新引入跨 Call 扫描；single_parent 收集同父项仍
保留原有的目标 Call 内扫描成本。

### 精确 recovery batch

`_reserve_recovery()` 不重新按 `batch_size` 聚合，也不混合独立 DispatchBatch。二分隔离依赖失败的
确切 DispatchBatch，因此 recovery 必须保持边界。若 barrier 只命中其中一部分，它产生保持原顺序和
`udf_retries` 的 exact live sub-batch；全 barriered 时不构造空 `DispatchBatch`。

---

## 7. generation fencing：为什么 GrainRef 不随重试改变

重试保留同一个 `GrainRef(call, entity)`，只递增 generation：

```text
generation 0: READY -> IN_FLIGHT -> RETRY
generation 1: READY -> IN_FLIGHT -> REPORT
```

若 generation 0 的慢报告之后才到，`validate_in_flight(grain, 0)` 会看到 record generation 已是 1，
抛出 `stale generation`。因此无需为每次 attempt 制造新 Grain identity，也不会让旧报告覆盖新结果。

```mermaid
sequenceDiagram
    participant D as DispatchState
    participant A0 as old attempt g@0
    participant A1 as new attempt g@1

    D->>A0: reserve generation 0
    D->>D: retry; generation = 1
    D->>A1: reserve generation 1
    A0-->>D: late report generation 0
    D-->>A0: reject stale generation
    A1-->>D: report generation 1
    D->>D: seal Grain
```

`seal()` 只接受 IN_FLIGHT 且 generation 匹配的 Grain，然后执行
`IN_FLIGHT + REPORT -> SEALED`。

---

## 8. 恢复策略与物理动作怎样分工

`RecoveryPolicy.decide_udf()` 是纯函数，只根据 completed retries 与 DispatchBatch size 返回
`RecoveryAction`。DispatchState 不重新解释配置，只执行动作。

### `RETRY_IMMEDIATE`

整个 DispatchBatch 执行：

```text
IN_FLIGHT -> READY
generation += 1
udf_retries += 1
append _immediate_retry_queue
```

### `RETRY_TAIL`

状态变化相同，但 append `_deferred_recovery_queue`，让正常工作优先。

### `SPLIT_TAIL`

只接受已经至少失败重试一次且包含多个 Grain 的 DispatchBatch。整批先原子 release，然后按
midpoint 拆成两个 deferred-recovery batches；两个子 batch 继承 completed `udf_retries`。

它不是在 DispatchState 内判断“该不该继续二分”；policy 会在下一次失败时根据 batch size 返回
`SPLIT_TAIL` 或 `FAIL_SINGLETON`。

### infrastructure recovery

基础设施失败不是数据/UDF 失败，因此：

- DispatchBatch 原样进入 `_immediate_retry_queue`；
- generation 递增；
- 每个 Grain 的 `infra_failures` 递增；
- `udf_retries` 不变。

Actor 替换由 Executor 完成，DispatchState 从不持有 handle。

---

## 9. `_transition()`：DispatchBatch 原子性的关键

对一个 exact DispatchBatch，源码分两步：

1. **preflight**：确认非空、每个 Grain 存在且全部处于 expected phase；
2. **mutation**：逐 record 应用相同 `grain_transition()`。

只有所有成员都合法才开始修改，因此不会出现前两个 READY→IN_FLIGHT、第三个却非法而留下部分 batch
reservation。

`_release()` 先用 `_transition()` 把整个 DispatchBatch 从 IN_FLIGHT→READY，再统一增加 generation/counter。这里
没有 Ray 调用或外部 user code，mutation 段是封闭的。

---

## 10. 跟踪一次 isolate-tail

假设 batch `[g0, g1, g2, g3]` 中某行让 opaque UDF 整批抛错：

```mermaid
flowchart TD
    B0["READY batch<br/>g0 g1 g2 g3"]
    Retry["first failure<br/>RETRY_TAIL"]
    B1["deferred-recovery batch<br/>g0 g1 g2 g3<br/>udf_retries=1"]
    Split["fails again<br/>SPLIT_TAIL"]
    L["deferred: g0 g1"]
    R["deferred: g2 g3"]
    L2["continue split or singleton"]
    R2["successful sub-batch seals"]

    B0 --> Retry --> B1 --> Split
    Split --> L --> L2
    Split --> R --> R2
```

每次 retry 都保留 GrainRef、增加 generation；成功子 batch 正常 report，最终失败 singleton 才由
Engine 发布该 Grain outputs 的 `FAILED`。DispatchState 本身不知道 Item outcome。

---

## 11. 为什么没有 `active_attempt` 或 actor 字段

IN_FLIGHT phase + generation 已足够验证报告。当前占用哪个 actor 由 Executor 的 `_PendingRpc`
保存；把 actor handle 再抄入 GrainRecord 会制造两份物理所有权，并让 Ray-free DispatchState
依赖执行后端。

同理，ready queue entry 不需要保存完整 `GrainInvocation`：payload binding 可能随 runtime state
生命周期管理，真正 dispatch 时由 Engine 即时投影。

---

## 12. 修改 DispatchState 前的检查清单

- 新字段是否真是无法从现有 phase/generation/queue 推导的权威物理状态？
- 新动作是否应先由纯 RecoveryPolicy/transition 决定？
- exact DispatchBatch 是否总是先完整 preflight 再 mutation？
- retry 是否保留 GrainRef 并递增 generation？
- UDF retries 与 infrastructure retries 是否继续分开计数？
- recovery DispatchBatch 是否保持边界，避免重新与 READY Grain 混批？
- `parent_anchor` 是否只在 READY 时冻结，并由所有可运行队列和 suppression barrier 共享？
- barriered tombstone 是否只访问一次，且不产生空 batch？
- actor handle、Item outcome、payload 是否仍未进入本组件？
- 对外是否只暴露 immutable snapshot，而非 GrainRecord？

如果新功能需要 DispatchState 判断 Filter mask 或调用 `ray.kill()`，它应分别回到 Engine 或
Executor。

下一篇：[05：Worker ABI](05_worker_abi.md)。
