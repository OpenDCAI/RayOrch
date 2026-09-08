# V3.6 双层数据失败与同父抑制实现记录

状态：**已实现并通过回归；核心实现提交为 `1dcb404`**

记录日期：2026-08-26

源码行号快照：`ff3a1b8`（实现时以类型和函数名重新定位，行号仅用于当前 review）

本文规划 V3.6 的两种显式、终态数据失败：

```python
return mg.RecordFailure(cause)  # 只失败当前 Grain
return mg.GroupFailure(cause)   # 失败当前 Grain，并隔离同 Call、同直接父级的兄弟 Grain
```

目标是在不增加异常索引协议、Manager、额外请求所有权对象或分布式取消的前提下，同时支持：

- 不经过 `reduce` 时，只丢弃精确坏项，健康兄弟仍可独立输出；
- 用户确认整个父级工作单元已失效时，停止尚未执行的同类兄弟，并丢弃晚到结果；
- 混合 parent batch 中，其他 parent 的健康结果一次执行后直接提交，不做 collateral replay；
- 用现有 `FAILED / SUPPRESSED` 区分“自身被确认损坏”和“自身未判坏但因同父 cohort 失效而不可见”。

本文明确删除旧方案中的 `BadItemError`、`DispatchFailureKind.BAD_ITEM`、物理 batch index 映射和
attributed-dispatch replay。UDF 必须正常返回完整 batch，Worker 才能得到逐 Grain reports。

---

## 0. 实施结果（2026-08-26）

实现严格沿用本文的双返回值设计，没有增加 Manager、跨 actor 控制面、异常 index 协议或新
Grain phase：

- 公开 API 只新增 `GroupFailure(cause)`；`RecordFailure` 行为保持精确 Grain 失败；
- Worker 使用 `GroupFailure > RecordFailure > normal` 的稳定 join，并继续返回完整逐 Grain
  reports；
- Engine 使用一个私有、microbatch-local `_SuppressionBarrierIndex`，按
  `(CallRef, direct parent EntityRef)` 建立单调 barrier；
- Executor 将完整 WorkerDispatchResult 一次交给 `commit_reports()`，同批 report 顺序不会泄漏到语义；
- READY、WAITING/admission、late in-flight success、ready/immediate-retry/deferred-recovery 以及 opaque/infra
  recovery 均经过同一 barrier 语义；
- ready queue 仍按 Call 分区，`any_parent` entry 只增加期望 O(1) 哈希 membership 查询；
- cleanup-only reserve 不发送空 RPC，并用 progress turn 避免误报 deadlock。

当前回归证据：

```text
V3.6 unit                         74 passed
V3.6 tracked benchmark           51 passed, 1 skipped
V3.6 Ray Executor integration    17 passed
changed source pyright           0 errors, 0 warnings
compileall / git diff --check    passed
```

全目录 pyright 仍会报告既有 benchmark 问题：tracked `mineru_daft.py` 的 Ray 动态 `.options`，以及
用户未跟踪的 `mineru_ray_data.py` 依赖/类型错误；本次修改的源码文件单独检查为零错误。

---

## 1. 公开语义与 outcome 代数

### 1.1 `RecordFailure`：精确 Grain 失败

```python
mg.RecordFailure("bad page")
```

表示用户已经精确确认当前输入对应的 Grain 是终态坏数据：

- 当前 Grain 的所有输出原子地发布为 `FAILED`；
- 同父健康兄弟正常提交，不建立隔离 barrier；
- 下游只按真实依赖传播；
- required `reduce` 可以因一个失败成员自然得到 `SUPPRESSED`，但其他健康成员仍可被其他分支消费。

这是现有 V3.6 行为，必须保持兼容。

### 1.2 `GroupFailure`：同父 sibling cohort 失败

```python
mg.GroupFailure("document is poisoned")
```

表示用户确认当前 Grain 不仅自身失败，而且当前 Call 下、共享同一个直接 runtime parent 的兄弟
已失去继续执行或提交的业务价值：

- 返回 `GroupFailure` 的明确坏 Grain 发布为 `FAILED`；
- 同 scope 尚未提交的健康兄弟发布为 `SUPPRESSED`；
- barrier 前已经提交的 `PRESENT` 事实不回滚；
- 其他 parent、其他 Call 不被 barrier 直接影响。

公开名称使用 `GroupFailure` 而不是 `ParentFailure`：后者容易被理解为 parent Entity 在所有
Call/分支中全局失败；本合同只隔离当前 Call 的 sibling cohort。

### 1.3 四种 Item outcome 各自只表达一种事实

| outcome | 唯一含义 | 是否持有 ValueBinding |
| --- | --- | --- |
| `PRESENT` | 当前 Grain 成功产生可见值 | 是 |
| `DROPPED` | control/filter 语义主动删去该值 | 否 |
| `FAILED` | 当前 Grain 被精确确认计算/数据失败 | 否 |
| `SUPPRESSED` | 当前 Grain 本身未判坏，但因依赖或 suppression barrier 而不应执行/提交 | 否 |

禁止把 sibling isolation 也发布成 `FAILED`。否则系统无法区分 poison origin 与被节省的工作，
论文统计、错误归因和用户审计都会失真。

### 1.4 两种 sentinel 都是 return value，不是 Exception

`RecordFailure` 和 `GroupFailure` 都表示用户已经做出确定性数据判断，因此：

- 都不触发 `RecoveryPolicy`；
- 都不消耗 UDF retry budget；
- 都要求 UDF 正常返回完整、行对齐的 batch；
- 都不应被 `raise`。

可能重跑后恢复的错误必须继续使用普通 Python exception：

```text
raise Exception
→ DispatchFailure(UDF_ERROR)
→ retry_batch / retry_tail / isolate_tail / abort
→ 若重试正常返回，则正常提交，不存在 barrier
→ 若某次重试正常返回 GroupFailure，才在该次成功 RPC 的提交点安装 barrier
```

基础设施异常仍由 actor replacement 与 `infra_retries` 处理。系统不能从 opaque exception 或
Ray failure 猜测 parent poison。

---

## 2. 隔离 scope 与单调边界

### 2.1 唯一 barrier key

```text
(CallRef, direct runtime parent EntityRef)
```

- “同类”固定表示同一个静态调用点 `CallRef`，不是相同 Python UDF class；
- “同父”固定表示 runtime lineage 中唯一的直接父 Entity，不是任意祖先或 root；
- root Grain 没有父时使用自身作为 `parent_anchor`，所以 parent-suppression scope 自然退化为该 Grain；
- barrier 由单个 `MicrobatchEngine` 独占，不跨 microbatch；
- 一个 Call 的 barrier 不能直接取消同一 parent 上的另一个 Call。

只用 parent 作键会误伤其他算子；只用物理 batch 作键会误伤 any-parent DispatchBatch 中其他 parent。

### 2.2 唯一终态优先级

逐 Grain 提交只遵守下面的顺序：

```text
1. 当前 Grain 显式返回 RecordFailure / GroupFailure  → FAILED
2. 成功 Grain 命中已有或本批新建的 suppression barrier   → SUPPRESSED
3. 其他成功 Grain                                    → PRESENT / 原有 Expansion 语义
```

显式失败证据永远优先于 barrier。因此，同一 parent 的两个 in-flight Grain 分别返回
`GroupFailure` 时，两者都是 `FAILED`；不会把第二个明确坏项降格为 `SUPPRESSED`。

### 2.3 单调线性化规则

```text
barrier 前已经 commit 的事实       → 保持原终态，不回滚
barrier 后仍在 READY                → reserve barrier check 将其 SUPPRESSED
barrier 后才到达 Call input         → admission barrier check 将其 SUPPRESSED
barrier 后才收到成功 report         → commit barrier check 将其 SUPPRESSED
同一 WorkerDispatchResult 中先排列的成功项   → 仍受该结果后部 GroupFailure 影响
```

最后一条要求 Executor 不能继续逐 report 边看边提交；必须先看完整 WorkerDispatchResult 中有哪些
`GroupFailure`，否则 Python tuple 顺序会改变语义。

已经提交的 `PRESENT` 不回滚。回滚意味着删除 child Entity、Expansion、其他 Call 的下游事实和
潜在外部副作用，属于事务/补偿系统，不是本计划的隔离合同。

---

## 3. Worker 与 batch commit 合同

### 3.1 Worker 的完整 batch 返回

Worker 保持现有“先执行整个 UDF，再规范化输出列”的模型。每个物理位置的所有输出仍是一个
multi-output 原子 Grain：

| 同一位置各输出列 | Worker report |
| --- | --- |
| 都是正常值 | `GrainReport` |
| 至少一个 `RecordFailure`，没有 `GroupFailure` | 精确 `GrainFailureReport` |
| 至少一个 `GroupFailure` | suppression-bearing `GrainFailureReport` |

当多输出在同一位置给出多个失败 sentinel 时采用确定性的 join：

```text
GroupFailure > RecordFailure > normal value
```

cause 选择规则也固定：按编译后的 output layout 顺序，取第一个最高优先级 sentinel 的 cause。
这样 output 列迭代顺序可审计，不依赖字典顺序，也不会出现一半输出成功、一半输出失败。

### 3.2 Wire DTO 不暴露 public scope enum

公开 API 只新增：

```python
@dataclass(frozen=True, slots=True)
class GroupFailure:
    cause: Any
```

现有内部 `GrainFailureReport` 增加一个有默认值的行为字段，例如：

```python
suppress_siblings: bool = False
```

- `RecordFailure` 生成 `False`；
- `GroupFailure` 生成 `True`；
- 默认值保持现有内部构造兼容；
- 不公开 `FailureScope`、字符串 scope 或新的 recovery knob。

这是一个封闭的二值 wire 事实，不需要新增 `GroupFailureReport`、异常 DTO 或 dispatch failure kind。

### 3.3 Executor 必须按 WorkerDispatchResult 整批提交

Executor 必须将完整 WorkerDispatchResult 一次交给唯一的 batch commit 入口：

```python
engine.commit_reports(pending_rpc.dispatch_batch, result)
```

Engine 的 batch commit 至少分为四步：

1. **Identity preflight**：reports 与 DispatchBatch grains 一一对应，无重复、无缺失、Call 一致，全部
   generation 匹配且仍为 `IN_FLIGHT`；失败时 canonical state 不变。
2. **Barrier discovery**：收集所有 GroupFailure 的 `(Call, parent_anchor)`，与既有 barrier 合并，
   但不能受 report 顺序影响。
3. **Payload preflight**：只对最终不会被 suppress 的 `GrainReport` 执行现有 output、Expansion、
   cardinality 与 control 校验；被 suppress 的 payload 永远不创建 provisional children。
4. **Mutation turn**：安装 barriers，按显式失败 / suppressed success / live success 应用终态，最后
   只运行一次 `advance()`。

为避免在 `commit_reports()` 中混合校验与发布，应把 mutation frontier 前的逻辑抽成私有
prepare helper，返回不可变 commit intents；apply helper 只消费已经验证的 intents。这是局部
提交边界重构，不引入 Store、Coordinator 或通用事务框架。

### 3.4 同一 RPC 的混合 parent 不重放

示例 WorkerDispatchResult 对应 `[A0, B0, A1, C0]`，其中 A1 返回 `GroupFailure`：

```text
A1 → FAILED
A0 → SUPPRESSED
B0 → 正常 commit
C0 → 正常 commit
```

因为 UDF 已经完整返回逐 Grain reports，B0/C0 不需要 retry。相比 raise-based 方案，这直接删除了
“坏项以外的 collateral grains 重新入队”状态和额外算力。

---

## 4. 一个共享门禁，覆盖三个生命周期入口

### 4.1 私有 `_SuppressionBarrierIndex`

在 `runtime/engine.py` 内定义一个小型私有值对象：

```python
class _SuppressionBarrierIndex:
    _causes_by_call: dict[CallRef, dict[EntityRef, object | None]]

    def establish(self, call, parent_anchor, cause): ...  # first writer wins
    def is_barriered(self, call, parent_anchor): ...      # expected O(1)
    def cause(self, call, parent_anchor): ...
    def anchors_for(self, call): ...                     # membership view
```

它只保存 canonical barrier 与首个 barrier cause：

- 不拥有 Grain phase；
- 不拥有 ready/recovery queues；
- 不解析 lineage；
- 不发布 Item/Expansion；
- 不进入 Worker 或 Ray actor；
- 不是 Manager，也不单独创建 runtime 模块。

多个 `GroupFailure` reports 命中同一 scope 时，barrier cause first-writer-wins；同一个
WorkerDispatchResult 内按 `DispatchBatch.grains` 的稳定顺序选择 origin，不依赖 reports tuple 顺序；不同 RPC 之间按 driver
commit 顺序线性化。每个显式失败 Grain 仍保留自己的 cause。`SUPPRESSED` sibling 使用 canonical
barrier cause，便于解释“因哪个 GroupFailure 被跳过”。

### 4.2 `parent_anchor` 成为 Grain 的唯一物理 scope 快照

实现前 ready queue 的 `_ReadyEntry` 持有 `parent_anchor`，而 immediate-retry/deferred-recovery queues 只持有
`GrainRef`。本次已把 `parent_anchor` 收口进 `GrainRecord`：

```text
GrainRecord = phase + generation + infra_failures + parent_anchor
ready       = CallRef -> deque[GrainRef]
```

Grain 进入 READY 时由 Engine 用 `_parent_anchor(entity)` 冻结 anchor。ready、immediate-retry、
deferred-recovery、late commit 和 failure partition 都从同一个 authoritative record 读取，不新增第二张
`grain -> parent` 表。input terminal、从未 READY 的 Grain 可以保持 `parent_anchor=None`。

### 4.3 Admission barrier check

`_accept_call_input()` 在把一个 Call slot 转为 READY 前，用 `(call, _parent_anchor(entity))` 查 suppression barrier：

```text
barriered → inputs_terminal + 所有 Call outputs SUPPRESSED
live      → 继续现有 call_transition
```

不扫描 `pending_grains`，也不建立 pending-by-parent 反向索引。已经 WAITING 的 sibling 在后续输入
事实到达时被封闭；有限 DAG 的事实 fixed point 保证其最终获得输入或终态传播。

### 4.4 READY reservation barrier check

ready、immediate-retry、deferred-recovery 三种队列在任何 `READY -> IN_FLIGHT` 前共享同一 barrier membership check：

```text
barriered READY → READY + SUPPRESS → SEALED，并返回给 Engine 发布 SUPPRESSED
live READY    → READY + RESERVE  → IN_FLIGHT，进入 DispatchBatch
```

因此 Grain FSM 只增加一个事件和一条边：

```text
READY + SUPPRESS → SEALED
```

不新增 `QUARANTINED`/`CANCELLED` phase。in-flight 晚到结果仍使用现有
`IN_FLIGHT + REPORT -> SEALED`，只是 Engine 在 publication 前选择 `SUPPRESSED`。

Recovery DispatchBatch 可以被部分过滤：barriered Grain 终态化，live Grain 保持相对顺序、generation 和
原 `udf_retries`。全 barriered 时不得产生空 DispatchBatch。

### 4.5 Cleanup-only reserve

lazy tombstone 允许某次 reserve 只清理 barriered READY Grain，没有 live RPC。内部结果需要同时表达：

```text
(DispatchBatch | None, suppressed_grains)
```

Engine 发布 suppression 并运行事实闭合；Executor 只对非空 batch 创建 `GrainInvocation` 和 `_PendingRpc`。
`ready_count` 可以短暂包含尚未出队的 barriered tombstone，不为精确 runnable count 增加 counter。

### 4.6 In-flight commit barrier check

其他 actor 已经领取的工作不被物理追杀。报告到达 driver 后：

```text
validate DispatchBatch coverage + generation + IN_FLIGHT
→ 显式 failure 始终 FAILED
→ success 命中 barrier 则 SUPPRESSED
→ 其他 success 正常 commit
```

命中 barrier 的 expanded success 必须在创建成功 Expansion、child Entity 或 RowBinding publication
之前截断。共享 Block 中未被引用的行由 ObjectRef 生命周期回收，其他 parent 的健康行仍可引用
同一个 Block。

---

## 5. Opaque/infra recovery 与既有 barrier

`GroupFailure` 本身不进入 recovery，但 barrier 建立后，另一个更早发出的 in-flight RPC 仍可能
以 opaque UDF failure 或基础设施异常结束。此时必须先分区其 `DispatchBatch`：

```text
barriered subset   → SEALED(SUPPRESSED)，不 retry
live subset        → 保持原策略进入 retry/bisect/infra recovery
```

规则如下：

- barriered subset 不消耗 UDF 或 infra retry budget；
- live subset 构造 exact sub-batch，保持原 `udf_retries`；
- generation 只对真正 requeue 的 live Grain 增加；
- 整个 DispatchBatch 都 barriered 时不产生 recovery batch；
- infrastructure failure 即使整个 DispatchBatch 都 barriered，仍替换不可信 actor；actor 健康与数据终态正交；
- opaque exception 永远不会主动安装 suppression barrier。

这只是让既有 recovery 尊重已经存在的终态门禁，不新增 recovery mode。

---

## 6. 多 actor、重复失败与递归传播

### 6.1 Driver 单写，不需要分布式锁

canonical state 和 `_SuppressionBarrierIndex` 都由 driver 内一个 `MicrobatchEngine` 单写。Ray Worker 只
消费 `GrainInvocation` 并返回不可变 DTO。因此多 actor 只产生不同的 report 到达顺序，不产生共享状态
写冲突，也不需要 actor 间广播 barrier。

### 6.2 同批和跨批到达顺序

- 同一个 WorkerDispatchResult：先预扫描全部 GroupFailure reports，tuple 顺序不影响 sibling success；
- 不同 RPC：以 driver 处理完成结果的顺序为线性化顺序；
- success 先 commit：保持 `PRESENT`，之后 barrier 不回滚；
- barrier 先 commit：之后晚到的同 scope success 变为 `SUPPRESSED`；
- stale generation 仍由现有 fencing 拒绝，不增加“barrier 可忽略 stale”旁路。

### 6.3 Parent lookup 不递归

barrier lookup 只使用冻结的直接 `parent_anchor`，期望 O(1)。禁止沿 `_parent_of()` 向祖先 walk；
否则 sibling isolation 会悄悄变成 subtree cancellation，并给每次出队增加 lineage-depth 成本。

### 6.4 下游递归只走现有事实图

被隔离 Grain 的 Call outputs 发布 `SUPPRESSED` 后，现有状态代数继续闭合：

```text
CallInput: FAILED/SUPPRESSED → SUPPRESS_OUTPUTS
Filter/Broadcast             → 传播非 PRESENT outcome
Expand                       → FAILED Expansion（unknown cardinality）
Reduce                       → SUPPRESSED
```

`Engine.advance()` 用 Fact FIFO 到达局部不动点，不递归扫描 descendants。尚未创建的 descendants
不会从失败 Expansion 中产生；barrier 前已经创建或提交的 descendants 不回滚。

---

## 7. 复杂度合同

设目标 Call 共接纳 `N` 个 queue entries，其中 `K` 个最终命中 barrier，batch size 为 `B`：

- 建立/查询 barrier：哈希期望 O(1)；
- 每个 ready entry 只从所属 Call deque 弹出一次；
- barriered entry 被访问后直接 SEALED，不重新放回；
- 一次 reserve 为 O(B + S)，`S` 为本次清理的 barriered tombstones；全部 reserve 的
  `sum(S) <= K`；
- WorkerDispatchResult 预扫描和 DispatchBatch 分区均为 O(batch size)；
- `any_parent` ready queue 排空总成本为期望 O(N)；
- 新增内存为 O(barriered scopes)，不是 O(all grains) 的第二份索引；
- publication 与 downstream facts 的 O(K + facts) 是不可避免的语义闭合成本。

这不会重新引入 PR #9 修复的跨 Call `O(N²/B)` 扫描。仍需诚实保留两个已有边界：

- `pack_by_parent=True` 为收集同 parent 仍可能扫描目标 Call queue；本计划不宣称修复这一既有成本；
- immediate-retry/deferred-recovery queues 仍是 rare-path deque；本计划只保证每个被检查的 barriered Grain不回队。

连续 K 个 tombstone 可能形成一次 O(K) driver cleanup turn，但不会重复扫描。第一版不增加 cleanup
budget、timed wait 或 progress scheduler；只有 benchmark 证明这次线性 pause 成为瓶颈时再设计。

---

## 8. 精确源码修改地图

以下行号基于 2026-09-08 的实现快照。后续变更应优先按类型/函数名定位；未列模块原则上不改。

| 文件与当前行 | 计划修改 | 明确不做 |
| --- | --- | --- |
| [`protocol.py` L91–104](../../rayorch/experimental/multigrain_v3_6/protocol.py#L91-L104)、[L175–190](../../rayorch/experimental/multigrain_v3_6/protocol.py#L175-L190) | 新增公开值 sentinel `GroupFailure(cause)`；`GrainFailureReport` 增加默认 `suppress_siblings=False`；更新 union/export | 不新增 public enum、异常类型、dispatch failure kind 或第二套 report hierarchy |
| [`__init__.py` L14–29](../../rayorch/experimental/multigrain_v3_6/__init__.py#L14-L29) | root API 只新增 `GroupFailure` export | 不公开 gate、barrier、scope 或 policy knob |
| [`worker.py` L78–214](../../rayorch/experimental/multigrain_v3_6/execution/worker.py#L78-L214) | 识别两种 sentinel；同位置按 `GroupFailure > RecordFailure` join；保留完整 batch/多输出原子报告；GroupFailure report 设置内部行为位 | 不捕获新异常、不接收 batch index、不读取 parent/lineage |
| [`transitions.py` L15–31](../../rayorch/experimental/multigrain_v3_6/runtime/transitions.py#L15-L31) | 增加 `SUPPRESS` 事件及唯一合法边 `READY + SUPPRESS -> SEALED` | 不新增 Grain phase 或 WAITING 子状态 |
| [`dispatch.py` L17–66](../../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py#L17-L66) | `parent_anchor` 移入 `GrainRecord`；ready queue 存 `GrainRef`，不保留 `_ReadyEntry` | 不增加第二张 grain-parent 表 |
| [`dispatch.py` L118–224](../../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py#L118-L224)、[L226–383](../../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py#L226-L383) | `inputs_ready` 冻结 `parent_anchor`；ready/immediate-retry/deferred-recovery reserve 共享 `barriered_anchors`；返回 live batch 与 suppressed grains；支持 cleanup-only | 不扫描其他 Call，不在 barrier 建立时扫整条 queue |
| [`dispatch.py` L385–443](../../rayorch/experimental/multigrain_v3_6/runtime/dispatch.py#L385-L443) | 提供 report generation preflight、late-report 所需 `parent_anchor` 查询，以及 opaque/infra `DispatchBatch` 的 barriered/live exact partition | 不发布 Item/Expansion，不静默吞 phase/generation 错误 |
| [`engine.py` L88–113](../../rayorch/experimental/multigrain_v3_6/runtime/engine.py#L88-L113) | 本文件内增加私有 `_SuppressionBarrierIndex`，每个 Engine 独占 | 不新增 `runtime/isolation.py`、Manager 或全局 registry |
| [`engine.py` L189–209](../../rayorch/experimental/multigrain_v3_6/runtime/engine.py#L189-L209) | reserve 传递目标 Call barrier view；发布 lazy suppression；允许 cleanup-only 返回 `None` | 不把 publication callback 放进 DispatchState |
| [`engine.py` L244–248](../../rayorch/experimental/multigrain_v3_6/runtime/engine.py#L244-L248)、[L1161–1177](../../rayorch/experimental/multigrain_v3_6/runtime/engine.py#L1161-L1177) | 复用直接父和既有 lineage，增加必要断言 | 不做 ancestor walk、不改变 Entity identity |
| [`engine.py` L441–700](../../rayorch/experimental/multigrain_v3_6/runtime/engine.py#L441-L700) | batch-aware `commit_reports` 预扫描 GroupFailure；将成功提交拆为 pure prepare + apply；命中 suppression barrier 的成功报告直接发布 SUPPRESSED | 不按 report 顺序安装 barrier，不让 suppressed payload 创建 child/Expansion |
| [`engine.py` L702–795](../../rayorch/experimental/multigrain_v3_6/runtime/engine.py#L702-L795) | 显式 failure 仍发布 FAILED；opaque/infra recovery 前按已有 barrier 分区，仅 live subset 进入原 policy | 不新增 GroupFailure recovery mode，不让 opaque failure 建 barrier |
| [`engine.py` L868–950](../../rayorch/experimental/multigrain_v3_6/runtime/engine.py#L868-L950) | Call input admission 前查 suppression barrier；集中复用 Call outputs 的 SUPPRESSED/FAILED publication | 不建 pending-by-parent reverse index |
| [`executor.py` L242–256](../../rayorch/experimental/multigrain_v3_6/execution/executor.py#L242-L256) | 一次调用 `commit_reports(pending_rpc.dispatch_batch, result)` 完成整个 RPC 的提交 | 不让 Executor 解析 parent、scope 或 payload |
| [`executor.py` L415–463](../../rayorch/experimental/multigrain_v3_6/execution/executor.py#L415-L463) | cleanup-only 时不发 RPC；只有 live batch 才创建 `_PendingRpc` 并更新 metrics | 不 cancel 其他 actor 上的 pending RPC |
| [`executor.py` L467–515](../../rayorch/experimental/multigrain_v3_6/execution/executor.py#L467-L515) | opaque/infra failure 调用 Engine 的 barrier-aware exact partition；infra actor 仍替换 | 不新增 BAD_ITEM 分支或 collateral replay 指标 |

明确不修改：

- `runtime/state.py`：沿用现有 Item/Expansion/lineage canonical tables；
- `program/*` 与 `api.py`：parent-scoped suppression 不是静态 Call option；
- `recovery.py`：两种返回 sentinel 都是终态，不是 recovery policy；
- `execution/result.py`：第一版不增加公开统计字段；
- `ray_backend.py`：现有 WorkerDispatchResult wire 足够，不增加 Ray 专用控制面。

---

## 9. 少量、高覆盖 dummy 测试计划

测试目标是覆盖状态维度的笛卡尔积，而不是为每个 if 堆独立 case。

### T1：双 sentinel、完整 batch 与多输出 join

修改位置：
[`test_runtime_plan_semantics.py` L84–130](../../test/experimental/multigrain_v3_6/unit/test_runtime_plan_semantics.py#L84-L130)、
[`test_executor.py` L205–285](../../test/experimental/multigrain_v3_6/integration/test_executor.py#L205-L285)。

一个参数化 Worker dummy 同时覆盖：

- 全正常、单个 `RecordFailure`、单个 `GroupFailure`、不同 parent 各一个 GroupFailure；
- multi-output 同位置分别出现正常值、`RecordFailure`、`GroupFailure` sentinel；
- `GroupFailure > RecordFailure`，cause 按 output layout 顺序确定；
- 每个输入位置恰有一个 report，grain/generation/顺序与 `GrainInvocation` 对齐；
- `RecordFailure` 保持当前只失败一个 Grain 的行为；
- 两种 sentinel 都不会生成 `DispatchFailure` 或触发 retry。

### T2：同一 WorkerDispatchResult 的 order-independent batch commit

修改位置：
[`test_runtime_plan_semantics.py` L654–765](../../test/experimental/multigrain_v3_6/unit/test_runtime_plan_semantics.py#L654-L765)。

构造 `[A-success, B-success, A-GroupFailure, C-success]`，再交换 report tuple 中 failure 与 success
的相对位置，断言：

- 明确 A 坏项 `FAILED`；A healthy success `SUPPRESSED`；B/C 正常 `PRESENT`；
- 结果与 report 顺序无关；
- multi-output 的每个 Grain 所有 ports 同终态；
- suppressed expanded report 不创建 child Entity 或成功 Expansion；
- 任一 identity/generation/重复/缺失 preflight 错误发生前 canonical state 不变；
- 非 barriered success 的 output contract 错误发生前不安装半个 barrier。

### T3：READY、WAITING、late IN_FLIGHT 与三个队列

修改位置：
[`test_transition_algebra.py` L38–68](../../test/experimental/multigrain_v3_6/unit/test_transition_algebra.py#L38-L68)、
[`test_recovery.py` L36–117](../../test/experimental/multigrain_v3_6/unit/test_recovery.py#L36-L117)、
[`test_recovery.py` L195–292](../../test/experimental/multigrain_v3_6/unit/test_recovery.py#L195-L292)。

一个 Engine/Dispatch 组合场景让同一 barriered parent anchor 同时存在于 ready、immediate-retry、deferred-recovery、WAITING
和另一个 in-flight DispatchBatch：

- 三个队列均只移除 barriered Grain，live Grain 相对 FIFO 不变；
- partial recovery DispatchBatch 保留 `udf_retries`，全 barriered 不产生空 batch；
- cleanup-only 能继续驱动 facts，最终 `is_idle/all_sealed/complete`；
- barriered READY 使用唯一新 transition；replay 的 live Grain才 generation+1；
- admission 后到达的 sibling 不进入 READY；
- late mixed success 只 suppress 相同 `(Call,parent)`，其他 parent 正常提交。

### T4：retry 后恢复与 retry 后显式 GroupFailure

仍放在 `test_recovery.py` 与 `test_runtime_plan_semantics.py`：

1. opaque raise → retry → 全正常：全部 PRESENT，没有 barrier；
2. opaque raise → retry → `RecordFailure`：只精确 Grain FAILED；
3. opaque raise → retry → `GroupFailure`：该次 commit 才建立 barrier；
4. barrier 已存在时另一个 pending RPC opaque failure：barriered subset 不 retry，live subset走原策略；
5. 同样覆盖 infrastructure failure：barriered subset terminal，live subset按预算，actor 仍替换。

同时断言 sentinel 不增加 retry counters；只有真实 replay Grain 计入现有 retry 指标。

### T5：不 reduce 与 nested reduce 的语义对照

构造同一输入图的两个输出分支：

```text
document → expand pages → page Call ───────→ per-page public output
                                └→ reduce → assemble document
```

参数化 `RecordFailure / GroupFailure`：

- `RecordFailure`：坏页 FAILED，健康页面输出仍 PRESENT；document reduce/assemble SUPPRESSED；
- `GroupFailure`：坏页 FAILED，同父未提交页面 SUPPRESSED，另一个 document 完整 PRESENT；
- barrier 只匹配 page Call + direct document parent；同 parent 另一独立 Call 不被直接命中；
- downstream recursion 只靠 Item/Expansion facts；不做 ancestor lookup；
- barrier 前已成功创建的 page/region 不回滚，barrier 后未知 cardinality Expansion 正确闭合。

### T6：真实 Ray 多 actor 晚到结果与复杂度

修改位置：
[`test_executor.py` L205–360](../../test/experimental/multigrain_v3_6/integration/test_executor.py#L205-L360)。

使用两个 replicas、小整数 payload 和显式同步 dummy：

- 一个 RPC 返回 GroupFailure，另一个已领取的 mixed-parent RPC 晚到；
- 同 scope late success SUPPRESSED，其他 parent 结果正常；
- actor 不被取消，Executor 没有永久 busy slot；
- 不依赖脆弱 sleep 断言墙钟顺序；Ray-free T2/T3 提供确定性 order proof。

另在 `test_recovery.py` instrument dequeue/barrier-check visits：

- 放入其他 Call 的 2,000 个 entries 与目标 Call 的 N 个 entries；
- `any_parent` 模式不访问其他 Call；
- barriered ready entry 最多访问一次；总 visits 与 `N + selected` 同阶；
- single-parent 只验证语义/FIFO，不虚假声称其既有扫描已变为严格 O(N)。

---

## 10. 实现顺序、硬门禁与完成定义

### 10.1 实现顺序

1. `protocol.py + __init__.py + worker.py`：双 sentinel 和稳定逐 Grain wire；
2. `transitions.py + dispatch.py`：唯一新 transition、`parent_anchor` 收口、三队列 barrier check；
3. `engine.py`：barrier、batch preflight/prepare/apply、admission/commit/recovery partition；
4. `executor.py`：整批 reports 提交与 cleanup-only 调度；
5. Ray-free T1–T5 与复杂度 instrumentation；
6. 真实 Ray T6 和全部 V3.6 regression；
7. 只有实现与测试成立后，才同步 design、maintainer guide、getting started 和 walkthrough。

### 10.2 实施硬门禁

- 不增加 `BadItemError`、batch index 或 `DispatchFailureKind.BAD_ITEM`；
- 不公开 enum/string scope；公开 API 只有两个明确的 Failure value types；
- 不允许 `RecordFailure` 建 barrier；
- 不允许 `GroupFailure` 进入 retry policy；
- 不允许按 RPC 整体 suppress 或 replay 已返回逐 Grain结果的其他 parent；
- 不允许 barrier 跨 Call、跨 microbatch或递归祖先；
- 不允许 suppressed success 创建 child Entity/成功 Expansion 后再回滚；
- 不允许各队列持有独立 barrier registry；
- 不允许 Worker/Executor 解析 lineage；
- 不允许 `any_parent` ready path 扫描其他 Call 或重复放回 barriered tombstone；
- 实现若需要新 Manager、跨 actor 共享状态、pending-RPC revocation 或 pending reverse index，停止并回到
  设计 review。

### 10.3 完成定义

只有同时满足以下条件，才能把本计划标记为 implemented：

1. `RecordFailure` 保持精确失败；`GroupFailure` 提供同 Call+直接父 sibling isolation；
2. 两种 sentinel 均完整 return batch、均不重试，普通异常恢复行为不变；
3. 同一 WorkerDispatchResult 的 suppression barrier 与 report 顺序无关；
4. 明确坏 Grain FAILED，被隔离兄弟 SUPPRESSED，其他 parent/Call 正常提交；
5. READY、WAITING、ready/immediate-retry/deferred-recovery、late in-flight、opaque/infra recovery 无遗漏入口；
6. nested downstream 仅靠现有 facts 闭合，不做 ancestor cancellation；
7. generation、phase、Item、Expansion、queue、completion 和 multi-output 原子性全部通过；
8. `any_parent` barrier check expected O(1)/entry、aggregate O(N)，PR #9 的 per-Call FIFO 复杂度不退化；
9. 全部 Ray-free 与真实 Ray regression 通过；
10. 文档诚实声明：该机制节省 barrier 后尚未 dispatch 的工作并阻止晚到结果传播，不节省已经
    执行完本次完整 UDF batch 的计算。
