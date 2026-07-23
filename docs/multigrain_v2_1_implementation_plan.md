# Multigrain V2.1 实现计划

状态：已被 `multigrain_v2_2_implementation_plan.md` 取代的历史开发路线。

本文档仅保留 V2.1 实现思路，不再指导开发。V2.2 实现必须同时遵循
`multigrain_v2_2_architecture.md` 与 `multigrain_v2_2_implementation_plan.md`。

## 1. clean-break 原则

新建：

```text
rayorch/experimental/multigrain_v2/
test/experimental/multigrain_v2/
```

保留不动：

```text
rayorch/experimental/multigrain/
```

旧 experimental Multigrain 仅用于：

- workload/UDF fixture；
- user-facing 行为示例；
- baseline；
- counterexample/regression test。

禁止在新包边界中传递：

- PortBatch；
- MicroBatch；
- RuntimeResult；
- LineageStore；
- ErrorTrace；
- V1 retry token；
- V1 handler/runtime object。

## 2. 原版 RayOrch 的复用策略

### 借鉴并适配

- `rayorch/dag/pipeline.py`
  - `forward()` introspection；
  - 临时替换 module 为 TraceProxy；
  - 禁止 symbolic value 参与 Python data-dependent control flow。
- `rayorch/dag/compiler.py`
  - 参数签名绑定；
  - unique node naming；
  - dependency/consumer/topology validation。
- `rayorch/dag/executor.py`
  - bounded admission；
  - ready/running/done 状态；
  - synchronous `ray.wait`；
  - completion-driven pipeline overlap；
  - consumer 完成后的 intermediate release。
- `rayorch/runtime/executor.py`
  - Executor-owned actor lifecycle；
  - startup exception rollback。
- `rayorch/ray_module.py`
  - 每个 actor 持有一个常驻 UDF/model 实例。
- `rayorch/env_registry.py`
  - runtime environment。

### 不直接继承

- RayModule `remote/gather/fanout/collect`；
- dispatch mode registry；
- per-node `max_inflight`；
- driver-side payload collection；
- RuntimeRayModule；
- actor-side lineage；
- rank-order concat；
- generic exception quarantine。

尽量复制小型纯算法或抽取无语义 helper，不通过继承旧 executor/private state 强行兼容。

## 3. Phase 0：合同与 test fixture 冻结

### 工作

- 建立 V2 package/test 空骨架；
- 将架构文档中的 prototype 转成 importable frozen types；
- 定义五类 primitive 的最小 deterministic UDF fixtures；
- 定义 PositionKeys/ProvidedKeys canonical codec fixtures；
- 定义 diamond、nested Expand/Reduce、M:N Relate fixtures；
- 定义 terminal writer 的 idempotent test fixture；
- 将所有 intentional breaks 编码为 tests。

### 必须先失败的 break tests

- 不能构造 IdentityGroupId；
- actor 不能返回 EntityId/provenance；
- Relate duplicate parent tuple 被拒绝；
- generic exception 不能变成 PermanentlyMissing；
- zero-output terminal node 不创建 dummy PortData；
- unordered Relate 不被无条件 canonical sort；
- actor-owned `ray.put` ref 不属于合法 return protocol；
- LocalExecutor/public backend registry 不存在。

### Exit gate

所有保留语义和 intentional break 都有一条可读测试名称；尚未实现时允许测试失败，但
不得以旧 V1 行为作为隐含 contract。

## 4. Phase 1：identity、graph 与 authoring

### 实现顺序

1. `identity.py`
   - BatchId/NodeId/PortId/DomainId/RowId/EntityId；
   - canonical value/key encoding；
   - RowRef/IdentityKey。
2. `graph.py`
   - OutputRelation；
   - Operation union；
   - CompiledInput/CompiledInputGroup/CompiledPort/CompiledNode/CompiledGraph。
3. `kernel.py`
   - PrimitiveKernel function bundle type；
   - registry validation。
4. `authoring.py`
   - Port/GraphBuilder/TraceProxy/Pipeline；
   - PrimitiveModule/OperatorFactory/ModuleConfig；
   - explicit input_group/keyed source admission；
   - explicit output arity validation；
   - Select lowering；
   - `Primitive.sink()` zero-output lowering。
5. `kernels/__init__.py`
   - 唯一静态 CORE_KERNELS。

### Compiler 必须处理

- GraphBuilder 只存在于 compile 期间；
- CompiledGraph.nodes 已是 topological order；
- source groups 直接按 DomainId 建立；
- 默认每个 forward 参数独立成组，只有显式 input_group 才共享 source domain；
- ProvidedKeys inputs 必须携带完全相同的 canonical key set；
- SameAs/SubsetOf/Aggregate/Expand/Relate domain propagation；
- operation 与 output relation 相容；
- multi-output Expand 共享 domain；
- selector/role 合法；
- 同一 PrimitiveModule instance 重复形成多个 call sites 时 compile fail；
- zero-output node 必须为 leaf；
- `forward() -> None` 合法；
- value-producing dead leaf 报错；
- graph indexes 可从 nodes/ports 重建。

### 复用边界

可以复用原版 compiler 的 signature/type helper 和 symbolic error 设计，不复用其
`NodeSpec(module=live RayModule)` 数据模型。

### Exit gate

- compiler 不创建 actor、不运行 UDF；
- 一个 graph 只有一份 compiled representation；
- compile tests 覆盖单/多输入、diamond、zero-output 和全部五类 operations；
- 没有 durable serde。

## 5. Phase 2：PortData、provenance、WorkUnit 与 PrimitiveKernel 纯语义

### 实现

- `data.py`
  - ValueShard；
  - PortData；
  - PortSlice/ShardTake。
- `provenance.py`
  - SourceRows/AliasRows/ChildRows/AggregateRows/RelatedRows；
  - parent_edges；
  - ancestor/dependency/role/member sealed queries。
- `work.py`
  - InvocationRef/WorkUnitRef；
  - Row/Fiber/JoinKey/WholeInvocation coordinates；
  - RowAligned/FiberGrouped/RelationClosure；
  - InvocationPlan 和 contiguous WorkRange。
- `execution/arena.py`
  - RunArena；
  - rebuildable indexes；
  - OutputBundle atomic validation/publication。
- built-in kernels 的 `validate/plan/materialize`。

### 单一 materializer

```python
def materialize_invocation(
    node: CompiledNode,
    plan: InvocationPlan,
    accepted: Sequence[AcceptedBatch],
    arena: RunArena,
) -> OutputBundle:
    kernel = CORE_KERNELS[type(node.operation)]
    bundle = kernel.materialize(node.operation, plan, accepted, arena)
    validate_bundle(node, bundle, arena)
    return bundle
```

所有 EntityId 和 typed provenance 只从该 driver path 产生。

### 纯测试重点

- Map/Filter identity reuse；
- Expand parent + sibling ordinal identity；
- Reduce anchor identity 与完整 member CSR；
- Relate ordered parent-tuple identity；
- diamond IdentityKey gather；
- invalid/missing/conflicting ancestors；
- relation-specific order evidence；
- empty/suppressed/failed 的严格区分；
- physical completion/shard order 不进入语义。

### Exit gate

纯函数 property tests 在随机 batch boundaries、replica assignment 和 completion
permutation 下保持 identity/provenance invariant。此阶段没有 LocalExecutor。

## 6. Phase 3：Ray multi-return transport spike

这是正式 runtime 前的必要技术验证，防止在错误 ObjectRef ownership 上继续开发。

### 最小 spike

```text
driver
  -> prepared test actor
  -> actor method static num_returns
  -> manifest_ref + value_refs
  -> driver 只 get manifest
  -> 第二个 actor 直接消费 value_refs
```

### 必测事实

- `actor.method.options(num_returns=N)` 可按 compiled arity 调用；
- manifest 和每个 output column 分别成为 task-return ObjectRef；
- direct task returns 数与 CompiledNode output arity 完全一致；
- driver 不 `ray.get` value refs；
- value refs 可直接传入 downstream actor；
- nested ObjectRefs 在 PortSlice 中由 downstream actor 显式 resolve；
- actor kill 后已完成的 caller-owned task-return refs 不因 actor-owned
  `ray.put` owner 消失而立刻失效；
- exception 会通过 manifest_ref 正确传播；
- terminal writer 使用 `num_returns=1`；
- dropped stale refs 不进入 PortData。

### 明确禁止

```python
def bad_actor_run(self, value):
    return ray.put(value)
```

### Exit gate

在 single-node Ray 上跑通：

```text
Source values
  -> actor A direct multi-return
  -> driver manifest only
  -> actor B receives payload ref
  -> actor B terminal ack
```

driver instrumentation 证明未获取 intermediate value。

## 7. Phase 4：最小真实 Ray vertical slice

### 首条链路

```text
CompiledInput
  -> Map
  -> Map 或 Expand
  -> zero-output terminal writer
```

### 实现

- `ray/actor.py`
  - persistent UDF construction；
  - kernel.invoke；
  - PortSlice gather；
  - direct multi-return。
- `ray/resources.py`
  - complete resource plan；
  - graph-level placement group；
  - transactional prepare/rollback；
  - replacement actor in reserved bundle。
- `execution/node_runtime.py`
  - idle actor ownership；
  - per-invocation contiguous queues；
  - round-robin invocation selection；
  - PendingBatch state。
- `execution/executor.py`
  - one `ray.wait` event loop；
  - one active RunArena per admitted microbatch；
  - manifest-only driver completion；
  - node barrier；
  - downstream ObjectRef routing；
  - terminal writer payload release；
  - lightweight RunResult。

### 与原版 scheduler 的差别

原版：

```text
one node call -> fanout all replicas -> collect all payloads -> node done
```

V2.1：

```text
one invocation plan
  -> many contiguous WorkUnit actor batches
  -> each idle replica receives one batch
  -> collect manifests only
  -> all ranges settle
  -> driver materialize/publish
```

### Exit gate

- active RPC 数不超过 replicas；
- actor mailbox 没有 hidden queue；
- downstream 看不到未 settle node；
- zero-output pipeline 结束后不保留 payload refs；
- Executor close 完整释放 actors/placement group；
- 不调用 `ray.shutdown()`。

## 8. Phase 5：完整五类 primitive data path

按以下顺序接入 kernel.invoke 与 Ray payload：

1. Map；
2. Filter/Select；
3. Expand；
4. Reduce；
5. key Relate；
6. custom Relate。

每类同时实现：

- compile validation；
- WorkUnit planning；
- WorkLayout；
- PortSlice gather；
- worker ABI；
- raw evidence；
- driver materialization；
- sparse failure propagation；
- atomic multi-output publication；
- Ray property/integration tests。

### 必测案例

- multi-input Map diamond；
- Filter false 与 failed 的区别；
- Filter N targets -> N outputs；
- multi-output Expand child counts；
- zero-child Expand；
- nested Expand -> Reduce；
- complete empty Reduce group；
- suppressed Reduce fiber；
- repeated join keys；
- duplicate Relate tuple rejection；
- custom WholeInvocation Relate；
- unordered Relate set equivalence；
- terminal Map/Reduce/Relate zero outputs。

### Exit gate

全部五类 primitive 在 single-node 和 multi-replica Ray 下满足 normalized
identity/provenance/failure invariants，Reduce/Relate payload 不经过 driver。

## 9. Phase 6：retry、split、timeout 与 actor replacement

### 实现顺序

1. BatchAttemptToken/current-stale fence；
2. ContractViolation immediate abort；
3. InfrastructureFailure exact retry；
4. indexed BadRecord；
5. unindexed BadRecord binary split；
6. generic exception exact retry + binary isolation；
7. real deadline；
8. actor kill/replacement；
9. fail-closed suppression propagation；
10. terminal-writer idempotence tests。

### 固定规则

- split 只发生在 WorkUnit ranges 之间；
- Fiber/JoinKey closure 不拆；
- generic failure 进入 isolation 后，child ranges 不重置 exact UDF retry budget；
- actor replacement 前 token 先 stale；
- stale manifest 整体丢弃；
- generic exception 永远不转 PermanentlyMissing；
- Ray 自动 actor-task retry 关闭；
- split depth/retry priority 不暴露为 API。

### 确定性 work-amplification gate

单个持续失败 WorkUnit、初始 batch size 为 `n`：

- 无 index BadRecord 且无 exact retry：attempted WorkUnits `< 3n`；
- generic/timeout 且一次 exact retry：attempted WorkUnits `< 4n`；
- isolation RPC 数不超过：

```text
initial + exact retries + 2 * ceil(log2(n))
```

多坏记录最坏可达 `O(n log n)`，作为实验指标报告，不伪装成线性保证。

### Exit gate

fault injection 覆盖 actor death、timeout、late result、stale token、generic failure、
direct BadRecord 和 downstream suppression；所有 terminal outcomes 都可解释。

## 10. Phase 7：run_stream、fairness 与生命周期

### 实现

- `Executor.run_stream` synchronous iterator；
- `max_inflight` count-bounded admission；
- 多 microbatches 在 nodes 之间 overlap；
- 同 node 内 invocation round-robin；
- completed/yielded microbatch 释放 active slot；
- terminal writer 完成后立即释放 payload；
- returned final ports 仅保留 caller-owned final refs；
- normal close 与异常 rollback。

### 不实现

- 同一 actor RPC cross-microbatch co-batching；
- StageBatcher；
- byte-level admission；
- background Python scheduler thread；
- async public iterator。

### Fairness gate

使用 virtual scheduler 检查 dispatch trace，而不是 wall-clock：

```text
K 个 invocation queues 持续 READY、R 个 replicas 时，
每个 invocation 最迟在 ceil(K / R) 个 refill rounds 内获得 slot。
```

### Lifetime gate

- active RunArena count `<= max_inflight`；
- terminal-writer result 不保留 payload refs；
- stale/unaccepted refs 不进入 published ports；
- Executor close 后 active arenas、NodeRuntimes、actor handles 为零；
- 用户显式持有的 returned final refs 不算 leak。

## 11. Phase 8：metrics、显式 lineage export 与实验 harness

### 默认运行

- in-memory RunMetrics；
- compact status logging；
- sampled failure rendering；
- 不自动写文件。

### 显式实验/debug 输出

- versioned event schema；
- anonymized WorkUnit attempt events；
- provenance/terminal-failure export；
- actor/GPU/network/object-store/process-memory metrics；
- correctness comparison artifacts。

metrics 命名必须对应真实测量值。不得用：

- object-store bytes 冒充 process RSS；
- task time 冒充 wall time；
- submitted work 冒充 GPU utilization；
- proxy counter 冒充 network bytes。

## 12. 全局实现 gates

功能 ready 必须满足：

1. 一个 semantic fact 只有一个 authority。
2. 新包不接收任何 V1 runtime object。
3. actor 不生成 identity/provenance。
4. driver 不收集大型 intermediate payload。
5. actor values 使用 direct task returns。
6. replicas 是 actor capacity，不是 all-replica fanout。
7. max_inflight 只表示 run-level microbatch admission。
8. WorkUnit 同时驱动 batching、replay、failure 和 materialization。
9. node output 只能完整原子发布一次。
10. unordered relation 不被强制排序。
11. generic failures 不会静默丢数据。
12. terminal writer 无 dummy output，并遵守 idempotent retry contract。
13. 没有 LocalExecutor、StageBatcher、lease、manager actor 或 backend registry。

## 13. 推荐的首批源码提交切分

实际开始编码后，建议按可独立 review 的变化拆分：

1. identity + immutable graph types；
2. authoring/compiler + zero-output tracing；
3. provenance + PortData + WorkUnit types；
4. PrimitiveKernel protocol + Map kernel pure semantics；
5. Ray multi-return transport spike；
6. resources + ReplicaActor + NodeRuntime skeleton；
7. Map terminal-writer vertical slice；
8. remaining primitives；
9. recovery；
10. streaming/metrics。

每个提交只在用户明确要求时创建；本文档不授权自动 commit。
