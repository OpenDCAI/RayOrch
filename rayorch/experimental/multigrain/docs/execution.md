# Execution：typed handlers、Local、Ray 与 Recovery

执行层原则：

> Primitive 定义逻辑语义；executor 决定何时、在哪里、以什么合法 partition 执行。

## 1. Mandatory graph verification

`verify_graph(graph)` 不是可选 pass：

- `Pipeline.compile()` 构图后调用；
- `MultigrainExecutor.execute()` 调用；
- `MultigrainRayExecutor.execute_stream()` 调用。

验证失败抛 `GraphValidationError`，execution 不开始。它检查 ref/topology、grain、
operation/relation pairing、Reduce anchor/member、Relate roles 和 node-local output
forest。

## 2. Typed operation handlers

```python
class OperationHandler(Protocol):
    def prepare(self, node: NodeSpec): ...
    def execute(
        self,
        node: NodeSpec,
        runtime,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution: ...
```

`OperationHandlerRegistry` 以 `type(node.operation)` 查找 handler：

| Operation | Handler behavior |
|---|---|
| `MapOp` | 从 factory 构造 Map；执行 attributable recovery |
| `FilterOp` | 构造 Filter 并执行 aligned mask |
| `ExpandOp` | 从 `ChildrenOf` 恢复 parent/child grain label |
| `ReduceOp` | 从 `AggregateOf` 恢复 incomplete policy 并 group_by |
| `RelateOp` | 从 matcher 与 `RelatedFrom` 恢复 roles/join/adapter |
| `FilterByMaskOp` | 无 UDF runtime；过滤 inputs/annotations 并排除 mask port |

handler 不使用 operation enum，也不处理 graph-wide scheduling、actor lifecycle 或 shard
planning。

Expand handler 会明确拒绝 mixed relations，并报告 IR 已预留但 runtime
`mg.out.same/children` materialization 未实现。

## 3. Local executor

`MultigrainExecutor.execute(graph, inputs)`：

1. `verify_graph(graph)`；
2. 用 `GraphInputRef` 初始化 `PortRef → PortBatch` context；
3. 按 `graph.nodes` 的拓扑顺序执行；
4. 由 registry resolve handler；
5. prepare/cache 每 node runtime；
6. 校验 output count 和 `PortBatch.name == OutputSpec.grain`，再按 `NodeOutputRef` 写回
   context；
7. 按 `graph.outputs` 返回。

runtime cache key 是 node name，同时保存完整 `NodeSpec`，防止同一个 executor 对不同
graph 错误复用同名 model。

Local recovery 当前只支持 Map inline record retry。deferred retry、非 Map record
retry 和 Ray-specific shard/actor recovery 在执行前显式拒绝。

## 4. ExecutionCoordinator

coordinator 是 driver-side DAG scheduler，负责：

- 多 microbatch admission 和 sequence；
- 根据 graph dependencies 判断 readiness；
- 独立 branches 与不同 microbatches overlap；
- bounded inflight/backpressure；
- 默认 ordered delivery；
- 聚合和 drain deferred Map records。

它不执行 UDF、不维护 lineage、不选择 shard indexes。logical output metadata 仍由
primitive output builder 生成。

## 5. Ray executor

Ray backend 在 Local node semantics 上加入：

- `WorkerPoolSpec(replicas, gpus_per_worker)` 驱动 persistent actor pools；
- worker 内复用一个 `MultigrainExecutor` 和 lazy operator/model；
- coordinator 驱动 stream execution；
- row-shard planning、shard retry、actor replacement 与 adaptive isolation。

### Capability decisions

```python
if is_row_partitionable(node):
    # row sharding is allowed
```

当前 row partitionable operations 是 Map、Filter、FilterByMask 和 Expand。Reduce 需要
完整 `AggregateOf` group，Relate 需要 cross-role context，因此二者 whole-batch。没有
单独的 complete-group capability 查询。

capability 是必要条件，还依赖 verified graph、row-local UDF purity、aligned input
partitioning 和 adapter permutation-equivariance。

### Exact shard plan

planner 返回 `list[list[int]]` 后，Ray 必须调用：

```python
validate_shard_plan(partitions, row_count)
```

它检查 indexes 不越界、不重复、不遗漏，确保每个 row 恰好属于一个 partition。所有
aligned inputs 对同一 partition 使用相同 indexes。

contiguous planner 连续等分；LPT 按预计 work 贪心重排。两者都只能改变物理顺序。

### Persistent pools

当 node 配置多个 replicas 或 GPU 时，Ray 为它创建 persistent actor pool。每个 actor
prepare/warm 一次 operator，跨 shard 和 microbatch 复用。row-partitionable node 把
同一 invocation 的 shards 分配到 pool；Reduce/Relate 等 whole-batch node 则在不同
microbatches 之间 round-robin，避免所有请求挤在 actor 0。重试可切换 replica，actor
死亡只替换对应 slot；data-level bad record 不重启 actor。

## 6. Recovery

`RecoveryPolicy` 包含：

- `max_record_retries`；
- `retry_timing` (`INLINE` / `DEFERRED`)；
- `max_shard_retries`；
- `on_shard_exhausted` (`ABORT` / `DEGRADE`)；
- bounded `IsolationBudget`。

### Record-level

`BadRecordError(index=local_index, retryable=...)` 将失败归因到 invocation-local row。
Map 可移除坏行、保持 healthy dense execution、singleton retry，并按原 identity order
合并。deferred retry 由 Ray streaming coordinator 聚合后强制 inline drain。

### Shard-level

普通 exception 没有 row attribution，Ray 重试整个 shard。预算耗尽且 policy 为
`DEGRADE` 时，只对 row-partitionable node 做 bounded recursive isolation；成功子集
立即保留。

### Fail-closed cascade

永久隔离的 descendant error 带 ancestry。`Reduce(..., missing_child=FAIL_CLOSED)` 根据
anchor ancestry 跳过 poisoned anchor UDF，输出 incomplete placeholder 和
`suppressed_incomplete` error。

## 7. Mixed-output runtime boundary

`ExecutionGraph` 可以手工表达一个 Expand node 的 `SameAs` / `ChildrenOf` identity
forest，`verify_graph()` 也能验证 earlier-output source、grain 和无 forward/self
reference。

当前 handler/runtime 只支持所有 Expand outputs 共享同一个 `ChildrenOf` relation 和
group shape。`mg.out.same`、`mg.out.children`、marker analysis 和 Local/Ray materializer
仍 deferred。

## 8. 明确不存在的 execution claims

当前 graph operation union 只有 Map、Filter、Expand、Reduce、Relate 和
FilterByMask。没有 optimizer pass 插入的 rebatch/materialize operation，也没有
对应 handler 或 storage backend。公开 `data.rebatch()` 是对 `PortBatch` 的数据辅助，
不是 graph execution stage。

## 9. 执行不变量

1. compile、Local、Ray 都必须验证 graph；
2. shard plan 必须 exact-cover rows；
3. aligned ports 使用相同 partition；
4. multi-output shared cohorts 保持 identity 对齐；
5. shard merge 不丢 errors/relation evidence；
6. Reduce 前满足 complete-group requirement；
7. retry 不改变 logical identity/order；
8. unsupported relation/policy 显式失败；
9. Ray 不重新实现 primitive output/lineage semantics。

## 10. Compiled Relate boundary

compiled `RelateOp` 必须携带 `KeyJoinSpec`（来自 `on=`）或
`RelationAdapterSpec`（来自 dotted `relation_adapter=`）。live `relation_fn` 仅允许
eager execution；compiled executor 没有 relation-function injection。
