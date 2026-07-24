# Multigrain V2.5 node batch mode 开关

状态：仅记录为 TODO，尚未进入 production 实现。

## 目标

允许单个 node 在两种物理 Dispatch 组批模式之间切换：

```text
elastic streaming
sealed-domain sharding
```

两种模式必须复用同一套 Logical-Grain 语义和执行路径。该开关只能改变：

- READY grains 何时形成 Dispatch；
- 哪些 grains 被装入同一个 RPC；
- RPC 发往哪个 actor replica。

该开关不得改变：

- lineage；
- GrainId/EntityId；
- failure attribution；
- retry；
- suppression；
- normal absence propagation；
- Reduce 的 ordinal 和输出顺序。

## 建议 API 外形

公开名称尚未冻结。最小接口可以是：

```python
Map(MyUdf).ray_options(
    replicas=4,
    batch_mode="elastic",
    batch_size=128,
)
```

或：

```python
Map(MyUdf).ray_options(
    replicas=4,
    batch_mode="sealed_shard",
    batch_size=256,
)
```

该配置属于 node，不属于 Executor 全局配置。同一条 pipeline 的不同 stages
可以选择不同组批模式。

## 模式一：elastic streaming

这是当前 V2.5 已实现的行为：

1. 上游结果到达后，对应 logical grain 进入 node READY queue；
2. 满足任一条件时形成 Dispatch：
   - READY grain 数达到 `batch_size`；
   - `max_batch_wait_ms` 到期；
   - node input domain 已关闭；
3. 一个 Dispatch 发送给一个 actor replica；
4. 同一 Arena 内来自不同 parents 的 grains 可以进入同一个 Dispatch。

该模式强调在线推进、较低首包延迟和 cross-parent elastic rebatching。

## 模式二：sealed-domain sharding

该模式尚未实现：

1. 等待当前 Arena 中 node 的 driving input domain 关闭；
2. 确认所有 candidate occurrences 均已完成分类；
3. 收集该 Arena/node 的完整 READY set；
4. 按 grain 数量在可用 replicas 之间做 deterministic count-balanced 分片；
5. 每个分片仍通过现有 `DispatchPlan` 和 worker manifest 执行。

首版若实现，只允许 count-balanced 分片，不加入 LPT、value-based cost
estimation 或 workload-specific scheduling hint。

`batch_size` 在两种模式中保持一个统一含义：

> 一次发送给一个 replica 的 RPC 最多包含多少个 logical grains。

在 sealed-domain 模式中，如果某个 replica partition 大于 `batch_size`，
必须继续拆成多个有界 Dispatch。

## 必须保持不变的合同

两种模式必须共享：

- 相同的 `GrainId` 和 `EntityId` derivation；
- 相同的 `GrainRecord` schema；
- 相同的 role-tagged inputs 和 output slots；
- 相同的 `AttemptToken` 与 generation fencing；
- 相同的逐 grain `GrainAck` manifest entries；
- 相同的 Success/Failed/Suppressed/normal-absence 语义；
- 相同的 bad-record isolation；
- 相同的 infrastructure retry；
- 相同的 `FiberBarrier` 与 canonical Reduce order；
- 相同的 Arena reclaim 和 detached `RunResult`。

切换模式可以改变 RPC grouping 和 completion order，但不得改变 logical
output、lineage、canonical cause order 或 failure ownership。

## 实现边界

如果未来实现，该改动应限制在物理 planning 层：

1. compiled node execution options 增加一个小型枚举；
2. 在 `Arena.reserve_dispatch` 前选择 packing policy；
3. 继续构造普通、有界的 `DispatchPlan`；
4. 复用现有 Ray transport、worker ABI、manifest validation、commit、retry
   和 isolation。

不得为该开关复制第二套 Executor，也不得增加第二套 outcome 或 lineage
representation。

## 启用前必须覆盖的测试

同一 pipeline 分别运行两种模式，验证：

- Source/Map/Filter/Expand/Reduce identity parity；
- dynamic fan-out，包括 zero-output；
- all-filtered 和 partial-filtered fiber；
- Reduce 输出值和顺序一致；
- Suppressed causes canonicalization 一致；
- `BadRecordError` 定位到同一个 GrainId；
- generic-error isolation parity；
- infrastructure retry 和 stale-result fencing parity；
- `batch_size` hard bound；
- sealed 模式在 input domain 关闭前不得 Dispatch；
- elastic 模式允许在 domain 关闭前 Dispatch；
- Arena reclaim 与 multi-Arena ordered delivery 不变。

性能测试必须与 correctness parity 测试分开。

## 明确不包含

该 TODO 不包含：

- LPT；
- value-based work estimation；
- scheduling-weight metadata；
- cross-Arena batching；
- unbounded actor mailbox；
- actor-side pull scheduling；
- distributed placement/join；
- 自动选择 batch mode；
- 对 Logical-Grain 语义的任何修改。

## 实现触发条件

不得仅为了修复 MinerU 吞吐而实现该开关。当前 elastic 模式通过合理设置
per-replica OCR batch，已经回归旧 MG 性能：

```text
V2.5 elastic，OCR batch 128    521.836 s
旧 MG                            519.950 s
差异                               0.36%
```

只有明确的 regression/ablation 或新 workload 证据表明维护两种物理策略有
价值时，才启动该 feature。
