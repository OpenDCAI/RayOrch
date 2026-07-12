# TODO：谱系引导的部分重放

状态：当前 Runtime MVP 之后的研究方向。

## 动机

功能丰富的框架并不会自动成为研究贡献。
GPU 编排、DAG 执行、谱系、重试和 Ray 集成均很有用，
但每一种机制在相关系统中都已存在。

研究问题应当是：

> 异构 GPU 管道如何利用细粒度谱系、operator 成本
> 和故障历史，选择一种隔离与恢复范围，从而在坏记录和系统故障下
> 保持 goodput？

这会将编排、谱系和恢复转化为一个机制，而非
三个彼此独立的功能。

## 建议定位

暂定标题：

> A Lineage-Guided Fault-Isolated Runtime for Heterogeneous AI Data Pipelines

目标工作负载是一个多阶段 AI 数据管道，其中包含 CPU、GPU、
VLM 和 LLM operator，并具有微批重叠。

## 核心机制

谱系必须参与恢复决策，而不应只作为
执行后的报告设施。

对于坏记录：

```text
one PDF fails in OCR
  -> identify the affected record and dependency branches
  -> reuse successful PDF decode and layout outputs
  -> replay only the necessary OCR sub-batch and downstream subgraph
  -> allow unrelated microbatches and stages to continue
```

对于系统故障：

```text
GPU actor or node fails
  -> inspect committed outputs and lineage
  -> identify records whose computation was not committed
  -> restart from the nearest recoverable stage
  -> avoid restarting the entire pipeline
```

## 候选贡献

### 异构 AI DAG 执行

在 CPU、GPU、VLM 和 LLM operator 之间对管道微批进行流水化，同时遵守
每个阶段的资源、副本数和 inflight 限制。

### 混合粒度谱系

- 将保留记录的 `1:1` 执行压缩为共享路径谱系。
- 仅在 `1:N`、`N:1` 或 `N:M` operator 改变
  记录身份时具体化父边。
- 支持低成本影响分析，而无需完整的单元格谱系。

第一项是**必需优化，而非当前行为**。MVP
当前为每行存储多个 Python 元数据对象，并通过 `take` 和 `with_values`
复制谱系 / 祖先 / ordinal 结构。这对于 7k 页 MinerU 实验已经足够，但在声称
百万记录规模之前，必须替换为共享的 `1:1` 路径和紧凑/列式祖先信息。

### 谱系引导的故障隔离与部分重放

利用谱系、operator 成本、故障类型和既往故障，在以下方式之间进行选择：

- 直接移除坏记录；
- 递归批次隔离；
- 从已具体化的中间结果重放；
- actor 级重试；
- 子图重放；
- 对不安全的副作用 operator 中止。

该策略应处理坏输入、Python 异常、GPU OOM、actor 崩溃
和节点丢失，且不应不必要地重放健康工作。

## 当前基础

Runtime MVP 已提供：

- Ray Actor 副本和资源配置；
- 管道级微批重叠所需的原语；
- 记录身份；
- 多父 DAG 谱系；
- 坏记录隔离；
- 健康记录的持续执行；
- 与 Flash-MinerU 兼容的 DAG 拓扑。

当前谱系主要用于报告和错误归因。

重要执行注意事项：公共 multigrain executor 当前提供
逐节点分片和持久化池，但 MinerU 基准测试使用的高性能
render/OCR/assemble 重叠，是通过 `Flash-mineru/mg_bridge/run_bench.py` 中的私有
pool/shard API 编排的。图本身是框架原生的（`MinerUReal` 将
`Pipeline → Expand → Map → Reduce` 编译为被动 IR）；特定于基准测试的
物理调度器尚未置于一个公共 executor API 之后。统一该
调度器是真实的、阶段全局延迟重放屏障的前提条件。

## 缺失的研究闭环

论文级系统仍需要：

```text
lineage
  -> determine precisely affected computation
  -> locate reusable committed intermediates
  -> replay only affected records and DAG nodes
  -> measure the recovered GPU goodput
```

这需要：

- 中间结果具体化和提交策略；
- 确定性、非确定性和副作用 operator 声明；
- actor 和节点故障检测；
- 重放规划；
- 通过缓冲 emit API 支持 `1:N` 和 `N:M` 谱系；
- 持久化谱系和执行元数据；
- 自适应隔离，而不是无条件二分拆分。

## 范围边界

正确性论证假设 UDF 值纯度：一条记录的输出独立于
内部 ID、全局位置、执行时间和副本。
因此，当前模型刻意不定义全局有状态/会话化 operator、全局排序、跨记录
去重、迭代/循环数据流或不安全外部副作用的语义。
这些需要显式的状态/具体化/提交契约，而非
被悄然视为可重排序的 map。

`Relate` 表示 M:N 证据和谱系，但它当前的键 join 执行
是在进程内存中进行的哈希 join，且会针对每个键进行笛卡尔扩展。大型
分布式 M:N join 需要分区、溢出/背压和分布式
具体化；表示的完整性不应与执行可扩展性混为一谈。

## 评估要求

使用 Flash-MinerU，另加至少一个 LLM 或多模态数据治理工作负载。
尽可能在一个 8-32 GPU 集群上评估。

对比对象：

- 普通 Ray 管道；
- Ray Data 或 Ray Data LLM；
- 整个微批重试；
- 逐记录执行；
- 固定二分拆分隔离；
- 无谱系执行。

测量：

- 无故障吞吐量和 GPU 利用率；
- 每单位 GPU 时间的成功记录数（`goodput`）；
- 在不同故障位置和故障率下浪费的 GPU 工作；
- 重放范围和恢复延迟；
- 谱系运行时、内存和存储开销；
- 谱系查询延迟；
- 跨阶段、副本和 GPU 的可扩展性。

预期结果是：

- 无故障时接近基线的性能；
- 故障时比粗粒度重试具有更高的 goodput；
- 显著更小的重放范围；
- 低且可预测的谱系开销。

## 验收标准

当系统证明谱系引导恢复同时满足以下条件时，该方向将成为可信的 VLDB Research Track 贡献：

1. 相比批次或阶段重放，具有实质性更高的精确度；并且
2. 在真实异构 AI 管道上，端到端 GPU goodput 可测量地更好。

在此之前，Runtime 仍是一个强大的工程框架和实验
平台，但研究主张尚不完整。
