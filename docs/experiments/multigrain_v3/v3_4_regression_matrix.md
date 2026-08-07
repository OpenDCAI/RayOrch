# Multigrain v3.4 单 Executor 回归矩阵

更新时间：2026-08-04

## 1. 本轮结论

v3.4 继承 v3.3 的 Port、Domain、Grain、Item、Shape、Arena 与 Worker ABI，不改变业务数据结构和结构关系语义。本轮只精简执行层：删除并列的 LocalExecutor/RayExecutor，完整 Pipeline 统一由一个 Executor 在 Ray 上执行。

这不是新增 backend 抽象。v3.4 明确不引入 ExecutionBackend、ReferenceRunner 或 local runner：

- Ray 是正常多进程执行环境；
- Arena/Worker 的低层单测仍可不启动 Ray；
- dummy、nested、failure、packing、multi-Arena 和 retry 端到端回归全部走真实 actor/RPC；
- v3.3 性能数据仅作为历史基线，不能直接改写为 v3.4 结果。

## 2. 源码收敛

| 项目 | v3.3 | v3.4 |
| --- | --- | --- |
| 顶层执行入口 | LocalExecutor + RayExecutor | Executor |
| 生产执行器代码 | 171 行 local + 482 行 Ray | 485 行单一 Executor（含启动失败清理） |
| 完整 Pipeline 测试 | local 与 Ray 混合 | 全部 Ray |
| Worker 名称 | LocalWorker | Worker |
| 块引用 | BlockRef + RemoteBlockRef | 单一 BlockRef(handle)，协议不感知 Ray |
| 无 Ray 代码 | LocalExecutor、LocalBlockStore、Arena、Worker | 仅 Arena/Worker 直接单测 |
| backend 接口 | 无 | 无，且不新增 |
| 顶层选择 | 开发者选择 Local/Ray | 无选择 |

删除内容包括 LocalRunResult、本地专用 CallMetrics、LocalBlockStore、LocalExecutor、RemoteBlockRef 及 local dummy matrix。Worker ABI 单测使用的 MemoryBlockStore 只定义在测试文件中，不进入生产包和论文组件图。

## 3. 当前回归证据

完整测试结果：53 passed，4 warnings，57.87s。

| 能力 | v3.4 证据 |
| --- | --- |
| compile-time Port/Domain 规则 | 单/多输入输出、分支、broadcast、reduce、optional 等保持通过 |
| Arena 语义 | fan-out、filter、group、failure、hard limit、stale generation 保持通过 |
| 深层结构 | 20 层 unary、5 层 binary 均通过唯一 Executor |
| batching | 1,000-parent、batch cap 16/32/64 的 parent-bound/elastic 摘要一致 |
| multi-Arena | in-flight overlap、有序输出和 Arena HWM 通过 |
| failure | bad leaf parent isolation、RecordFailure、multi-output atomicity 通过 |
| actor 生命周期 | startup barrier、持久实例、跨 run 指标隔离通过 |
| recovery | actor crash、replacement、generation replay 通过 |
| 结构操作 | F.expand/F.reduce/F.filter 不创建 actor 或 RPC |
| import 边界 | 导入 v3.4 不初始化 Ray；顶层只公开 Executor |

测试告警均来自 protobuf、pynvml 和 Ray 的未来行为提示，不是 v3.4 correctness failure。

## 4. 现有实验的可迁移性

v3.3 已经验证可表达的 V3-managed 拓扑在 v3.4 中保持可表达，因为 Program、Domain ancestry、GroupShape、atomic commit 和 Worker ABI 均未改变：

- MinerU elastic / parent-bound；
- Docling page-level 与 core-stage 多分支；
- Video A/B/C；
- Page→Region 多层 Expand/Reduce；
- deterministic bad leaf 与多输出失败。

Ray Data、native 和 reference-only baseline 不属于 v3.4 Program，继续复用原 runner。

## 5. 性能状态

本轮没有运行 v3.4 MinerU 4、48 或 368 PDF，因此不发布 v3.4 pages/s、GPU 利用率或端到端 wall time。

历史 v3.3 clean 结果继续记录在 v3_3_regression_matrix.md：

- 368 docs / 7,072 pages；
- elastic measured 592.975s，11.9263 pages/s；
- 相比历史 v3 elastic 约 -0.88%，处于约 1% 的真实运行波动范围；
- 旧 635.947s 样本因重叠 Ray session 被判定为无效。

这些数字只能作为 v3.4 后续 paired regression 的比较基线。

## 6. v3.4 真实 workload gate

必须按以下顺序推进：

1. 4 PDFs：correctness、actor、ObjectRef smoke；
2. 48 PDFs：可行性、显存、packing、throughput；
3. 368 PDFs：elastic 与 parent-bound paired regression。

每一级同时检查文档/page 数、逐文档 correctness、RPC/grains、Arena HWM、driver/GPU memory、输出 artifact 和资源隔离。GPU/Ray preflight 失败时，性能数据只能保留 correctness 结论。

## 7. 设计价值

单 Executor 使论文和实现共享同一条故事线：

    Port/Domain 定义关系
      → Arena 维护唯一语义事实
      → Executor 驱动多 Arena 并管理 Ray actor/RPC
      → Worker 执行 value-only 列式 UDF

Local execution 不再是系统模式，测试也不再维护第二套调度循环。因此新增功能只需要回答其 Program/Arena/Executor/Worker 归属，不需要同步修改两种 Executor 或证明 backend parity。
