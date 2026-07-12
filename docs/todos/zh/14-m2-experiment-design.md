# M2 — 实验设计：真实工作负载、真实基线、规模

这是 M2（成败攸关的里程碑）的设计优先文档。目标是在**真实**的基数变化 GPU
流水线上、对照**可与 CCF-A 对比的基线**、在**多节点规模**下证明该机制，并衡量我们的
核心优势（记录级恢复 + 关系溯源 + 可证明的重排安全再平衡）——而不只是吞吐量。

目标投稿场所：VLDB / EuroSys（见 [`../TODO.md`](../TODO.md)）。所选基线应匹配这些
场所的审稿人会要求的内容（Ray Data、Trident、Spark）。

## 1. 工作负载

选择主工作负载 W1（与 Trident 的评估匹配，因此比较直接）以及第二个 W2 以验证通用性。

- **W1 — 文档解析（主要）。** PDF → 页面（Expand 1:N）→ 布局/OCR（GPU
  Map，每页面成本呈长尾）→ 图/表抽取（第二分支）→ `Relate(on=page_id)` 链接 →
  按文档组装（Reduce N:1）。使用来自公开语料库（如 arXiv/DocLayNet 风格）及我们的内部
  CEPH 语料库的真实 PDF。这是类 MinerU 的流水线，并有意与 Trident 的 PDF 工作负载重叠。
- **W2 — 多模态描述（通用性）。** 图像/视频帧（Expand）→ 通过 vLLM 的 VLM
  描述（GPU Map）→ 去重/聚类（Relate）→ 按资源聚合（Reduce）。使用真实图像；VLM =
  Qwen2.5-VL 类。

两者均具备：真实的长尾 1:N 扇出、数据相关过滤器、GPU 密集阶段以及跨分支 M:N 链接——
即它们覆盖了分类法（`10-...md`）中的每个原语和每种气泡来源。

## 2. 基线

| 基线 | 原因 | 方法 |
|---|---|---|
| **Ray Data（流式批处理）** ★ | 最接近的竞争者；块级血缘 + 动态重分区 | 将 W1/W2 移植到 Ray Data `map/flat_map/groupby` |
| **Trident** ★ | 相同的 PDF 工作负载、自适应调度；导师联系人 | 通过 Binhang Yuan 获取实现；原样运行 W1 |
| **Spark（+ 可选 Daft）** | 经典数据流 + 分区血缘 | PySpark 流水线，通过 UDF/mapPartitions 使用 GPU |
| **朴素 Ray actors** | 无再平衡 / 无关系血缘 | 手写 Ray 任务、连续分片 |
| **我们的方案（MultigrainRayExecutor）** | 该系统 | 被动 IR + LPT + 记录血缘 |
| **我们的方案 − LPT（连续）** | 内部消融 | 分离再平衡收益 |

cedar/Pecan 将被引用用于优化器/UDF 提示的定位，但不一定实际运行。

## 3. 指标

- 固定硬件下的端到端**吞吐量 / makespan**（docs/s、pages/s）。
- 宽 GPU 阶段的**GPU 利用率 / 空闲气泡**（我们的主打机制）。
- **恢复**：注入行/任务/节点故障；测量 (a) 正确性（健康项目上的输出 ==
  无故障运行），(b) 恢复工作量 = 重算记录数，相对于全阶段重算（Spark/Ray Data），(c) 恢复墙钟时间。
- **血缘开销**：有无记录级血缘时的稳态吞吐量/内存；必须优于 Titian 的约 20–30%
  （目标 < 约 10%，理想情况下为低个位数）。
- **规模**：跨节点 × GPU 的强/弱扩展；相对于 OPT bound 的效率
  （复用 `10-...md` 中的 LPT/Graham 分析）。
- **大规模重排安全性**：在真实运行中实证确认 M1 定理（跨分片计划字节一致的结果），而不只是在单元规模上确认。

## 4. M2 运行前需要补齐的原型缺口

按杠杆作用排序。这些是具体的“进一步原型设计”任务。

1. **多节点执行。** 当前 `MultigrainRayExecutor` 进行节点内副本分片 + 全图 microbatch 重叠。需要：跨节点放置、节点间数据移动（Ray object store / plasma）、感知节点数的 `shard_planner`。（阻塞“规模”指标。）
2. **真实算子包装器。** 将实际模型包装为 multigrain UDF：真实布局/OCR（或 MinerU 组件）及 vLLM VLM 描述器，同时保持 UDF 纯净（value-in/value-out，不含 ids —— 保留 M1 假设）。提供 `num_gpus_per_replica`、`gpu_heavy` 属性。
3. **插桩。[已完成]** `metrics.RunMetrics` / `NodeMetric` +
   `lineage_footprint`：每节点 makespan、每分片 busy -> idle-bubble fraction、血缘记录/字节开销、恢复计数器。已穿透 `MultigrainExecutor` 和 `MultigrainRayExecutor`。（真实算子落地后添加真实 GPU-util 采样。）
4. **基线测试框架。** 在 Ray Data、Spark、朴素 Ray 上表达相同的 W1/W2；从 CEPH 共享数据集加载器；共享正确性检查器（比较健康输出）。
5. **故障注入框架。[已完成，MVP]** `ray_executor.FaultSpec` 注入确定性任务/节点崩溃；executor 仅重试失败分片，因此 `recovery_rows` 保持血缘局部（< 全阶段）。通过 `BadRecordError` 的行级隔离已存在。由 `test_metrics_and_recovery.py` 覆盖。（通过真实 Ray actor 死亡的节点终止属于后续工作。）
6. **数据摄取。** 将 CEPH 中的真实 PDF/图像流式输入到带有稳定 `display_key`s（文档 id）的源 `PortBatch`，从而使 trace 在大规模下保持可读。
7. *(推迟到 M4)* 面向偏斜的 Reduce/Relate 分片——W1/W2 初版不需要，但若 Reduce 变为 GPU 密集则需要。

## 5. 实验矩阵

| Exp | 问题 | 设置 | 预期证据 |
|---|---|---|---|
| E1 throughput/bubble | 我们是否在真实长尾上削减 GPU 气泡？ | W1，我们的方案 vs 连续 vs Ray Data vs Trident，1 节点 | 气泡 ↓，吞吐量 ≥ 竞争者 |
| E2 scale | 在多节点下是否仍成立？ | W1/W2，2–8 节点 | 近线性扩展，相对于 OPT 的效率 |
| E3 recovery **[DONE §5c]** | 记录级恢复是否更廉价且正确？ | poison page，真实流水线，我们的方案 vs 分片级 | ✅ 我们的方案 COMPLETE（丢失 1 页，0 冗余 re-OCR）；分片级 ABORT（丢失 248 页，1.5× GPU 浪费） |
| E4 lineage overhead | 血缘是否廉价？ | W1，有/无血缘 | 开销 < 约 10%，优于 Titian |
| E5 reorder-safety | 定理是否在大规模下成立？ | W1，跨分片计划 | 字节一致输出（大规模 M1） |
| E6 ablation | 每个机制的贡献？ | 切换 relation-IR / lineage / LPT | 分离每项收益 |

## 5b. 首次真实工作负载运行（E1 原型，2026-07-10）

**设置。** 在 **4×H20** 上，针对 **368 PDFs**（7072 pages，`MinerU2.5-2509-1.2B`）的真实 Flash-MinerU 解析，通过框架原生 `MinerUReal` 流水线
（`Expand(RealPdfToPages) → Map(RealVlmOcrPage) → Reduce(RealAssembleDoc)`），由 `MultigrainRayExecutor` 驱动。GPU 阶段在**每 GPU 持久 actor 池**上运行（RayModule factory pattern：vLLM 每副本加载一次，通过 `PhysicalHints(replicas=4, num_gpus_per_replica=1.0)` 声明）；渲染/OCR/组装流水线化（4 个 CPU actors 上渲染、4 个 GPU actors 上 OCR、driver 上组装）。基线 = 原始 Flash-MinerU（文档粒度、`SHARD_CONTIGUOUS`、batch=8、inflight=4），通过同一 368 PDFs / 4×H20、**同一 rayorch repo/branch**（`codex/runtime-raymodule-mvp` @ 36e07d2）上的 `regression_run.py` 在本机测量：**wall = 891.98 s** → 0.41 pdf/s（7072 pages → 7.93 pages/s），甚至略高于此前引用的约 840 s。双方 wall 均排除模型加载/预热。基线期间的实时 `nvidia-smi` 直接显示文档粒度连续气泡（GPU1/2 @ 100%，GPU3 @ 61%）。完整日志：`Flash-mineru/logs/baseline_raw.log`。

| config | wall (s) | pages/s | OCR makespan (s) | OCR bubble | 相对 892 s |
|---|---|---|---|---|---|
| **我们的方案，页面粒度 + LPT**        | **567.5** | 12.46 | 463.8 | 6.5% | **1.57×** |
| 我们的方案，页面粒度 + 连续     | 617.2 | 11.46 | 509.4 | 4.2% | 1.45× |
| Flash-MinerU 基线（文档粒度，已测量） | 892 | 7.93 | — | — | 1.00× |

**如实解读。**
- **我们胜过了实测基线（1.45–1.57×）。** 收益是*架构性的*，并非来自 LPT：页面粒度扇出向 vLLM 输入比文档粒度 batch=8 大得多、密得多的批次；持久 actors 移除了每批次的模型/import 成本；渲染/OCR/组装重叠。OCR 约占 wall 的 82%（`463.8/567.5`）；剩余约 18% 是渲染→driver→OCR 的**图像往返**（图像两次穿过 object store）+ priming + 组装尾部。
- **这里 LPT 相对连续仅约 9%，且处于运行间方差内。** 此数据集在已排序 chunks 内是同质的（23 篇基础论文 × 16 个相同副本），且*每页面* OCR 成本相当均匀，因此连续页面切分已约为均衡（4.2% 气泡）。LPT 的决定性胜利需要**高每记录工作量方差**（长尾），我们已在 `bench_gpu_longtail.py` / `bench_gpu_complex.py` 中以合成方式展示。TODO：使用**打乱的**文档顺序（异质 chunks）和/或 token-length weight 重新运行，以暴露该真实工作负载上的 LPT 差距。

**拓宽差距（并使 LPT 重要）的下一步优化：**
1. **Actor-to-actor 页面交接**（渲染 actor → OCR actor，通过 object refs），使图像不经由 driver 往返——应收回大部分约 18%。
2. **Shuffle / token-weighted LPT**，以制造该机制针对的长尾。
3. 对该真实流水线故障注入 → E3 恢复数据。

产物：`Flash-mineru/mg_bridge/run_bench.py`、`mg_bench_results.jsonl`、`logs/{lpt,contig}_raw.log`。框架变更：持久 `_StageActor` 池 + `MultigrainExecutor`/`MultigrainRayExecutor` 中 factory-pattern 算子缓存。

### 输出等价性检查（计算逻辑未变，2026-07-10）

目标：证明页面粒度重构未改变*计算什么*，仅改变*如何调度*。三次运行（baseline / ours-LPT / ours-contig）都在相同 `{stem}/vlm/{stem}.md` 布局下写入每文档 markdown，因此我们进行了比较。

按构造使用同一代码路径：multigrain 包装器调用与基线**完全相同**的 MinerU 函数——`load_images_from_pdf(dpi=200)`、`MinerUClient.batch_two_step_extract`、`result_to_middle_json`、`vlm_union_make`（`dispatch_mineru_class.py` vs `mg_bridge/ops.py`）。Multigrain 仅将页面重新分组为可分片的页面粒度记录；每页面计算是原始代码。

在全部 368 篇文档上测量（baseline vs ours-LPT）：

| metric | value |
|---|---|
| 原始字节一致 | 0/368 |
| 规范化图像名称后字节一致 | 2/368 |
| **shift-robust token Jaccard，中位数** | **0.9989** |
| token Jaccard 平均值 / 最小值 | 0.9844 / 0.5835 |
| token Jaccard ≥ 0.98 的文档 | 344/368 |

**为何不字节一致——以及为何这是预期的。** 批量 vLLM 推理**不是按位可复现的**：连续批处理以不同邻居组合每个页面，因此浮点归约顺序（attention/matmul）改变，贪心解码偶尔翻转一个 token 或使预测 bbox 位移数个 px。证明是引擎而非我们：**ours-LPT vs ours-contig**（代码完全相同，仅分片顺序不同）*也*不字节一致——仅 36/368 匹配，中位 line-match 0.991。原始引擎相对于自身也不可复现。因此 baseline-vs-ours 差异（中位 token Jaccard 0.9989）与这种内在批处理抖动属于**同一量级**，而非逻辑变更。

两种差异来源，均无害：(1) `images/<sha256>.jpg` 名称改变，因为 1-px bbox 抖动改变 crop bytes → 新 content hash；(2) 少量 token 抖动（`Bengio`↔`Ben-gio`、`l/32`↔`l / 32`）。按朴素 line-diff 得出的 `min=0.006`“最差”文档是行移位伪影（插入一行使 zip 错位）——其 token Jaccard 为 0.999。复现：针对三个输出目录的临时比较脚本。

结论：**计算逻辑等价**；残余差异是未经修改的基线自身就表现出的内在 GPU 批处理非确定性。

## 5c. E3 — 真实流水线上的记录级 vs 分片级恢复（2026-07-10）

相对于 Ray Data / Trident / Spark 的核心优势是**恢复粒度**。E3 在*真实* MinerU 流水线上用*相同* vLLM OCR 算子和*相同*确定性 poison page 测量它——唯一变量是框架如何恢复。测量发生在昂贵 GPU 工作所在的 OCR（contents）粒度，因此论述不会与 assembler 对有缺口文档的行为纠缠。

**设置。** 48 PDFs → 992 pages，`MinerU2.5-2509-1.2B`，4×H20，4 个持久 GPU actors，`max_retries=2`。Poison = 一页（`2410.19313v1_copy_2#p1`），它在 OCR 中确定性失败（poison pill / 非瞬态数据故障）。
- **记录级（我们的方案）：** 算子抛出 `BadRecordError(index=i)`；`Map` 隔离第 *i* 行（quarantine + lineage trace），并将分片健康行**作为一个批次**重新运行（向 `_run_with_bad_index` 添加的快路径；没有逐行串行化）。
- **分片级（Spark / Ray Data 粒度）：** 算子在 OCR 后抛出；`Map` 不捕获它，因此 `_run_shards` 重试**整个分区**。故障是确定性的 → 重试耗尽 → 作业中止（分区级行为）。

| strategy | outcome | quarantined | pages lost | **page-OCR execs** (GPU work) | wall (s) |
|---|---|---|---|---|---|
| **记录级（我们的方案）** | **COMPLETE** | 1（已定位） | **1** | **991** (= 992−1，每个健康页面一次，0 冗余) | 82.7 |
| 分片级（Spark-like） | **ABORT** | – | **248**（整个分区） | **1488**（992 有效 + 496 浪费的重试，1.5×） | 166.3 |

**解读。**
- **相同故障，相反结果。** 记录级*完成*且恰好丢弃一个坏页面；分片级*丢失整个 248 页分区并中止作业*。
- **重算足迹是可信的成本指标**（与硬件无关）。我们的方案重新 OCR **0** 个健康页面（991 execs = 每个健康页面一次）；分片级为追逐一个永远无法清除的故障消耗 **1488**（1.5×）。一般规律：分片浪费 = `partition × max_retries`（随重试次数增长），记录浪费 = `0`（与重试无关）。它还在**一半 wall 时间**内成功完成。
- **定位是自动的。** quarantine trace 指出精确罪魁——`logical_item = 2410.19313v1_copy_2.pdf/page=1`、`failed_op = PoisonOcrPage`、`upstream_path = [RealPdfToPages, PoisonOcrPage]`——且不向 UDF 泄露内部 IDs（M1 value-purity 成立）。

**框架变更。** `Map._run_with_bad_index` 现在丢弃已知坏行，并将剩余部分**作为单个批次**重新运行，仅在批次标记另一坏行时递归（poison shard 的 `run_calls` 从 143→4）。最少重算次数*且*不串行化。现有行隔离测试仍通过。

**如实范围。** 这模拟的是**确定性数据故障**（poison pill）——记录级占优的情形。对于*瞬态*基础设施故障（非确定性），分片重试会成功，二者都能恢复；记录级仍将影响范围限制在受影响的行。*下游*消费者的记录级重放（页面修复后仅重跑一个文档的 Reduce）是自然的下一项扩展。产物：`Flash-mineru/mg_bridge/{mineru_poison_ops.py,run_recovery_e3.py}`、`mg_e3_recovery.jsonl`、`logs/e3_full.log`。

## 5d. 全规模 E2E + 输出等价性（无故障及恢复后）

在全规模运行了整个真实 MinerU multigrain 流水线（渲染 → OCR → 组装），并验证产生的 markdown 与裸 Flash-MinerU 基线**逻辑等价**——无故障和记录级恢复后均如此。等价性 = **规范化 token Jaccard**（图像 refs `images/<sha>.jpg` → `images/IMG`，空白折叠）；字节一致不可能，因为即使基线也无法自我复现（VLM batching jitter）。比较器：`Flash-mineru/mg_bridge/compare_md.py`。

**E2E 性能（368 PDFs / 7072 pages / 4×H20，LPT planner）。**
当前运行仅使用公共框架路径：
`MinerUReal.compile()` → `MultigrainRayExecutor.execute_stream`；benchmark 不含 `_pool_for`、`run_shard`、手写 concat，或手写 render/OCR/assemble scheduler。

| metric | value |
|---|---|
| wall | **519.95 s**（全部 368 文档已组装） |
| 相对实测基线（891.98 s） | **1.72×** |
| 相对先前 benchmark-specific scheduler（585.13 s） | **1.13×** |
| OCR bubble fraction | **0.0653**（LPT 保持长尾空闲较低） |
| throughput | 13.6 pages/s，0.708 pdf/s |

**输出等价性，无故障（368 vs 基线）。** 匹配 368/368，0 missing，0 extra。token Jaccard 中位数 **0.9957**，平均值 0.9922；**100 % ≥ 0.90，98.9 % ≥ 0.95，89.4 % ≥ 0.98**；seq-ratio 中位数 0.9996。低 Jaccard 尾部仍是独立基线运行中出现的相同 OCR/table-jitter 系列（`DuoAttention`、`hyper-connection`、密集 malformed tables）；全部 7072 页面和 368 文档输出均存在。没有框架级内容丢失。详情：`logs/cmp_engine_full_lpt_20260712.jsonl`。

**记录级恢复后的输出等价性（48-PDF 子集，1 poison page）。**
Driver `Flash-mineru/mg_bridge/run_recovery_md.py` 在 `1838_reformer…#p0` 上以 `PoisonOcrPage`（record mode）及 `missing_child="fail_closed"` 运行完整流水线。

- **链式抑制有效**：被 poison 的文档产生**0 markdown files**（其 assembler 从未被调用），并作为 `suppressed_incomplete` anchor-grain error 浮现——没有截断/损坏文件，也没有 block。`assembled_docs=47, suppressed=[1838_reformer…]`。
- **幸存者逻辑完整**：47 个健康文档相对基线 → **47/47 ≥ 0.98** token Jaccard（中位数 0.9972，最小值 0.9892），**0** 个文档有 >5 % char gap。

**要点。** Multigrain 输出 ≡ 基线（模去非语义 VLM jitter），无论在正常路径还是记录级故障后；一个永久丢失的页面会级联为*有标记、未写入的*文档，而不是悄然截断的文档。产物：`Flash-mineru/mg_bridge/{compare_md.py,run_recovery_md.py}`、`logs/{cmp_engine_full_lpt_20260712.jsonl,recover_md_result.json,cmp_recover.jsonl}`、`outputs_engine_full_lpt_20260712`。

## 6. 风险与缓解措施

- **Trident/Ray Data 压缩新颖性** → 以记录级恢复 + 关系溯源 + 证明为主；吞吐量是次要的。尽早与 Binhang Yuan 排除重叠。
- **基线工程成本** → 从导师处获取 Trident 实现；开箱即用 Ray Data/Spark；保持 W1 足够小以忠实移植。
- **集群/数据访问** → 先启动单节点 E1/E3/E4（无多节点依赖）以获得大部分论述；集群可用后添加 E2 规模。
- **vLLM/真实模型方差** → 固定版本；预热；报告中位数（复用 `bench_gpu_mineru.py` 中 warmed-actor 方法）。

## 7. 建议执行顺序

1. 原型缺口 #2（真实算子）+ #3（插桩）+ #6（CEPH 摄取）→ 运行 **E1** 单节点（相对 Ray Data 的主打气泡/吞吐量）。最高信号，无多节点依赖。
2. #5 故障注入 → **E3** + **E4**（恢复 + 开销）——核心优势。
3. #1 多节点 → **E2** 规模。
4. **E5**/**E6** 复用上述运行。
5. 在 E1 前与 Binhang Yuan 交谈以安排 Trident 基线。
