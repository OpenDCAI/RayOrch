# Docling core-stage V3 feasibility

## 1. 修正后的接入

旧 prototype 对每个 page 调用完整 `DocumentConverter(image)`，错误绕过了 Docling 的 PDF
backend 和核心 stage batch 接口。

当前 prototype：

```text
Docling PDF backend + PagePreprocessingModel
→ immutable DoclingPageSource
→ LayoutModel.predict_batch
→ OcrAutoModel/RapidOCR
→ LayoutPostprocessingModel
→ TableStructureModel
→ PageAssembleModel
→ V3 Reduce
→ ReadingOrderModel
```

Stage 间不传可变 Page。每个 actor 从 immutable source/value 重建临时 Page，只返回本
Stage 的 value result。

## 2. 核心模型特征

单 12-page PDF 原生 profiling：

```text
native E2E        23.28 s CPU
layout            11.42 s
table             10.80 s
OCR                5.75 s
```

H20 layout batch curve：

```text
batch 1     37.5 pages/s
batch 4     64.8 pages/s
batch 8     70.0 pages/s
batch 12    63.6 pages/s
```

GPU batch 8 有约 `1.87×` 的单页吞吐提升潜力。

CPU batch 越大反而越慢，因此 Docling 性能实验必须使用 GPU layout。

## 3. Correctness

单 12-page PDF、V3 core parent-bound：

```text
Markdown chars       39,771 vs native 39,775
tables               5 vs 5
pictures             5 vs 5
texts                351 vs 354
Markdown Jaccard     ~0.99945
```

12 个多样短 PDF / 43 pages：

```text
Markdown Jaccard     12/12 = 1.0
table/picture count  全部一致
text count           少数文档相差 1–4
```

layout detection 对物理 batch peers 存在轻微数值差异，原生 pipeline 本身的 batch 边界也受
producer/queue 时序影响。因此以 Markdown 和结构分布 gate 验证，不要求 cluster bitwise
一致。

## 4. 初步性能

公平单 H20：

```text
Native Docling sequential     35.49 s
V3 parent-bound               22.11 s
V3 elastic                    21.88 s
```

V3 相对 native 约 `1.60×`，主要来自：

- 4 个 PDF parse actors；
- 4 个 CPU OCR actors；
- persistent layout/table actors；
- Stage 之间的 pipeline parallelism。

补充 tuned-native：

```text
Native doc concurrency 1     33.33 s
Native doc concurrency 2     22.75 s
Native doc concurrency 4     18.76 s
V3 parent-bound              22.11 s
V3 elastic                   21.88 s
```

Docling 自带实验性 document concurrency 调到 4 后比当前 V3 更快。说明此前 `1.60×` 主要
来自默认 `doc_batch_concurrency=1`，不是 V3 独有能力。

因此论文中必须报告 tuned native；当前 Docling 不能作为 V3 性能胜出的证据。它仍是重要
架构 case：

- V3 可以把 core models 组成 General DAG；
- 可独立配置 per-stage replicas/resources；
- 可加入 lineage/failure/retry；
- 可继续扩展 Page→Table/Picture 二层 fan-out；
- 但纯吞吐上必须先超过 tuned-native 才能声称系统加速。

当前不能归因到 cross-parent elastic：

```text
parent-bound ≈ elastic
```

## 7. 资源调度短板

初版把：

```text
Docling model num_threads=4
```

直接映射为：

```text
每个 Ray actor num_cpus=4
```

当增加 layout/table replicas 时，静态 reservation 可超过节点 32 CPU，使
`ExecutionPool.ready()` 永久等待 pending actor。例如：

```text
parse 4 + layout 8 + OCR 16 + table 4 + reduce 1 = 33 CPU
```

Docling native 多 document threads 共享进程和模型，不会为每个 stage actor永久保留这些
CPU。V3 adapter 已将 `actor_num_cpus` 与模型内部 `num_threads` 解耦；这是公平调参和后续
runtime readiness deadlock 检测的专项优化点。

## 5. Elastic 触发审计

`parse_batch_size=1` 时，每个 PDF completion 单独发布 pages，layout queue 每次只看到一个
parent，layout RPC 仍为 12。

当 parse batch 增到 2/4 时，elastic layout RPC 从 12 降到 7，但 wall 反而升高：

```text
parse batch 1:
  parent 21.96 s
  elastic 23.94 s

parse batch 2:
  parent 24.70 s
  elastic 26.46 s
  elastic layout RPC 7

parse batch 4:
  parent 33.01 s
  elastic 32.74 s
  elastic layout RPC 7
```

原因是 PDF parse 本身较重，合并多个 PDFs 到一个 Expand RPC 会降低 parse parallelism；
layout 节省不足以抵消。

## 6. 当前结论

已经证明：

- 抽取 Docling 核心 stage 后，V3 可保持高 correctness；
- V3 core pipeline 在该多文档 workload 上快于原生 Docling；
- 主要收益来自跨 stage/persistent actor/pipeline parallelism；
- Docling GPU layout 具有 batch efficiency。

尚未证明：

- cross-parent elastic layout 带来端到端净收益；
- table region Expand 带来收益；
- 大规模独立文档上的统计结果。

下一步应保持 parse batch=1 和并发 parse，不为了填 layout batch牺牲上游 parallelism。若要
获得 elastic 净收益，需要更多并发 Arenas/documents，使多个 parse completions 自然重叠，
而不是把多个 PDFs 串进同一 Expand RPC。

补充 sweep：

```text
single Arena, parse batch=1

layout wait   layout RPC   E2E
200ms         12           21.85s
500ms          9           22.37s
1000ms         9           23.97s
2000ms         6           22.00s
```

等待 2s 可以把 43 pages 聚合成 6 个 layout RPC，接近 batch cap 8，但 wall 没有改善。
原因是 GPU layout 在整个 pipeline 中已经不是主瓶颈；等待抵消了 inference savings。

多 Arena 不会跨 Arena 合并 RPC，因为 V3 queue/lineage state 按 Arena 隔离。这是当前清晰
的语义边界，不应为 Docling 实验临时加 shared queue 飞线。

因此当前 Docling 的论文价值是：

- V3 core-stage pipeline 相对原生 Docling 的端到端 pipeline parallelism 收益；
- 不把这组数据包装成 cross-parent elastic 净收益；
- cross-parent layout packing 作为机制/敏感性负结果。

## 8. 与 tuned-native 的逐 Stage 差距

相同 12 文档/43 pages/单 H20：

```text
Native concurrency=4        18.76–20.69 s
V3 初版                    21.9 s
```

Native profile：

| Stage | Calls | Mean batch | Busy sum | Span | Max concurrency |
| --- | ---: | ---: | ---: | ---: | ---: |
| parse | 43 | 1.00 | 11.97s | 11.81s | 4 |
| layout | 32 | 1.34 | 11.20s | 11.88s | 4 |
| OCR | 26 | 1.65 | 34.33s | 14.63s | 4 |
| table | 41 | 1.05 | 6.11s | 14.64s | 2 |

V3 初版：

| Stage | RPC | Mean grains | Busy sum | Span | Workers |
| --- | ---: | ---: | ---: | ---: | ---: |
| parse | 12 | 1.00 PDF | 34.61s | 10.49s | 4 |
| layout | 12 | 3.58 pages | 2.72s | 10.00s | 1 |
| OCR | 16 | 2.69 pages | 19.36s | 13.83s | 4 |
| table/assemble | 16 | 2.69 pages | 5.49s | 13.99s | 1 |

Native instrumentation 对每个模型调用执行 Python wrapper 和加锁记录，因此 profile wall
`20.69s` 高于无 instrumentation 的 tuned-native median `17.26s`。逐 Stage 数据用于定位
并发/调用形态，不用于最终 wall 表。

对齐后的主要差距：

| 维度 | Native tuned | V3 初版 | 专项修正 |
| --- | --- | --- | --- |
| PDF parse | 4 document threads，共享进程 | 4 actors，但 eager 3-scale PNG | Source 仅传1×，高分辨率按需原生 render |
| Layout | 共享单模型，最多4线程并发 | 单 actor、单并发 | 保持单actor；大batch/多replica无收益 |
| OCR | 共享单模型，最多4线程并发 | 4独立 actors | 4 actors保留；actor concurrency 2仅小幅收益 |
| Table | 共享单模型，观察到并发2 | 单 actor、单并发 | 并发/replica未改善，保持1 |
| CPU reservation | 线程共享 | 初版 model threads=Ray CPUs | actor reservation与model threads解耦 |
| Stage values | 同进程 mutable Page | 跨进程 immutable DTO | 保留，换取隔离/lineage/retry |

关键解释：

1. Native 4 document runs 共享同一模型实例，可并发调用同一 layout/OCR/table model；
   V3 actor 默认单并发。
2. V3 source 初版为每页 render/PNG-compress 1×/2×/3× 图像，parse busy sum 是 native 的
   约 2.9×。
3. V3 Stage 间传值有序列化成本：

   ```text
   Page source       ~2.16MB/page
   OCR result        ~0.29MB/page
   Page assembly     ~0.08MB/page
   ```

4. 增加 layout/table replicas 没有改善，因为同一 GPU 上复制模型增加 startup/显存和竞争；
   OCR replicas 从 4 降 2 也更慢。

## 9. 专项优化结果

### Source payload

```text
1×/2×/3× PNG source      21.97s
只传 1× PNG              16.94s
```

去掉未必会被 OCR/table 使用的 2×/3× eager render 是最大收益，约 23%。

完全不传图片、由每个下游 actor 重开 PDF lazy render 为 `21.38s`，因为 layout/OCR/table
各自重复打开/render PDF，不如一次传 1× 图像。

### Actor concurrency

新增：

```text
per-stage actor max_concurrency
actor CPU reservation 与 model num_threads 解耦
```

结果：

```text
1× PNG, actor concurrency 1     16.94s
1× PNG, concurrency 2           16.52s
1× PNG, concurrency 4           16.76s
```

并发 2 有小幅收益；高于 2 出现共享模型/CPU竞争。

当前最佳 `16.52s` 已快于 tuned-native 单轮 `18.76–26.42s`，但有 correctness 风险：
只传 1× 图像时 OCR/table 需要的 2×/3×图像通过 resize 近似生成，部分文档文本 item 数明显
下降，最低 Markdown Jaccard 约 `0.9703`。因此该配置不能作为最终公平结果。

下一步正确优化应缓存原生高分辨率 crop/render，而不是用 1× resize 代替，保证输出 parity
后再比较。

实现策略：

```text
Source 只传原生 1× page image（供 layout）
OCR/Table actor 请求 2×/3× 或 crop 时：
  从 actor-local PdfDocument cache 按 Docling 原生 PDFium 路径 render
```

不能从 1× PNG resize 到 2×/3×；只有请求 scale 精确命中 source image 时才直接解码。

按需原生 PDFium render 完成后：

```text
Tuned native concurrency=4    23.77 s（同轮）
V3 optimized                  17.08 s
Markdown Jaccard              12/12 = 1.0
texts/tables/pictures         全部逐文档一致
```

V3 相对 tuned native：

```text
1.392×
```

该配置：

```text
Source                  只传原生1× page image
OCR/Table               actor-local PDF cache按需原生2×/3× render
OCR actor concurrency   2
模型/权重/options        与native一致
GPU                     同一张H20
```

这是当前第一个同时满足性能和严格结构 parity 的 Docling 优化结果。

### 严格模型实例/并发对齐

Native concurrency=4 实际共享：

```text
1 Layout model，最大并发4
1 OCR model，最大并发4
1 Table model，观察并发2
4 个并发 document runs，各自拥有短命 page-stage queues
```

V3 公平配置为：

```text
4 parse actors
1 Layout actor，max_concurrency=4
1 OCR actor，max_concurrency=4
1 postprocess/table/assemble actor，max_concurrency=2
layout/OCR/table batch cap 均为4
Source 只传原生1×图；高分辨率由 actor-local PDF cache 原生 render
```

不能把 Native 的 4 个 document threads 映射成 4 个独立 OCR/Layout/Table model actors；
那会复制模型实例和内存，不是同资源比较。

2026-08-02 在同一张 H20、同一 12 PDF / 43 page 输入上，显式初始化 Native 后运行 3 次：

```text
Native startup                 6.303 s
Native measured                16.649 / 17.495 / 17.509 s
Native measured median         17.495 s

V3 parent-bound startup        6.446 / 6.178 / 6.451 s
V3 parent-bound measured       13.113 / 13.270 / 13.208 s
V3 parent-bound median         13.208 s
V3 parent-bound E2E median     19.559 s
```

结论必须分两个口径：

- post-readiness 核心执行：V3 parent-bound 为 `1.325×`；
- 当前每次 `Executor.run()` 都新建并销毁 actors，startup-inclusive first run 仍比
  warm Native measured 慢约 `2.06s`；
- 持久化 ExecutionPool/actor pool 是明确产品短板，但不应混进 elastic 机制结论。

### Stage 调用形态与短板

带 Python instrumentation 的 Native 逐 Stage 数据只用于归因，不用于 wall 主表：

| Stage | Native calls / mean batch | Native max concurrency | V3 RPC / mean batch | V3 peak concurrency |
| --- | ---: | ---: | ---: | ---: |
| parse/preprocess | 43 / 1.00 | 4 | 12 PDF / 1.00 | 4 |
| layout | 32 / 1.34 | 4 | 16 / 2.69 | 4 |
| OCR | 26 / 1.65 | 4 | 16 / 2.69 | 4 |
| table | 41 / 1.05 | 2 | 16 / 2.69 | 2 |

V3 已达到与 Native 相同的重型模型并发上限，并用更少调用处理相同 43 pages。一次严格
profile 中：

```text
V3 measured                     15.052 s
first dispatch → last commit    15.050 s
driver 外围空耗                  0.002 s
commit delay sum                <0.010 s
Markdown Jaccard                12/12 = 1.0
text/table/picture counts       12/12 exact
```

所以当前主要短板不是 driver event loop 或 commit，而是：

1. 每次 run 重建 8 个 actors/模型，约 6.1–6.5s；
2. immutable DTO 跨进程序列化与 PDF render 分散到不同 actors；
3. V3 adapter 把 postprocess、table、assemble 合并成一个 physical Stage，便于减少
   ObjectRef，但不适合把内部三段分别归因；
4. 该 43-page workload 很短，startup 对 E2E 占比过大。

### Parent-bound 与 elastic

同一模型实例、batch cap、并发和输入，仅切换 batch scope：

```text
V3 parent-bound measured median   13.208 s
V3 elastic measured median        13.866 s

parent-bound RPC / fill           72 / 0.671
elastic RPC / fill                64–65 / 0.750–0.773
```

elastic 确实减少 RPC、提高 fill，但本 workload 上 wall 反而慢约 5%。这是有价值的负结果：
OCR/layout 的大 batch 收益不足以抵消重排、共享模型竞争和尾部等待。它不能作为 Docling
elastic 加速证据。

把 batch wait 从 2ms 增到 500ms 也不改变 parent-bound 的 16 个 page-stage RPC，因为
parent-bound 不允许跨 nearest parent 合并；只增加延迟。因此 timer 不是该差距的修复点。

## 10. 论文使用边界与下一步

论文必须同时报告：

```text
startup-inclusive first run
post-readiness steady execution
Native tuned document concurrency
V3 parent-bound
V3 elastic
```

当前 Docling case 可支持：

- V3 core-stage disaggregation 保持严格输出 parity；
- 在同模型实例数/并发下，V3 parent-bound steady execution 快于 tuned Native；
- 更大的跨 parent batch 不保证端到端收益，V3 可用 scope 开关避免负优化。

当前不能支持：

- 把全部 V3 收益归因于 elastic rebatching；
- 用 warm Native 对比 cold V3，或反过来；
- 用 4 个独立重型模型 actors 冒充 Native 4 document threads。

### 10.1 2026-08-03 当前进度快照

本次实验已完成并可作为后续论文复盘的事实源：

| 项目 | 状态 | 已冻结结论 |
| --- | --- | --- |
| Native default | 已有初步结果 | 只能作为非调优基线；不能替代 tuned Native |
| Native tuned concurrent documents | 已完成源码与运行核对 | `doc_batch_size > 1` 且 `doc_batch_concurrency > 1` 时，`DocumentConverter` 用线程池并发 document conversion；每个 document 有独立 page-stage context，converter 缓存并复用模型 |
| V3 parent-bound | 已完成 3 次严格对齐运行 | 43-page 小 workload 上 post-readiness median `13.208s`，相对 tuned Native `17.495s` 为 `1.325×` |
| V3 elastic | 已完成 3 次严格对齐运行 | RPC/fill 改善，但 median `13.866s`，比 parent-bound 慢约 5%；此 workload 上是负优化 |
| Correctness | 已完成 | Markdown Jaccard 与 text/table/picture structure 均为 12/12 exact |
| Driver/commit 归因 | 已完成 | measured `15.052s` profile 中 dispatch span `15.050s`，driver 外围空耗 `0.002s`；不应优先改 driver 多线程 |

本次**未完成**，因此不能作为论文最终主表或已交付能力：

1. 尚未将四臂矩阵收敛为仓库内一个统一 CLI、统一 JSON schema 和固定 manifest；
2. 尚未在更大、公开、多样的 PDF corpus 上做 `1 warmup + 3 measured`；
3. 当前 `Executor.run()` 仍按 run 创建/销毁 ExecutionPool，未实现跨 run actor/model reuse；
4. fused postprocess/table/assemble Stage 尚无内部子阶段 timing；
5. 尚未证明 Docling 的 elastic 机制能在任一正式规模 workload 上带来净 wall-time 收益。

这些未完成项是下一轮工作清单，不应因为已有小规模 steady-state 数字而被省略。

### 10.2 四臂 runner smoke（2026-08-03）

已新增仓库内可复现入口：

```text
rayorch.experimental.multigrain_v3.benchmark.document_docling.core_compare
```

它固定生成：

```text
native_default
native_tuned
v3_parent_bound
v3_elastic
```

并记录 input manifest、环境版本、每臂完整参数、startup/measured/E2E、V3 RPC/fill 以及
相对 Native default 的 Markdown/结构 gate。measured repeats 会循环轮转四臂执行顺序，
JSON 中保留 `execution_order`，避免固定顺序偏差。

在同一 12 PDF / 43 page manifest、1 H20 上的一次 smoke：

```text
native_default  startup 4.748s   measured 30.299s   E2E 35.047s
native_tuned    startup 1.409s   measured 18.360s   E2E 19.768s
v3_parent       startup 5.419s   measured 13.604s   E2E 19.024s
v3_elastic      startup 5.618s   measured 13.234s   E2E 18.852s
correctness     all four arms: Markdown 12/12 Jaccard=1.0; structure 12/12 exact
```

这是**runner smoke，不是最终统计结果**：只有一次 repeat，且 GPU/cache 状态会使 elastic
相对 parent 的单轮方向与 2026-08-02 三次中位数不同。正式论文只使用固定 manifest 上
`warmup=1, repeats=3` 的 median，并保留每 trial 的 execution order。该 smoke 的目的仅是
证明四臂已被同一 CLI、同一 JSON schema 和同一 correctness gate 覆盖。

下一步专项优化按优先级为：

1. 让一个已 ready 的 `Executor` 复用 ExecutionPool，单独消除每-run startup；
2. 增加 Stage 内部子阶段 timing，不拆 DAG、不增加 transport；
3. 用更大且公开的多样 PDF 集重复 1 warmup + 3 measured；
4. 只有在确认 layout/table batch-efficiency 后，才继续优化 elastic timer/queue。

## 11. Table / OCR batching P0/P1 prototype（2026-08-04）

根据 368-PDF 负结果，新增两个默认关闭的实验层，不修改已安装 Docling：

```text
table_batching.py
  TableJob / stable planner / semantic shadow / per-job reference fallback

ocr_batching.py
  OcrCropJob / exact normalized-shape bucket / strict shadow / per-job fallback
```

Table 的限制必须明确：

```text
当前 TableFormer encoder 支持 batch dimension；
当前 TableModel04_rs decoder 使用 scalar .item()；
因此现有 decoder 不是安全的 batch>1 decode。
```

所以 `table_batch_mode="encoder_shadow"` 只做：

```text
多个 table crop encoder batch
→ 原单 table decoder
→ reference semantic compare
→ table fallback
```

它是 correctness/feasibility hook，不是已验证加速模式。真实 12-PDF / 43-page smoke：

```text
reference == encoder_shadow outputs
prepared table jobs          1
batch errors                 0
table fallbacks              0
```

该 shard 的 table jobs 太少，不能代表性能；下一步需要 table-dense corpus 或显式
Page→TableRegion V3 Expand/Reduce 才能测跨 page table batching。

`ocr_batch_mode="recognition_shadow"` 已接入：

```text
detector/classifier         原样逐 rect
recognizer                  shape-compatible cross rect/page bucket
每个 rect                   原 RapidOCR shadow
boxes/text/scores drift     rect reference fallback
最终 page drift             page reference fallback
```

真实 2-PDF smoke：

```text
reference == recognition_shadow outputs
reference measured           3.027s
strict shadow measured       5.765s
```

strict shadow 更慢符合预期：原 reader 仍完整执行，candidate 只用于审计。它验证输出安全，
不构成加速证据。只有收集足够 crop-level exact/fallback 比例后，才能讨论明确命名的
non-shadow accelerate mode。
