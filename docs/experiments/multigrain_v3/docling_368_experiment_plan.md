# Docling 368-PDF rebatching 实验计划

## 1. 问题

本实验不再使用 12 PDF / 43 page 小样本下结论，而是在与 MinerU full benchmark 相同的
368-PDF 输入上回答：

```text
Docling tuned multi-document in-flight
vs V3 parent-bound
vs V3 elastic

在动态页数和 Stage 时延波动下，cross-parent elastic rebatching
是否带来稳定、可重复的净 wall-time 收益？
```

Native default 仍保留为未调优下界，但不是 elastic 归因的主要对手。

## 2. 输入冻结

固定目录：

```text
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru
```

2026-08-03 生成的有序 manifest：

```text
/tmp/mgv3-docling-368/manifest.json
manifest SHA-256:
b9030601160f0873d4790de7494d6c9a3bc9a8486778daaa9556872ba08cbb53
```

数据特征：

```text
PDFs                    368
pages                  7,072
input bytes            1,101,539,712
page count min/median  9 / 16
page count mean        19.217
page count p90/p95     34 / 37
page count max         53
unique page counts     13
```

该语料由 23 个基础 PDF 各复制 16 次构成。它适合控制内容差异并放大页数/算子时延分布，
但不是 368 个独立文档的 diversity 语料；论文必须如实说明。所有 arm 使用同一个有序
manifest，不允许重新 glob 或 shuffle source。

## 3. 四臂合同

```text
native_default
native_tuned
v3_parent_bound
v3_elastic
```

共同设置：

```text
4 × H20
Docling 2.117.0
same PDF backend / model weights / options
layout/OCR/table batch cap 8
num_threads 4
strict Markdown + pages/texts/tables/pictures correctness
```

Native tuned：

```text
doc_batch_size          24
doc_batch_concurrency    4
```

V3：

```text
microbatch_size         24 PDFs/Arena
max_inflight_arenas      3
parse replicas           4
1 layout actor × 1 GPU, concurrency 1
4 OCR CPU actors, concurrency 1
3 post/table/assemble actors × 1 GPU, concurrency 1
reduce replicas          4
```

V3 parent-bound 与 elastic 唯一语义差异是 `batch_scope`。

Native tuned 不是单进程只看见一张卡。它使用：

```text
4 independent processes
each process binds exactly one H20 before importing Docling/Torch
each process owns one DocumentConverter
each process doc_batch_concurrency=4
fixed page-count-balanced document shards
global wall = four-process makespan
```

V3 的 1 Layout + 3 Table GPU actors 是**按 Stage 分配四卡**，而不是每卡复制完整
DocumentConverter。这是 RayOrch per-stage resource allocation / pipeline parallelism 的系统
能力；不声称双方模型复制边界同构。若要求每 GPU 同时常驻 Layout+Table，则需要 composite
GPU lane 或 placement affinity，当前不为 benchmark 引入该 runtime 飞线。

## 4. 抗抖动与恢复

长时实验不使用四臂同进程 convenience runner。每个 `(repeat, arm)` 是独立 Python
进程：

- Native 不与 Ray GPU context 共存；
- 一臂完成立即写 `summary.json`、`documents.json.gz` 和共享 `results.jsonl`；
- 中断后只重跑缺失 arm；
- 四个 repeats 使用 Latin rotation：

```text
r1: default → tuned → parent → elastic
r2: tuned → elastic → parent → default
r3: parent → default → elastic → tuned
r4: elastic → parent → tuned → default
```

每个 arm 因而各有一次首位，并且 parent/elastic 的前后顺序各出现两次，减少 GPU warm-up、
文件缓存、温度和后台负载固定顺序偏差。
报告同时提供：

```text
startup / measured / E2E 全部样本与 median
measured / E2E coefficient of variation
同 repeat 的 parent / elastic paired speedup
RPC / batch fill
逐 arm correctness artifact
```

单次异常快/慢不能作为结论。若任一核心 arm 的 E2E CV 明显偏高，继续补 repeats，而不是
删除异常值。

## 5. 执行入口

生成 16 条顺序命令：

```bash
python -m \
  rayorch.experimental.multigrain_v3.benchmark.document_docling.core_matrix \
  --manifest /tmp/mgv3-docling-368/manifest.json \
  --output-root /tmp/mgv3-docling-368/matrix \
  --repeats 4 \
  > /tmp/mgv3-docling-368/commands.json
```

按 `commands.json` 的顺序逐条执行 `command`，不能并行抢占同一 H20。完成后：

```bash
python -m \
  rayorch.experimental.multigrain_v3.benchmark.document_docling.core_report \
  --results-jsonl /tmp/mgv3-docling-368/matrix/results.jsonl \
  --expected-repeats 4 \
  --expected-documents 368 \
  --output /tmp/mgv3-docling-368/matrix/report.json
```

## 6. 分阶段 gate

正式 full matrix 前：

1. 48-PDF pilot：四臂各一轮，确认输出、内存和单臂时长；
2. 368-PDF pilot：先跑 tuned / parent / elastic 各一轮；
3. 若全部成功，再完成四 repeats；Native default 若时间过长仍需至少完整一轮，并在最终
   主表中明确其 repeats 数，不能伪装成四次中位数。

## 7. 结论规则

只有同时满足以下条件，才可写“Docling workload 上 elastic 有优势”：

```text
paired parent/elastic speedup median > 1
median wall 与 paired runs 方向一致
RPC/fill 改善与 wall 改善同时出现
correctness 不退化
系统波动不足以解释差值
```

如果 elastic 只减少 RPC、提高 fill，但 wall 不改善，则报告为机制成功、端到端负结果；
不把 MinerU 的 elastic 正收益迁移解释到 Docling。

## 8. 48-PDF cap=4 预实验与参数修正

2026-08-03 完成 48 PDF / 992 pages：

| Arm | Measured | E2E | all-stage RPC | all-stage fill |
| --- | ---: | ---: | ---: | ---: |
| Native tuned | 388.218s | 392.886s | N/A | N/A |
| V3 parent-bound | 457.622s | 463.643s | 864 | 0.9552 |
| V3 elastic | 472.289s | 478.196s | 840 | 0.9846 |

三臂 48/48 Markdown 和结构完全一致。elastic 减少 24 个全 DAG RPC、提高 fill，但单轮
wall 变慢。

该结果暴露出原参数不适合检验 rebatching：cap=4 时，48-PDF parent-bound 理论 page-stage
fill 已为 `0.96875`，368-PDF 也为 `0.93644`；绝大多数 PDF 自身已经能装满 4-page batch，
跨 parent 几乎没有空间。直接用 cap=4 跑 368×4 repeats 会消耗十余小时，却不能有效回答
rebatching 问题。

此前单 H20 layout curve 在 batch 8 仍保持最佳附近吞吐。因此正式参数修正为：

```text
Native layout/OCR/table cap = 8
V3 layout/OCR/table cap     = 8
```

双方同时修改，保持公平。cap=8 时 368-PDF：

```text
parent-bound theoretical fill       0.850
elastic within 24-PDF Arena fill    1.000
parent page-stage calls              1,040
elastic lower-bound calls              884
```

约 15% 的 page-stage call/fill 窗口足以检验 elastic；仍需先完成 48-PDF cap=8 pilot，再启动
368 full。V3 报告新增逐 Stage timeline summary，不能再仅使用全 DAG RPC/fill 归因。

## 9. 单卡结果作废与四卡 gate

此前 cap=4 的 48-PDF pilot，以及被中止的 cap=8 pilot，实际只配置：

```text
1 Layout actor × 0.5 GPU
1 Table actor  × 0.5 GPU
```

基本只使用一张 H20。它们只可作为单卡调试数据，**不得进入 4×H20 性能主表，也不能据此
判断 RayOrch pipeline/rebatching 的最终效果**。

四卡正式 pilot 必须在 timeline/GPU sample 中同时证明：

```text
4 张 H20 在 measured interval 内均有非零利用率；
V3 Layout/Table actor count 分别为 1/3；
Native 4 worker processes 分别绑定 GPU 0/1/2/3；
无其他 benchmark arm 并发抢卡；
parent/elastic 除 batch_scope 外参数一致。
```

四个 arm 仍应**彼此顺序运行**，但每个 arm 内部并行使用四卡。arm 间顺序运行与 arm 内四卡
并行不冲突。

### 9.1 2 Layout + 2 Table 四卡 pilot

48 PDF / 992 pages、cap=8：

```text
Native tuned 4-GPU        measured 122.466s  E2E 136.142s
V3 parent 2+2             measured 189.255s  E2E 196.533s
V3 elastic 2+2            measured 204.925s  E2E 212.464s
```

Correctness 均为 48/48 Markdown Jaccard 1.0、结构 exact。

Elastic 的机制指标确实改善：

```text
Layout RPC                144 → 130
OCR RPC                   144 → 127
Table RPC                 144 → 127
Layout pages/RPC          6.89 → 7.63
OCR/Table pages/RPC       6.89 → 7.81
```

但单次 wall 比 parent 慢约 8.1%。GPU/Stage 证据说明静态 2+2 分配不平衡：

```text
Layout worker busy sum       ~54–57s
Table worker busy sum       ~240–247s
Layout GPU mean util          ~1–2%
Table GPU mean util          ~17–20%
```

因此 2+2 不是四卡下合理的 per-stage resource allocation。下一轮固定改为 1 Layout + 3
Table；parent/elastic 两臂同时修改，仍只以 `batch_scope` 为控制变量。需用
counterbalanced repeat 判断 8.1% 是否为稳定机制结果，而不是单轮抖动。

### 9.2 1 Layout + 3 Table balanced pilot

保持输入、cap=8、Arena 24×3、OCR/parse/reduce 配置不变，只把 V3 四卡分配改为 1+3。
两轮 parent/elastic 前后顺序相反：

| Repeat | Parent measured | Elastic measured | Paired speedup |
| --- | ---: | ---: | ---: |
| r1, parent first | 154.823s | 142.816s | 1.084× |
| r2, elastic first | 154.825s | 146.957s | 1.054× |

```text
paired speedup median      1.069×
parent wall range          0.002s
elastic wall range         4.141s
correctness                48/48 Jaccard=1.0，structure exact
```

Packing 在两轮中稳定：

```text
Parent each page stage     144 RPC, 6.889 pages/RPC
Elastic each page stage    125 RPC, 7.936 pages/RPC
all-stage fill             0.8533 → ~0.979
```

两种执行顺序均显示 elastic 正收益，且 parent 极稳定，因此该配置通过 368 full pilot entry
gate。正式参数冻结为：

```text
4×H20
Layout                    1×GPU
Table/assemble            3×GPU
OCR                       4 CPU actors
Parse                     4 CPU actors
Reduce                    4 CPU actors
stage batch cap           8
microbatch                24 PDFs/Arena
max inflight Arenas       3
batch wait                2ms
```

48-PDF 下 Native tuned 4-GPU measured 为 `122.466s`，仍快于 V3；368 full 同时回答：

1. elastic 相对 parent 的机制收益是否随更大动态供给扩大；
2. V3 与 Docling 4-process multi-document baseline 的总吞吐差距；
3. V3 的优势是否只来自 rebatching，还是被 immutable stage transport/CPU OCR 抵消。

## 10. 368-PDF 四卡首轮 full pilot

固定 1 Layout + 3 Table、cap=8、Arena 24×3 后，首轮：

| Arm | Startup | Measured | E2E |
| --- | ---: | ---: | ---: |
| Native tuned 4-process | 10.917s | 827.069s | 856.044s |
| V3 parent-bound | 8.215s | 983.227s | 991.442s |
| V3 elastic | 8.450s | 940.341s | 948.791s |

Elastic 相对 parent：

```text
measured speedup       1.0456×
E2E speedup            1.0450×
```

机制证据是精确的 cap-8 packing：

```text
                       Parent             Elastic
Layout RPC             1,040              884
OCR RPC                1,040              884
Table RPC              1,040              884
pages/RPC              6.8                8.0
full 8-page batches    720                884
```

这说明在 9–53 pages 的动态 fan-out 下，elastic 已把每个 24-PDF Arena 的 page stages
完全装满。首轮净收益约 4.5%，低于 48-PDF 的约 6.9%，主要因为：

- PDF parse span 约 806–853s；
- CPU OCR busy sum 约 2,097–2,166s；
- GPU Layout/Table 并非唯一瓶颈；
- immutable DTO / per-stage transport 成本仍存在。

Native tuned 首轮仍比 V3 elastic 快：

```text
V3 elastic measured / Native measured     1.137×
V3 elastic E2E / Native E2E               1.108×
```

因此当前结论不是“V3 已胜过 Docling native”，而是：

> 在相同 V3 四卡资源配置内，cross-parent rebatching 带来可观测的约 4.5% 首轮净收益；
> 但 V3 split-stage adapter 的 parse/OCR/transport 开销仍使总吞吐落后于四进程
> Docling tuned baseline。

Correctness：

```text
Markdown token Jaccard median      1.0
minimum                            0.99968354
```

结构 exact 相对 Native 为 parent `340/368`、elastic `352/368`。差异集中在 layout
batch-peer 引起的 text/table/picture count 微小变化；Markdown 几乎完全一致。正式报告需列出
非 exact 文档的差值分布，不能仅写“strict exact”。

首轮不足以消除系统抖动；必须至少完成 elastic-first 的 repeat2，再判断约 4.5% 是否稳定。

### 10.1 前两轮与污染审计

前两轮 paired 方向相反：

```text
r1 parent-first    983.227 / 940.341 = elastic 1.0456×
r2 elastic-first   923.198 / 965.754 = elastic 0.9559×
paired median                           1.0008×
```

两种模式的 packing 指标都高度稳定：

```text
parent all-stage RPC       3,855–3,856
elastic all-stage RPC      3,335–3,337
parent fill                ~0.8423
elastic fill               ~0.988
```

但 wall 波动大于机制差值，因此两轮不足以下结论。

r3 parent 在运行中遭遇另一个外部 4×H20 MinerU 作业并发启动。其 NVML 特征从干净运行的：

```text
1.5–2.4GB/GPU，mean util 4–14%
```

变为：

```text
~84GB/GPU，mean util 40–47%
```

r3 elastic 也在同一污染窗口启动，随后已主动终止。r3 parent/elastic 均标记为 invalid，不纳入
CV、median 或 speedup。正式 runner 必须在每个 arm 前检查：

```text
GPU memory < 1GB
无其他 Ray/MinerU benchmark process
```

并在 arm 后根据 GPU samples 检测异常高显存/利用率，不能只靠开始前一次 `nvidia-smi`。

### 10.2 四组 2×2 balanced 干净结果

剔除所有 GPU peak memory 明显异常的污染 runs 后，保留两组 elastic-first、两组
parent-first：

| Pair | Order | Parent measured | Elastic measured | Parent / Elastic |
| --- | --- | ---: | ---: | ---: |
| r2 | elastic first | 923.198s | 965.754s | 0.956× |
| r3 clean | parent first | 845.112s | 955.337s | 0.885× |
| r4 | elastic first | 915.020s | 933.517s | 0.980× |
| r5 | parent first | 846.146s | 909.886s | 0.930× |

```text
elastic wins                  0 / 4
paired speedup median         0.943×
parent median                 880.583s
elastic median                944.427s
median-wall ratio             0.932×
parent CV                     4.82%
elastic CV                    2.63%
```

所以在该配置上，elastic **稳定慢约 6–7%**；首轮看到的正收益来自系统污染/顺序波动，不能
进入论文。

另一方面，packing 改善在四轮中完全稳定：

```text
                         Parent              Elastic
page-stage RPC           1,040               884–885
pages/RPC                6.8                 ~7.99
all-stage RPC            ~3,855              ~3,340
all-stage fill           ~0.842              ~0.987
```

这形成一个清晰负结果：

> logical-grain elastic rebatching 成功把物理 batch 填满，但 Docling 当前核心函数没有从
> 更大的 page batch 获得足够计算收益，额外同步/尾部延迟反而使端到端变慢。

源码归因：

- Layout 的 `predict_batch(images)` 是真实跨 page batch；
- RapidOCR 的 `__call__` 仍逐 page、逐 OCR rect 调 reader；
- TableStructureModel 的 `predict_tables` 仍逐 page、逐 table 调
  `multi_table_predict(..., [one_box])`；
- 因此把 OCR/Table 的逻辑 batch 从平均 6.8 提到 8 并没有对应的 kernel batching 收益。

四轮中位 Stage 数据也符合：

```text
Layout busy sum          397.1s → 387.0s   （小幅受益）
OCR busy sum            2027.0s → 2124.8s  （无 batch API，反而增加）
Table busy sum          1395.4s → 1423.2s
Layout span              860.6s → 928.6s
OCR span                 860.0s → 928.4s
Table span               873.8s → 937.4s
```

因此 Docling 结果不能用于声称“elastic 对所有 dynamic fan-out 都有收益”；它适合作为
模型 batchability 的边界案例。MinerU/vLLM 仍是 elastic 正收益的主证据。
