# Docling TableJob P2：模型内核复用的新 V3 workflow

日期：2026-08-04
环境：Sol，4×NVIDIA H20，Docling 2.117.0，Ray 2.50.0，PyTorch 2.7.1

## 结论

workflow 已从 page 内串行 Table stage 改成：

```text
Document -> Expand(Page) -> Layout -> OCR -> Postprocess
         -> Expand(TableJob) -> TableFormer core
         -> Reduce(Page) -> Reduce(Document)
```

只复用 Docling/RapidOCR 核心模型与原生后处理语义。Page/TableJob 的对象边界、跨
PDF elastic rebatching、空 child group、Page reduce 和 Document reduce 都由 V3
负责。全量 48 文档、992 页的所有保留实验输出与 reference 解压 JSON 字节级一致：

```text
SHA-256 1bc6d39fce951a425189d10b520bcbe195b897ba9c0b907757271b8b8daff9a8
bytes   3863934
```

Layout cluster 先筛选再渲染：992 页中 336 页含表，656 页无表并完全跳过 2x render；
含表页产生 512 个独立 TableJob。真实 active-mask decoder batch 已在 5-table 定向样本
逐表通过完整 `Table.model_dump(mode="json")` 相等检查，并在全量运行中零错误、零
fallback。

## 性能结果

| 配置 | measured | Table busy | Table RPC/batches | 输出 |
|---|---:|---:|---:|---|
| 旧 reference | 168.616s | 240.283s | page-bound | exact |
| encoder elastic | 108.686s | 121.273s | 63 | exact |
| encoder parent-bound | **100.673s** | 124.383s | 336 | exact |
| decoder elastic | 122.242s | **75.396s** | 63 | exact |
| decoder parent-bound | 111.878s | 102.571s | 336 | exact |
| decoder elastic + OCR P1 | 120.884s | 75.273s | 63 | exact |
| decoder parent + OCR P1 | 108.522s | 105.251s | 336 | exact |

真实 decoder batching 让 elastic 的 Table core busy 相对 parent-bound 从 102.571s
降到 75.396s，下降 26.5%，并把 Table GPU 峰值利用率推到 97%。这证明跨 PDF
TableJob rebatching 能转化成真实模型内核收益。

但当前 workload 的端到端 elastic 尚未胜过 parent-bound。组合实验中 OCR 处理
7,408 个 recognition jobs：elastic 合为 1,489 batches，parent-bound 为 1,536，均
零回退/零错误；elastic 仍为 120.884s，parent-bound 为 108.522s。原因是瓶颈已
移到约 100s 的 Layout/OCR/CPU postprocess 流水时窗，Table kernel 的节省被调度尾部
和 CPU/ORT 竞争抵消。不能把阶段级 26.5% 误报为 E2E elastic 胜利。

## TableCore 物理 batch=4

新增独立 `table_core_batch_size`，使 Layout/OCR/Postprocess/TableExpand 继续使用
batch 16，只将 TableCore 物理 RPC cap 改为 4。三次 batch4 与两次 batch16 全量
输出均保持相同解压 SHA-256。

| 配置 | runs | E2E median | Table busy median | Table span median | 平均活跃 actor median |
|---|---|---:|---:|---:|---:|
| TableCore batch16 | 120.884 / 119.460s | 120.172s | **75.420s** | 103.454s | 0.729 |
| TableCore batch4 | 109.333 / 118.720 / 120.557s | **118.720s** | 82.429s | **101.602s** | **0.810** |

batch4 将 Table RPC 从 63/64 增加到稳定的 156，其中 108 个为满 4-job RPC；Table
actor busy 增加约 9.3%，但时间窗内平均活跃 actor 提高约 11.1%，Table span 中位数
缩短 1.85s。结构假设成立：较小物理 batch 能恢复三卡并行和更及时的 Page 输出。

E2E 中位数仅改善 1.45s（约 1.2%），且单次结果在 109.3–120.6s 间波动。因此 batch4
是更合理的实验配置，但现有样本不足以宣称稳定端到端加速，也暂不替换默认继承值。
下一步需要保存 per-arena timeline，区分第四个重 arena 的 admission 时刻与 CPU/ORT
快慢态。

## 368 PDF：microbatch 24 / inflight arenas 4

为减弱 48-PDF 小样本的 arena admission 双态，使用完整 368 PDF / 7,072页运行两次：

```text
stage batch: 16
TableCore physical batch: 4
microbatch_size: 24 documents/arena
max_inflight_arenas: 4
OCR: recognition_accelerated
Table: decoder_accelerated
```

| Run | measured | pages/s | Page RPC avg | Table RPC avg | live blocks peak |
|---|---:|---:|---:|---:|---:|
| r1 | 811.148s | 8.72 | 15.86/16 | 3.30/4 | 591 |
| r2 | 760.571s | 9.30 | 15.75/16 | 3.29/4 | 556 |

两轮中位数 785.860s，相对已有 Native tuned 单跑 821.370s 快 4.3%，相对 Native
default 1006.062s 快 21.9%。两轮 OCR 60,624 jobs、Table 3,312 jobs 均为零错误、
零 fallback；无表的4,928/7,072页完全跳过Table render。

Page与Table物理batch形态高度稳定，但E2E仍相差50.577s，双样本CV为3.22%。Table
busy为414.66/421.44s，span却为790.97/738.55s；OCR busy为2032.83/1987.47s，
span为794.74/737.74s。计算量变化远小于stage span变化，剩余波动仍来自供给/排队
重叠，而不是decoder kernel。

`368 / 24 = 16` arenas；`max_inflight_arenas=4`只让96个PDF同时admission，后续仍有
12个arena分批进入。因此该配置显著提高fill，但没有从原理上消除arena admission波。
若要四个arena覆盖全量，需使用 `microbatch_size=92`；或保持24并将inflight提高到16，
但需先评估Object Store与arena状态内存。

两次V3输出页数/表格数/图片数逐文档完全一致，342/368文档完整dict一致；26个文档
的text计数与10个文档的Markdown不同。Native tuned相对Native default自身也有24个
text计数和12个Markdown差异，说明完整workload存在OCR/Layout并发非确定性，尚未
达到byte-exact回归标准。

## 设计与回归门禁

- `ExpandDoclingTableJobs` 传递 immutable DTO/object，不重跑 PDF 解析；无表页发出
  empty child group，使内层 Reduce 仍恢复原 Page。
- `DoclingTableCore` 只加载 TableFormer；decoder accelerated 路径失败时直接抛错，
  禁止静默 serial fallback 伪造加速结果。
- batched core 输出继续经过原生 TFPredictor matching/postprocess，再进入 PageAssemble。
- correctness gate 使用完整 `documents.json.gz` 解压内容哈希，而不只比较 table 数量或
  Markdown 相似度。

## Artifact

```text
/tmp/mgv3-docling-tablejob/tuned16-i3
/tmp/mgv3-docling-tablejob/parent16-i3
/tmp/mgv3-docling-tablejob/decoder-elastic16-i3
/tmp/mgv3-docling-tablejob/decoder-parent16-i3
/tmp/mgv3-docling-tablejob/decoder-elastic-balanced
/tmp/mgv3-docling-tablejob/combined-elastic-ocrp1-tablep2-r1
/tmp/mgv3-docling-tablejob/combined-parent-ocrp1-tablep2-r1
/tmp/mgv3-docling-tablejob/combined-elastic-tablecore-b16-r2
/tmp/mgv3-docling-tablejob/combined-elastic-tablecore-b4-r1
/tmp/mgv3-docling-tablejob/combined-elastic-tablecore-b4-r2
/tmp/mgv3-docling-tablejob/combined-elastic-tablecore-b4-r3
/tmp/mgv3-docling-tablejob/full368-elastic-mb24-i4-tablecore-b4-r1
/tmp/mgv3-docling-tablejob/full368-elastic-mb24-i4-tablecore-b4-r2
```

## 下一步

1. 不再增加 OCR replica；4 到 8 actors 没有缩短 span。把 OCR detection/crop 与
   recognition 拆成一等 jobs，按 shape bucket 跨 page/PDF rebatch。
2. 将 Table native matching/postprocess 从 GPU actor 解耦为 CPU stage，避免 GPU actor
   在 Python cell matching 时占槽。
3. 固定 ORT 线程与 CPU affinity 后做至少 4 组 counterbalanced parent/elastic pair；
   端到端结论只采用 paired median/geometric mean。
