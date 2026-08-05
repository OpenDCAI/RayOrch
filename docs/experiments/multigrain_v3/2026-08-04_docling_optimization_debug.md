# Docling V3：CPU、Arena 与四卡配比优化记录

日期：2026-08-04；环境：Sol，4×H20，368 PDF / 7,072页。

## 最终结论

瓶颈同时来自队列深度和资源配比，不是 Table batching 单点：

1. `max_pending_per_actor=4` 把大小不一的 PDF 预绑到 actor 深队列，形成 head-of-line。
2. V3 仅4个 Parse actor，而 Native tuned 四进程最多并发16个文档。
3. 1 Layout / 3 Table GPU 配比失衡；全量应使用2 / 2。
4. stage round-robin 能消除小样本 arena 饥饿，但后续 full368 消融未观察到吞吐收益，
   不能再把它列为最终性能主因。

最终保留配置：

```text
microbatch_size=24, max_inflight_arenas=4
stage_batch_size=16, table_core_batch_size=4
parse/layout/ocr/table/reduce replicas=16/2/6/2/3
layout/table num_gpus=1/1
max_pending_per_actor=1, ray_num_cpus=128
OCR recognition_accelerated, Table decoder_accelerated
```

两次全量为520.032s / 500.116s，中位数510.074s，13.865页/s，双样本
CV=2.76%。页数、表格数、图片数逐文档一致；OCR与Table均零错误、零fallback。

## 与基线对比

| 路径 | measured中位数/单次 | 相对最终V3 |
|---|---:|---:|
| Native default，四进程×concurrency 1 | 1006.062s | 最终V3吞吐1.97× |
| Native tuned，四进程×concurrency 4 | 824.220s | 1.62× |
| Flash-MinerU native | 818.001s | 1.60× |
| 旧Docling V3，1/3 GPU、Parse 4 | 785.860s | 1.54× |
| MinerU V3优化版 | 587.781s | 1.15× |
| MinerU V2.5 batch128 | 521.836s | 1.02× |
| **最终Docling V3** | **510.074s** | **基准** |
| MinerU V2.5最快单次batch256 | 491.738s | 最终V3慢3.7% |

MinerU与Docling模型/UDF不同，表中只能比较系统吞吐，不能归因模型优劣。Native tuned
的高concurrency相对default将1006.062s降至824.220s；V3最终仍快38.1%。

## 框架根因与修复

旧 `_submit()` 对每个 arena 执行 `while can_submit(stage)`，先填满当前arena的全部
actor pending credit。96-PDF、4×24 arena的原始timeline显示：配置层active arena峰值
为4，但任一stage同时活跃arena最多2个；arena首次worker时间为0.00/1.96/12.79/
33.76s。

修复后按stage跨active arenas round-robin，每轮每arena最多提交一个dispatch。配合
`pending=1`：

| 96-PDF配置 | measured | Parse span | Parse/OCR arena峰值 |
|---|---:|---:|---:|
| 旧arena-major，pending4 | 266.367s | 39.81s | 2 / 2 |
| 公平调度，pending4 | 246.371s | 82.79s | 4 / 4 |
| 公平调度，pending1 | **228.329s** | **43.28s** | 4 / 4 |

公平调度消除arena饥饿；pending1再避免每个actor静态积压多个轻重不一的PDF。

### Full368：round-robin 独立消融

2026-08-05 保持最终配置与 `pending=1` 不变，仅在进程内禁用 round-robin cursor，
使每个stage恢复arena-major credit消费；没有修改仓库源码。

| 策略 | measured runs | 中位数 | 双样本CV | RPC均值 | fill均值 | live blocks峰值均值 |
|---|---|---:|---:|---:|---:|---:|
| round-robin | 520.032 / 500.116s | **510.074s** | 2.76% | 4606.5 | 0.7668 | 1199.0 |
| arena-major | 491.539 / 541.525s | 516.532s | 6.84% | **4274.5** | **0.8331** | **597.5** |

arena-major中位数仅慢6.458s（1.27%），两组run区间重叠；现有样本不支持
round-robin提高full368吞吐。相反，arena-major因保留arena-local聚集性，将RPC减少
约7.2%、fill相对提高约8.6%、live blocks峰值减半。round-robin的双样本CV更低，但
样本不足以把稳定性差异作为强结论。

两组均处理60,624个OCR jobs和3,312个Table jobs，OCR/Table零错误、零fallback；
arena-major两轮相对round-robin r2均为页/表/图368/368一致，完整结构346/368一致，
漂移规模与已有并发非确定性相同。因此最终性能归因应放在`pending=1`、Parse并发和
Layout/Table资源配比；round-robin只保留公平性语义，不作为吞吐必需修改。

## 单变量结果

| 试验 | 结果 | 决策 |
|---|---|---|
| Parse 4→16（48 PDF） | 121.955→103.829s；Parse span 49.18→12.98s | 保留16 |
| OCR actor 4→8（2 arenas） | 103.829→110.581s | 供给不足时负优化 |
| OCR actor 4→6（4 arenas） | 228.329→213.814s | 保留6 |
| OCR actor 6→8 | 213.814→212.814s，busy增4%，峰值仅7 | 回退6 |
| OCR inner batch 6→12 | 调用1774→1227，E2E反慢4.9% | 回退6 |
| GPU Layout/Table 1/3→2/2（全量） | 704.151→510.074s中位数 | 保留2/2 |
| 4+4 actor、每actor 0.5 GPU（48 PDF） | Table busy 82.9→115.4 actor·s | 回退 |
| microbatch 24→92，均inflight4 | Parse span 600→101s，E2E 713→774s；live blocks 1254→3696 | 回退24 |
| reduce 3→4 | 96 PDF快7.2%，全量506.478s落在reduce3波动内且busy增大 | 保留3 |

`microbatch=92`说明单独消掉Load空泡并不保证E2E更快：四个超大arena加重共享
CPU/ready-queue尾部。最终24×4让admission自然节流，整体更快、内存峰值更低。

## 全量阶段证据

最终第二轮各stage worker span：Parse 444.31s、Layout 460.79s、OCR 462.63s、
Postprocess 482.77s、TableCore 484.64s、PageReduce 485.82s。各stage尾部已接近，
继续单独增加OCR、Table或Reduce副本没有明确收益。

最终两轮中，60,624个OCR jobs和3,312个Table jobs均零fallback；4,928/7,072页
无表并跳过Table render。两轮347/368完整document dict相等；页/表/图全部相等，
19个文档text计数、10个Markdown存在并发OCR/Layout漂移。Native tuned相对Native
default自身也有24个text和12个Markdown差异，因此这是现有模型并发非确定性，不是
Table batching结构回归；严格byte-exact全量门禁仍未通过。

## 代码与Artifact

- 公平调度：`rayorch/experimental/multigrain_v3/driver.py`
- Raw timeline导出：`core_compare --timeline-output PATH`
- 最终两轮：
  - `/tmp/mgv3-docling-tablejob/full368-fair-parse16-ocr6-pending1-layout2-table2-r1`
  - `/tmp/mgv3-docling-tablejob/full368-fair-parse16-ocr6-pending1-layout2-table2-r2`
- Arena-major + pending1消融：
  - `/tmp/mgv3-docling-tablejob/full368-arena-major-parse16-ocr6-pending1-layout2-table2-r1`
  - `/tmp/mgv3-docling-tablejob/full368-arena-major-parse16-ocr6-pending1-layout2-table2-r2`
- 4×92反例：`/tmp/mgv3-docling-tablejob/full368-fair-parse16-ocr6-pending1-mb92-r1`

新增公平性单测验证credit顺序为`arena0,1,2,0,1,2`，timeline单测验证原始arena与
worker时间字段可持久化。真实Ray pipeline集成测试通过。
