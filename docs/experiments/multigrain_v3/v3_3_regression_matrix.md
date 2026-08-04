# Multigrain v3.3 现有实验回归矩阵

更新时间：2026-08-04

## 1. 结论

`experiment_matrix.md` 和 `paper_experiment_index.md` 中已经完成的 **V3-managed**
拓扑都能由 v3.3 的 Call/Port/Domain 模型表达。迁移时只替换 Pipeline 声明和 Executor，
业务 UDF、输入 manifest、模型、correctness gate、Ray Data/native baseline 均应复用。

这里的“可回归”分三层，不能混为一谈：

1. **Program 可表达**：compile test 已冻结拓扑、Domain 和 group 关系；
2. **执行链已验证**：相同 Worker ABI 已通过 local/Ray、多 Arena、failure/retry；
3. **真实 workload 已跑**：必须存在对应规模的输出与 correctness artifact。

只有第三层完成后，才能把历史性能数字替换为 v3.3 数字。

## 2. 当前状态

| Case | 历史状态 | v3.3 Program | v3.3 runner | 真实回归状态 |
| --- | --- | --- | --- | --- |
| MinerU elastic / parent-bound | 368 PDFs / 7,072 pages 已完成 | 已编译验证 | `benchmark/mineru.py` 已落地 | 4→48→368 gate 与双模式配对均完成 |
| MinerU Ray Data / native | 已完成 | 不属于 v3.3 Program | 原 runner 直接复用 | 无需迁移，只重新对比 |
| Ray Data full-value / reference-only | 已完成 | 不属于 v3.3 Program | 原 runner 直接复用 | 无需迁移 |
| Docling page-level | 已完成 smoke | 可表达 | 待适配 | 未运行 |
| Docling core-stage | 12 docs / 43 pages 已完成 | 已编译验证 | 待适配 | 未运行 |
| Docling 368 四臂矩阵 | 与本次工作并行执行的是 v3 作业 | 已编译验证 | 待适配 | v3 实跑不能标为 v3.3 已完成 |
| Video A/B | 已完成 smoke | 已编译验证 | 待适配 | 未运行 |
| Video C audio/frame | 已完成 smoke | 两个独立 child Domains 后 root merge 已验证 | 待适配 | 未运行 |
| Page→Region nested | 真实样本仍是计划 | 3 Domains、Filter、两级 Reduce 已验证 | dummy 已运行 | 真实 workload 未运行 |
| deterministic bad leaf | 主要是计划 | 已验证 | local/Ray failure path 已运行 | conformance 已完成 |

## 3. API 映射

### 3.1 MinerU / Docling page-level / Video A、B

```python
groups = render(parents)
children = F.expand(groups)
values = heavy(children)
value_groups = F.reduce(values)
return assemble(parents, value_groups)
```

这是标准 `1:M → M:1`，结构操作不创建 actor、RPC 或 Grain。

### 3.2 Docling core-stage 多分支

```python
pages = F.expand(parse(documents))
layout = layout_call(pages)
ocr = ocr_call(pages)
tables = table_call(pages)
merged = merge(layout, ocr, tables)
return finish(documents, F.reduce(merged))
```

同一 Page Domain 的多分支输入由 `EntityRef` 自然对齐，不需要 zip actor 或隐藏 join。

### 3.3 Video C

```python
audio = F.expand(audio_chunks(videos))
frames = F.expand(sample_frames(videos))

transcripts = F.reduce(whisper(audio))
visual = F.reduce(vision(frames))
return merge(videos, transcripts, visual)
```

Audio 和 Frame 是两个独立 child Domains，不允许按相同 ordinal 自动对齐；分别 Reduce
回 Video Domain 后才能 merge。child-level timestamp/window join 仍明确不在 v3.3 第一版
scope，不能通过隐式规则伪装。

### 3.4 Page→Region nested

```python
pages = F.expand(render_pages(documents))
regions = F.expand(detect_regions(pages))
selected = F.filter(regions, kind_mask(regions))
contents = ocr(selected)
contents_by_page = F.reduce(contents, members=selected)
contents_by_document = F.reduce(contents_by_page)
```

运行时已覆盖 20 层 unary、5 层 binary、intermediate empty group、bad leaf 和逐 parent
failure isolation，所以两级不是特判。

## 4. 已验证的基础证据

截至本文件更新时间：

- v3.3 全量：`44 passed`；
- 1,000-parent local dummy：batch cap `16/32/64`，parent-bound/elastic digest 完全一致；
- heavy RPC：`909 → 239/120/60`；
- Ray multi-Arena：`in_flight=1/2/3` 输出 digest 完全一致；
- persistent actor、actor crash/replacement、generation replay 已运行；
- batch 内单个 `RecordFailure` 只失败对应 Grain；multi-output 不会部分可见；
- MinerU v3.3 Program 固定为四个 Call、两个 Domains、四个 actor pools，业务 UDF 直接
  复用 v3 runner。

## 5. MinerU gate

真实回归必须严格按顺序推进：

```text
4 PDFs    correctness / actor / ObjectRef smoke
48 PDFs   feasibility / memory / throughput
368 PDFs  elastic 与 parent-bound paired regression
```

每一级同时检查：

- PDF/document/page count；
- Markdown 文档集合与逐文档 Jaccard；
- OCR RPC、grains/RPC、active Arena watermark；
- driver/GPU memory；
- 输出目录和 summary artifact 完整性。

如果 4 或 48 PDF gate 失败，不启动 368；不能用 compile test 或 dummy parity 代替真实
MinerU correctness。

## 6. MinerU 实跑结果（2026-08-03；2026-08-04 clean rerun）

所有 v3.3 runner 均直接复用 v3 的 Render、OCR、Metadata 和 Assemble UDF；模型、输入
PDF、4 张 H20、batch cap 和输出格式不变。actor-ready 屏障把模型初始化从 measured wall
中分离，确保与 v3 的计时口径一致。

| Gate / mode | docs / pages | startup | measured | pages/s | OCR RPC | OCR grains/RPC | Arena HWM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 elastic | 4 / 48 | 48.100s | 17.889s | 2.6832 | 4 | 12.00 | 2 |
| 48 elastic | 48 / 992 | 48.573s | 83.098s | 11.9377 | 20 | 49.60 | 2 |
| 368 elastic（clean） | 368 / 7,072 | 46.443s | **592.975s** | **11.9263** | 121 | 58.45 | 3 |
| 368 parent-bound（旧，性能待 clean paired） | 368 / 7,072 | 48.818s | 822.122s | 8.6021 | 368 | 19.22 | 3 |

### 6.1 性能结论与污染诊断

2026-08-04 在启动前确认 4 张 H20 均为 `0 MiB / 0%`、没有 compute process 或残留
raylet，并在运行期间确认只有一个 Ray session。clean 368 elastic 相比历史 v3 elastic
（587.781s，12.0317 pages/s）：

- measured wall 为 `592.975s`，只增加 `5.194s / 0.88%`；
- throughput 为 `11.9263 pages/s`，只降低 `0.88%`；
- 历史真实 MinerU elastic 的 run-to-run stdev 约为 1%，因此该差值属于正常波动；
- OCR packing 基本不变：`121 RPC / 58.45 pages per RPC`，历史 v3 为
  `120 RPC / 58.93 pages per RPC`。

2026-08-03 的 `635.947s / 11.1204 pages/s` **不是有效性能样本**。该次 measured
阶段约 85% 的时间与另一套独立 Ray session 重叠；另一 session 同时持有 4 个
`GPU: 1` 的 `RayStageWorker`。两个独立 cluster 都把相同 4 张物理 GPU 视为可用，
造成实际 GPU 竞争。进一步核对 actor stderr：

- 历史 v3 的两阶段 vLLM 累计时间为 `1481 GPU·s`；
- 被污染的 v3.3 为 `1659 GPU·s`；
- clean v3.3 恢复为 `1477 GPU·s`；
- clean v3.3 与 v3 的生成请求量分别为 `112,364 / 112,396`，工作量相同。

污染样本多出的 `178 GPU·s` 折合 4 卡约 `44.5s`，基本覆盖原先 `48.2s` 的 wall
差值。因此旧文档中的“measured 增加 8.2%、吞吐降低 7.6%”结论作废；它反映的是
benchmark 资源隔离失败，不是 v3.3 Call/Port/Domain、lineage 或 elastic scheduler
的性能回退。4/48 gate 仍可作为 correctness/feasibility gate，但其单次 wall 不用于
正式性能结论。

旧 parent-bound 运行仍提供有效的 packing 与 correctness 证据：

- OCR RPC `121 → 368`，平均批量 `58.45 → 19.22`；
- docs/pages、Arena 水位与 actor 数保持不变，证明 `batch_scope` 只改变 packing；
- 但其 wall-time 不再与污染 elastic 样本组成正式性能 pair。若论文需要精确的
  parent-bound/elastic speedup，必须在相同 clean preflight 下重新 paired 运行。

### 6.2 Correctness

| 对比 | matched | missing | Jaccard min | mean | median | ≥0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 elastic vs v3 | 4 | 0 | 0.9941 | 0.9973 | 0.9975 | 4/4 |
| 48 elastic vs v3 | 48 | 0 | 0.9895 | 0.9955 | 0.9961 | 48/48 |
| 368 clean elastic vs v3 | 368 | 0 | 0.9256 | 0.9926 | 0.9962 | 360/368 |
| 368 parent-bound vs v3 | 368 | 0 | 0.9292 | 0.9890 | 0.9931 | 362/368 |
| 368 parent-bound vs 2026-08-03 elastic | 368 | 0 | 0.9294 | 0.9893 | 0.9931 | 361/368 |

最低分集中在同一个字节级相同 PDF 的 32 个副本。旧 v3 对这些相同输入副本的内部
Jaccard 最低为 0.9259，clean v3.3 与 v3 的最低值为 0.9256，因此属于 VLM 自身的
非确定性范围，不是漏页、乱序或 lineage 错误。

### 6.3 Artifact

- clean 368 elastic：
  `/tmp/rayorch_v33_mineru_368_elastic_clean_20260804_r1`；
- clean 目录包含 `artifacts/summary.json`、`gpu_samples.jsonl`、
  `correctness_vs_v3.json`、`results.jsonl` 和完整 outputs；
- 污染样本仅保留用于诊断：
  `/tmp/rayorch_v33_mineru_368_elastic_20260803`；
- 旧 parent-bound：
  `/tmp/rayorch_v33_mineru_368_parent_bound_20260803`。
