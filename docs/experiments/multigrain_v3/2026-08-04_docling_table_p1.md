# Docling Table P1：TableFormer encoder batching 实验

日期：2026-08-04
环境：Sol，4×NVIDIA H20，Docling 2.117.0，Ray 2.50.0，PyTorch 2.7.1

## 结论

Table P1 已从 `encoder_shadow` 推进为真实 `encoder_accelerated` 路径：跨当前
V3 物理 page batch 聚合 table crops，最多 16 个 job stack 后一次调用原
TableFormer encoder，再按原顺序复用 batch=1 decoder。默认仍为 `reference`，
加速路径必须显式开启。

正确性成立，而且不是近似一致：48 文档、992 页的三次 accelerated 输出与
reference 的 `documents.json.gz` 解压内容 SHA-256 全部为：

```text
1bc6d39fce951a425189d10b520bcbe195b897ba9c0b907757271b8b8daff9a8
```

batch 合并也已由 actor audit 直接确认：512 个 table jobs 合并为 108 个 encoder
batches，平均 4.74 jobs/batch；`batch_errors=0`、`table_fallbacks=0`。因此输出一致
不是异常后走 reference 的假象。

但 encoder-only batching 没有加速当前 workload。两次最小实现分别比 reference
慢 13.56% 和 17.26%；一次额外消除重复 resize/prepare 的实验仍慢 18.78%，因此该
额外 hook 已回退。当前最准确结论是：**TableFormer encoder 可以安全合批，但原
batch=1 decoder、逐 table Python 后处理和 cell matching 仍是主要串行边界，P1
不能作为 V3 elastic 的 wall-clock 加速证据。**

## 实现范围

- 新增 `table_batch_mode`：`reference`、`encoder_shadow`、
  `encoder_accelerated`。
- 新增 `table_batch_max_jobs`，本轮使用 16。
- 当前物理 Table UDF batch 内允许跨 document 聚合 encoder jobs，并按原输入索引
  恢复 document/page assembly。
- accelerated 正常路径不再执行 reference；仅 batch 执行异常时完整重建 pages 并
  回退 reference。
- 非 reference 模式要求 `table_actor_concurrency=1`，避免临时 encoder replay
  hook 在同 actor 的并发 RPC 间交错。
- CLI、matrix 和 benchmark summary 已接通 Table P1 参数及 actor audit 指标。
- `reference` 仍是默认值，负收益实验不会改变生产语义。

涉及代码：

- `rayorch/experimental/multigrain_v3/benchmark/document_docling/core_stages.py`
- `rayorch/experimental/multigrain_v3/benchmark/document_docling/core_v3.py`
- `rayorch/experimental/multigrain_v3/benchmark/document_docling/core_compare.py`
- `rayorch/experimental/multigrain_v3/benchmark/document_docling/core_matrix.py`
- `test/experimental/multigrain_v3/benchmark/test_docling_core_compare.py`
- `test/experimental/multigrain_v3/benchmark/test_docling_core_matrix.py`

定向测试：18 passed，3 warnings。

## 实验配置

```text
manifest: /tmp/mgv3-docling-368/manifest-48.json
documents/pages: 48 / 992
arm: v3_elastic
stage_batch_size: 8
parse/layout/ocr/table/reduce replicas: 4/1/4/3/4
layout/table GPU: 1 GPU per actor
OCR: CPU reference, num_threads=4
microbatch_size: 12
max_inflight_arenas: 1
table_batch_mode: encoder_accelerated
table_batch_max_jobs: 16
```

## 结果

| 路径 | measured | Table busy sum | 相对 reference | 输出哈希 |
|---|---:|---:|---:|---|
| reference elastic | 168.616s | 240.283s | baseline | exact |
| accelerated #1 | 191.483s | 304.146s | +13.56% | exact |
| accelerated #2 + audit | 197.724s | 301.343s | +17.26% | exact |
| preprocess replay 诊断 | 200.274s | 299.043s | +18.78% | exact |

preprocess replay 仅让 Table busy 从 301.34s 小幅变为 299.04s，未转化为 E2E 收益，
所以没有保留。两次最小 accelerated 的 measured 中位数为 194.604s，相对 reference
慢约 15.41%。

## 解释

安装版本的 `TableModel04_rs.predict` decoder 使用 scalar `.item()` 推进 tag
sequence，不能直接 batch>1。P1 只能合并 encoder，随后仍需逐 table 执行 decoder、
bbox/class 输出、token matching、row/column index 修正和 Docling response 组装。

本 workload 共 512 tables，而物理 Table stage 是 124–126 个 page RPC；即使 encoder
已减少到 108 次调用，GPU 平均利用率仍只有约 9%–13%。这说明扩大 encoder batch cap
不会自动消除主要串行段。

## Artifact

```text
/tmp/mgv3-docling-p1/smoke/reference
/tmp/mgv3-docling-table-p1/smoke/accelerated
/tmp/mgv3-docling-table-p1/audit-preopt-12
/tmp/mgv3-docling-table-p1/optimized-full
/tmp/mgv3-docling-table-p1/summarize_table_p1.py
```

注意：本轮发现 `core_compare --limit` 对 manifest 路径未生效；名为
`audit-preopt-12` 的 artifact 实际仍处理 48 文档、992 页，summary 中的 512 jobs
也证明了这一点。正式结果均按完整 workload 解释。

## 下一步

1. 不继续扩大 encoder-only batch；优先实现 TableFormer decoder 的 batch state
   machine，至少消除 tag decode 中的 batch=1 scalar 控制流。
2. 将 cell matching / response postprocess 与 GPU decoder 解耦，评估 CPU pool 或页内
   table 并行，避免三个 Table GPU actors 大部分时间等待串行 Python。
3. 修复 manifest 模式下 `--limit`，再做 1–2 个 table-heavy PDF 的快速开发回归；
   任何新 Table kernel 都继续以完整解压 JSON hash 作为最终 correctness gate。
