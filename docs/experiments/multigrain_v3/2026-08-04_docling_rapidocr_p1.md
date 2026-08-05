# Docling RapidOCR P1：真实 recognition batching 实验

日期：2026-08-04
环境：Sol，4×GPU，Docling 2.117.0，Ray 2.50.0，PyTorch 2.7.1

## 结论

RapidOCR P1 已从 shadow 方案推进为可运行的真实加速路径：跨当前 V3 物理
page batch 聚合 OCR crops，按预处理后 tensor 的兼容键分桶，直接调用原
recognizer session 和 decoder。默认仍是 reference，P1 通过
`ocr_batch_mode=recognition_accelerated` 显式开启。

正确性通过：48 文档、992 页上，reference、shadow、accelerated，以及四组有效
parent/elastic pair 的 `documents.json.gz` 解压内容 SHA-256 全部为：

```text
1bc6d39fce951a425189d10b520bcbe195b897ba9c0b907757271b8b8daff9a8
```

V3 elastic 的调度收益稳定且确定：page-stage RPC 从 144 降到 124/125，整体 RPC
从 527/528 降到 467–470，batch fill 从约 0.8533 提升到 0.9777–0.9852；典型
elastic 运行达到 992 页 = 124×8 的全满 batch。

墙钟是正信号，但还不是强证明。四组平衡顺序 pair 中 elastic 赢 2/4；paired
speedup 中位数为 1.0267×，几何均值为 1.0318×。跨 arm 独立中位数为 parent
152.50s、elastic 140.04s，但 run-to-run span 方差较大，不能把 1.0889× 当作稳定
paired 收益。当前最准确表述是：**P1 证明 elastic 能把跨文档供给转成真实 OCR
recognizer batch 和更高 fill，并出现约 3% 的中心趋势收益，但 E2E 证据仍受
CPU/ORT 与流水线 span 波动影响。**

## 实现范围

- 新增 `ocr_batch_mode`：`reference`、`recognition_shadow`、
  `recognition_accelerated`。
- 新增 `ocr_recognition_batch_size`，本轮使用 8。
- 直接 stack 已归一化且 shape/dtype/语义兼容的 tensor，避免
  `TextRecognizer(img=list)` 再次计算 batch padding。
- 当前物理 OCR UDF batch 内允许跨 document 聚合 crops；按原输入顺序恢复 pages。
- accelerated 仅在候选 batch/rect 失败时延迟调用原 reader；shadow 保持严格
  reference/page gate。
- 非 reference 模式要求 `ocr_actor_concurrency=1`，避免 RapidOCR 可变状态重入。
- CLI 与 matrix 已接通两个 P1 参数；旧 Namespace 测试通过兼容默认值保持可用。

涉及代码：

- `rayorch/experimental/multigrain_v3/benchmark/document_docling/core_stages.py`
- `rayorch/experimental/multigrain_v3/benchmark/document_docling/core_v3.py`
- `rayorch/experimental/multigrain_v3/benchmark/document_docling/core_compare.py`
- `rayorch/experimental/multigrain_v3/benchmark/document_docling/core_matrix.py`
- `test/experimental/multigrain_v3/benchmark/test_docling_ocr_integration.py`

定向测试：9 passed，3 warnings。

## 实验配置

正式 pair 固定：

```text
manifest: /tmp/mgv3-docling-368/manifest-48.json
documents/pages: 48 / 992
stage_batch_size: 8
parse/layout/ocr/table/reduce replicas: 4/1/4/3/4
layout/table GPU: 1 GPU per actor
OCR: CPU, num_threads=4
microbatch_size: 24
max_inflight_arenas: 3
ocr_batch_mode: recognition_accelerated
ocr_recognition_batch_size: 8
only changed variable: batch_scope = parent_bound or elastic
```

## 正确性与 P1 单臂检查

在 `microbatch_size=12`、`max_inflight_arenas=1` 的全量检查中：

| 路径 | measured | OCR worker busy sum | 说明 |
|---|---:|---:|---|
| reference elastic | 168.62s | 236.70s | 正确性基线 |
| recognition shadow elastic | 271.33s | 673.91s | 双路径审计，预期更慢 |
| recognition accelerated elastic | 187.13s | 227.69s | OCR busy 降 3.8%，但本轮 E2E 变慢 |

这说明 P1 kernel 本身不是无条件加速：OCR busy 有下降，但流水线 span/重叠可能抵消。
因此正式判断使用更高压力的平衡 parent/elastic pair，而不是挑单臂最好值。

## 四组有效平衡 pair

两次运行顺序为 parent→elastic，两次为 elastic→parent。所有运行 GPU 峰值约
1.64–2.61GB，无外部占卡。

| Pair | 顺序 | parent | elastic | parent / elastic | 胜者 |
|---|---|---:|---:|---:|---|
| r1 | parent→elastic | 151.83s | 140.37s | 1.0817× | elastic |
| r3_clean | elastic→parent | 130.12s | 139.72s | 0.9313× | parent |
| r4 | parent→elastic | 153.17s | 132.27s | 1.1580× | elastic |
| r5 | elastic→parent | 153.98s | 158.48s | 0.9716× | parent |

汇总：

- elastic wins：2/4。
- paired speedup median：1.0267×。
- paired speedup geometric mean：1.0318×。
- parent mean / median：147.27s / 152.50s。
- elastic mean / median：142.71s / 140.04s。
- ratio of arm medians：1.0889×，仅作描述，不作为 paired 结论。

## 调度结构指标

Parent-bound 高度稳定地形成：

```text
每个 layout/OCR/table stage：144 RPC
batch histogram：112×8 + 16×4 + 16×2 = 992 pages
overall fill：约 0.8533
overall RPC：527/528
```

Elastic 形成：

```text
每个 layout/OCR/table stage：124/125 RPC
典型 histogram：124×8 = 992 pages
overall fill：0.9777–0.9852
overall RPC：467–470
```

相对 parent，elastic 的整体 RPC 约减少 11.4%，fill 相对提高约 15.4%。这些结构
指标跨运行复现，说明 V3 elastic 的跨 document packing 确实生效；不稳定的是它
转化成最终 wall-clock 的比例。

## 排除的污染运行

两次反序尝试在执行中被另一任务占用四卡，GPU 峰值约 95–97GB、utilization
达到 100%。对应 `/tmp/mgv3-docling-p1/pairs/r2` 和 `r2_clean` 不进入统计。

## Artifact

```text
/tmp/mgv3-docling-p1/smoke/reference
/tmp/mgv3-docling-p1/smoke/shadow
/tmp/mgv3-docling-p1/smoke/accelerated
/tmp/mgv3-docling-p1/pairs/r1
/tmp/mgv3-docling-p1/pairs/r3_clean
/tmp/mgv3-docling-p1/pairs/r4
/tmp/mgv3-docling-p1/pairs/r5
```

## 下一步

1. P1.1：把 OCR actor 的 `jobs/batches/rect_fallbacks/batch_errors` 汇总进 benchmark
   summary，并记录 recognizer batch-size histogram。
2. 固定 OCR actor CPU affinity/ORT threading，降低 worker span 的快慢态方差，再做
   batch size 2/4/8 小矩阵。
3. P2：推进 TableFormer 真批 decoder；当前 table 仍按 table/page 执行，是 elastic
   fill 难以稳定转化为 E2E 收益的主要剩余边界。
