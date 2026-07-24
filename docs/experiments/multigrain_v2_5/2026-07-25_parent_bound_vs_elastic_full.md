# V2.5 parent-bound 与 elastic 满载实验

日期：2026-07-24 至 2026-07-25

RayOrch commit：`9b91526c7ac2d610f5ff1c7ef83063ceb3f7c6f8`

Flash-MinerU commit：`7246a353554cee35674d7ef82f2c061db2f43003`

## 1. 目标

在真实满载下，只切换 V2.5 OCR 的物理组批范围：

```text
parent_bound：一个 RPC 不混合不同 PDF 的 pages
elastic：     一个 RPC 可以混合不同 PDF 的 pages
```

其他 Logical-Grain、actor、batch cap、Arena、transport、Reduce 和输入顺序
保持一致，以测量 cross-parent elastic rebatching 的净收益。

## 2. 配置

```text
PDFs                       368
pages                      7,072
input order                sorted
microbatch_size            24 PDFs
max_inflight_arenas         3
render replicas             4
OCR replicas                4
Reduce replicas             1
OCR batch cap              64 pages/RPC
max_batch_wait_ms          10
GPU memory utilization    0.9
object store               60 GiB
GPU                         4×NVIDIA H20
```

三次 paired repetitions，顺序为：

```text
rep 1：parent_bound → elastic
rep 2：elastic → parent_bound
rep 3：parent_bound → elastic
```

主指标为 readiness barrier 后的 `measured_wall_s`。

## 3. 满载结果

| Rep | Parent-bound | Elastic | Speedup | Wall reduction |
|---:|---:|---:|---:|---:|
| 1 | 813.704 s | 581.144 s | 1.400× | 28.58% |
| 2 | 806.834 s | 592.018 s | 1.363× | 26.62% |
| 3 | 817.803 s | 581.509 s | 1.406× | 28.89% |

汇总：

```text
Parent-bound median         813.704 s
Parent-bound mean           812.780 s
Parent-bound stdev            5.543 s

Elastic median              581.509 s
Elastic mean                584.890 s
Elastic stdev                 6.175 s

Paired speedup median          1.400×
Paired speedup range       1.363–1.406×
Paired wall reduction median  28.58%
```

吞吐中位数：

```text
Parent-bound                8.691 pages/s
Elastic                    12.162 pages/s
```

## 4. 与当前原生 Flash-MinerU 的收益分解

当前原生 Flash-MinerU 同一 368-PDF 输入：

```text
document-grain
PDF batch 8
inflight 4
measured wall              818.001 s
```

三层对照：

| 系统 | Grain | Cross-parent OCR packing | Measured wall |
|---|---|---|---:|
| 原生 Flash-MinerU | Document | 否 | 818.001 s |
| V2.5 parent-bound | Page | 否 | 813.704 s |
| V2.5 elastic | Page | 是 | 581.509 s |

结果解释：

```text
原生 → V2.5 parent-bound   1.005×，仅快 0.53%
parent-bound → elastic     paired median 1.400×
原生 → V2.5 elastic        1.407×，wall 降低 28.91%
```

在该满载配置下，V2.5 parent-bound 与原生 Flash-MinerU 性能基本一致。
显著提升主要出现在允许跨 parent 重新组 batch 之后，而不是仅来自
page-grain 表达或 V2.5 runtime。

## 5. OCR packing

中位数：

| Mode | OCR RPC | Pages/RPC | OCR fill |
|---|---:|---:|---:|
| parent_bound | 368 | 19.22 | 30.0% |
| elastic | 121 | 58.45 | 91.3% |

三次重复合计：

```text
Parent-bound:
  OCR RPC                   1,104
  每个 PDF 一个 RPC
  实际 size                  9–53 pages

Elastic:
  OCR RPC                     362
  满 64-page RPC              305
  其余为 bounded tails
```

因此 elastic 将 OCR RPC 数减少约 67%，同时把平均有效 batch 从
19.22 pages 提高到 58.45 pages。

## 6. 资源观测

中位数：

| 指标 | Parent-bound | Elastic |
|---|---:|---:|
| Live blocks across active Arenas | 125 | 99 |
| Worker RSS peak | 23.85 GB | 28.01 GB |
| NVML mean GPU utilization | 59.53% | 57.16% |

Elastic 吞吐更高，但 worker RSS peak 增加约 4.16 GB，说明更大的有效
OCR batch 以更高 worker memory 为代价。

NVML mean utilization 没有与吞吐单调对应。MinerU two-step UDF 包含 GPU
inference 和 CPU prepare/post-process；本实验以 wall time、实际 packing 和
输出正确性为主。

## 7. 输出正确性

三组 paired full outputs：

```text
Parent-bound outputs          368/368 × 3
Elastic outputs               368/368 × 3
Missing/extra                 0/0
Paired document comparisons   1,104
```

逻辑一致性：

```text
Token Jaccard median
  rep 1                       0.995502
  rep 2                       0.995995
  rep 3                       0.996113

Global minimum                0.933619
Jaccard ≥ 0.98                968/1,104
Jaccard ≥ 0.95              1,100/1,104
```

低分尾部与此前观察到的 vLLM table/OCR batching jitter 一致。所有文档和
页面均交付，没有 framework-level output loss 或 Reduce 对齐错误。

## 8. 结论

该实验固定了 page Logical Grains、batch cap、actors、Arena、transport、
Reduce 和输入顺序，仅切换是否允许跨 parent packing。

满载结果：

```text
Cross-parent elastic rebatching
→ OCR RPC 368 降至约 121
→ 平均 batch 19.22 提高到 58.45 pages
→ OCR fill 30.0% 提高到 91.3%
→ measured wall 中位数降低 28.58%
→ paired speedup 中位数 1.400×
```

因此在当前真实 MinerU workload 上，可以将 parent-bound 到 elastic 的
收益明确归因于 lineage-preserving cross-parent elastic rebatching。

## 9. 限制与下一步

- 当前仅为 sorted input；
- 数据由重复文档构成，需检查输入顺序和 fan-out skew；
- 三次 paired repetitions 支持稳定趋势，但不做强统计显著性声明；
- 尚未测量 1/2/4 GPU scale-up；
- Elastic worker RSS 更高，需要在更大输入下继续观察 memory trend。

下一步优先级：

1. fixed-seed shuffled 和 short/long interleaved 输入；
2. synthetic fan-out skew 与 elastic speedup 曲线；
3. 1/2/4 GPU scale-up；
4. poison density 与 isolation hard-bound 实验。

## 10. 外部实验产物

复现命令：

```bash
python -m rayorch.experimental.multigrain_v2_5.benchmark.mineru_ablation run \
  --config docs/experiments/multigrain_v2_5/configs/mineru_parent_bound_vs_elastic_full.json \
  --output-root /path/to/experiment \
  --flash-repo /path/to/Flash-mineru \
  --model /path/to/MinerU2.5-2509-1.2B

python -m rayorch.experimental.multigrain_v2_5.benchmark.mineru_ablation summarize \
  --config docs/experiments/multigrain_v2_5/configs/mineru_parent_bound_vs_elastic_full.json \
  --output-root /path/to/experiment
```

根目录：

```text
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/experiments/v25_full_parent_bound_vs_elastic_bs64_20260724
```

汇总：

```text
results/raw.jsonl
results/runs.csv
results/runs.json
results/summary.json
results/pairs.csv
results/pairs.json
results/pair_summary.json

comparisons/correctness_summary.json
comparisons/correctness_details.jsonl
```

每次运行的完整日志、输出和 timeline 位于：

```text
logs/<run>.log
outputs/<run>/
timelines/<run>/
```

实验目录总大小约 3.2 GB，不提交到 Git。
