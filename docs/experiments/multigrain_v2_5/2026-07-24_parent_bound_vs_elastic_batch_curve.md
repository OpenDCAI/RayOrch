# V2.5 parent-bound 与 elastic 组批对照

日期：2026-07-24

RayOrch commit：`9b91526c7ac2d610f5ff1c7ef83063ceb3f7c6f8`

Flash-MinerU commit：`7246a353554cee35674d7ef82f2c061db2f43003`

## 1. 实验问题

原生 Flash-MinerU 与 V2.5 的对比同时改变了 document/page grain 和组批
方式，无法单独测量 cross-parent elastic rebatching 的收益。

本实验在同一 V2.5 Pipeline、Logical Grains、Ray transport、Reduce 和硬件上，
只切换：

```text
parent_bound：一个 OCR RPC 只包含同一 PDF/parent 的 pages
elastic：     一个 OCR RPC 可以混合不同 PDF/parent 的 pages
```

目标是回答：

> 在其他条件相同时，允许跨 parent 重新组 batch 能否减少 RPC fragmentation
> 并提升真实 vLLM pipeline 吞吐？

## 2. 固定配置

```text
输入顺序                   sorted
PDF 数量                   48
页面数量                   992
microbatch_size            24 PDFs
max_inflight_arenas         3
render replicas             4
OCR replicas                4
Reduce replicas             1
max_batch_wait_ms          10
GPU memory utilization    0.9
object store               60 GiB
GPU                         4×NVIDIA H20
Ray                         2.50.0
vLLM                        0.10.1.1
```

Pipeline：

```text
Source
→ Expand(PDF→pages)
→ Map(real MinerU vLLM OCR)
→ Reduce(ordered pages→document)
```

## 3. 实验矩阵与方法

### E1 sanity

```text
OCR batch cap              128
parent_bound / elastic     各 1 次
```

先验证两种模式确实形成不同物理 batch，且输出一致。

### E2 paired batch curve

```text
OCR batch cap              16 / 32 / 64 / 128
mode                       parent_bound / elastic
paired repetitions         3
总运行次数                 24
```

运行顺序交替：

```text
rep 1：parent_bound → elastic
rep 2：elastic → parent_bound
rep 3：按 batch 交错顺序
```

每次运行独立保存：

- stdout/stderr 日志；
- Markdown/layout 输出；
- benchmark JSONL；
- Ray timeline；
- V2.5 Dispatch timeline；
- GPU utilization/RSS samples。

主指标使用 readiness barrier 之后的 `measured_wall_s`。

## 4. E1 sanity 结果

配置：

```text
batch cap = 128
```

| Mode | Measured wall | Pages/s | OCR RPC | OCR RPC sizes |
|---|---:|---:|---:|---|
| parent_bound | 92.845 s | 10.684 | 48 | 16×12，16×16，16×34 |
| elastic | 78.394 s | 12.654 | 13 | 7×128，其余 bounded tails |

```text
elastic speedup              1.184×
wall reduction               15.56%
```

sanity 明确验证：

- parent-bound 恰好每个 PDF 一个 OCR RPC；
- elastic 将多个 PDF 的 pages 合并为 128-page RPC；
- 两种模式均输出 48/48 documents；
- 48/48 文档 `token Jaccard ≥ 0.98`。

因此 `batch_scope` 的物理行为符合设计，继续执行 E2。

## 5. E2 paired batch curve

下表使用三次重复的中位数；speedup 由每次 paired wall time 相除后取中位数。

| OCR batch cap | Parent wall | Elastic wall | Elastic speedup | Speedup range | Parent pages/s | Elastic pages/s |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 117.207 s | 104.095 s | 1.126× | 1.119–1.150× | 8.464 | 9.530 |
| 32 | 106.916 s | 86.611 s | 1.242× | 1.234–1.250× | 9.278 | 11.454 |
| 64 | 93.338 s | 78.471 s | 1.191× | 1.187–1.202× | 10.628 | 12.642 |
| 128 | 93.162 s | 79.860 s | 1.169× | 1.123–1.179× | 10.648 | 12.422 |

四个 batch cap 下，elastic 三次 paired run 均快于 parent-bound。

最大中位数收益出现在 batch cap 32：

```text
speedup                     1.242×
wall reduction              19.50%
```

## 6. 实际 OCR packing

| Batch cap | Parent OCR RPC | Elastic OCR RPC | Parent pages/RPC | Elastic pages/RPC |
|---:|---:|---:|---:|---:|
| 16 | 80 | 63 | 12.40 | 15.75 |
| 32 | 64 | 34 | 15.50 | 29.18 |
| 64 | 48 | 20 | 20.67 | 49.60 |
| 128 | 48 | 13 | 20.67 | 76.31 |

Parent-bound 在 cap≥64 后不再获得更大的实际 batch：

```text
每个 parent 的 fan-out     12 / 16 / 34 pages
平均实际 batch             20.67 pages/RPC
OCR RPC                    固定为 48
```

Elastic 则可以继续跨 parent 填充：

```text
cap 64                     平均 49.60 pages/RPC
cap 128                    平均 76.31 pages/RPC
```

三次重复合计的主要 histogram：

```text
parent_bound, cap 128:
  48×12, 48×16, 48×34

elastic, cap 128:
  20×128，另有 120/32/24/16/12/4 tails
```

这直接验证：

> Parent-bound 的有效 GPU batch 受单个 parent cardinality 限制；elastic
> 可以跨 parent 填充更大的 OCR batch。

## 7. Batch cap 曲线解释

### Parent-bound

```text
cap 16 → 32 → 64：
117.2 s → 106.9 s → 93.3 s
```

cap 16/32 会切分 34-page parent；cap 64 后已能容纳所有单 parent pages，
继续增大到 128 不再改善：

```text
cap 64                      93.338 s
cap 128                     93.162 s
```

### Elastic

```text
cap 16 → 32 → 64：
104.1 s → 86.6 s → 78.5 s
```

cap 128 没有继续稳定改善：

```text
cap 128 median              79.860 s
```

说明本 workload 在 4×H20 上的有效 sweet spot 约为 64–128 pages/RPC；
elastic 的贡献不是“batch 越大越好”，而是解除 parent 边界，使 runtime
可以形成适合 GPU UDF 的 batch。

## 8. 控制面与资源观测

中位数。OCR fill 按 `实际 OCR pages/RPC ÷ configured OCR batch cap`
计算；tail fraction 是 benchmark 的全 node aggregate 指标。

| Batch cap | Mode | OCR fill | Aggregate tail RPC fraction | Live blocks across Arenas | Worker RSS peak |
|---:|---|---:|---:|---:|---:|
| 16 | parent_bound | 77.5% | 36.1% | 92 | 14.05 GB |
| 16 | elastic | 98.4% | 24.3% | 91 | 13.03 GB |
| 32 | parent_bound | 48.4% | 50.4% | 87 | 14.55 GB |
| 32 | elastic | 91.2% | 23.6% | 72 | 13.03 GB |
| 64 | parent_bound | 32.3% | 60.0% | 86 | 13.16 GB |
| 64 | elastic | 77.5% | 20.9% | 67 | 14.05 GB |
| 128 | parent_bound | 16.1% | 59.7% | 85 | 13.54 GB |
| 128 | elastic | 59.6% | 14.7% | 66 | 14.18 GB |

Parent-bound 的配置 batch cap 增大后，fill ratio 持续下降，因为实际 batch
无法越过 34-page parent 上限。Elastic 同时减少 OCR RPC 和 active-Arena
live blocks。

NVML 平均 GPU utilization 没有与吞吐严格单调对应。MinerU 的
`batch_two_step_extract()` 内部包含 layout inference、CPU prepare、
content inference 和 post-process，因此 actor busy/GPU 瞬时利用率不能单独
作为吞吐判断；本文以 wall、RPC packing 和输出正确性为主。

## 9. 输出正确性

12 组 paired comparisons：

```text
parent outputs                 48/48
elastic outputs                48/48
missing/extra                   0/0
文档对比总数                   576
token Jaccard ≥ 0.98           576/576
global minimum                 0.986417
median of per-run medians      0.996340
```

不同 batching 引起的非字节级差异处于既有 vLLM jitter 范围，没有发现
framework-level output loss 或 Reduce 顺序错误。

## 10. 结论

在相同 page Logical Grains、actors、batch cap、Arena、transport、Reduce 和
输入顺序下，只允许跨 parent 组 batch，即获得：

```text
1.126×–1.242× paired median speedup
11.19%–19.50% median wall reduction
```

机制证据为：

```text
parent_bound:
  实际 batch 被 12/16/34-page parent fan-out 限制

elastic:
  跨 parent 形成 32/64/128-page RPC
  OCR RPC 数下降
  batch fill 上升
  wall time 下降
```

因此当前实验可以把 V2.5 相对 parent-bound 的这部分收益明确归因于
cross-parent elastic rebatching，而不是 page-grain、persistent actor 或
单纯调整配置 batch cap。

## 11. 实验限制与下一步

- 当前数据仅有 48 PDFs，且由三类重复文档组成；
- 每个配置三次重复，足以观察稳定趋势，但不足以做强统计显著性声明；
- sorted input 可能降低或改变 parent skew；
- 需要继续测试 fixed-seed shuffled 和 short/long interleaved 输入。

368-PDF 满载 paired comparison 已完成，结果见：

`docs/experiments/multigrain_v2_5/2026-07-25_parent_bound_vs_elastic_full.md`。

后续：

1. 增加 sorted/shuffled/interleaved 顺序 ablation；
2. 运行 synthetic fan-out skew matrix，建立 skew 与 speedup 的因果曲线；
3. 在 1/2/4 GPU 下检查 scale-up 趋势。

## 12. 外部实验产物

复现命令：

```bash
python -m rayorch.experimental.multigrain_v2_5.benchmark.mineru_ablation run \
  --config docs/experiments/multigrain_v2_5/configs/mineru_parent_bound_vs_elastic_48.json \
  --output-root /path/to/experiment \
  --flash-repo /path/to/Flash-mineru \
  --model /path/to/MinerU2.5-2509-1.2B

python -m rayorch.experimental.multigrain_v2_5.benchmark.mineru_ablation summarize \
  --config docs/experiments/multigrain_v2_5/configs/mineru_parent_bound_vs_elastic_48.json \
  --output-root /path/to/experiment
```

实验根目录：

```text
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/experiments/v25_parent_bound_vs_elastic_20260724
```

主要文件：

```text
environment.txt
config.json
run_order.log
e2_schedule.tsv

results/e2_raw.jsonl
results/e2_runs.csv
results/e2_runs.json
results/e2_summary.json
results/e2_pairs.csv
results/e2_pairs.json
results/e2_pair_summary.json

comparisons/e2_correctness.json
comparisons/sanity_parent_vs_elastic_summary.json

logs/<run>.log
outputs/<run>/
timelines/<run>/{ray_timeline.json,dispatch_timeline.jsonl,gpu_samples.jsonl}
```

大体积日志、模型输出和 timeline 不提交到 Git。
