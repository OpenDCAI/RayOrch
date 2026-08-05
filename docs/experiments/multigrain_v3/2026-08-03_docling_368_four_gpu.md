# Docling 368-PDF 四卡对比（2026-08-03）

## 1. 问题与输入

比较 `Native default / Native tuned / V3 parent-bound / V3 elastic`。

```text
368 PDFs / 7,072 pages / 1.10GB
23 个基础 PDF，各复制 16 次
pages/PDF: min 9, median 16, mean 19.22, p95 37, max 53
manifest SHA-256:
b9030601160f0873d4790de7494d6c9a3bc9a8486778daaa9556872ba08cbb53
```

该输入用于控制内容并制造稳定 fan-out skew，不代表 368 个独立文档的 diversity。

## 2. 四卡配置

共同使用 4×H20、Docling 2.117.0、batch cap 8、相同 manifest/models/options。

Native：

```text
4 independent processes × 1 GPU
1 DocumentConverter/process
page-count-balanced fixed shards
default: doc_batch_concurrency=1
tuned:   doc_batch_concurrency=4
global wall = four-process makespan
```

V3：

```text
1 Layout actor × 1 GPU
3 Table/assemble actors × 1 GPU
4 CPU OCR actors
4 parse / 4 reduce actors
microbatch_size=24 PDFs
max_inflight_arenas=3
parent/elastic only differ in batch_scope
```

arm 之间顺序运行；每个 arm 内同时使用四卡。V3 使用两组 parent-first、两组 elastic-first
的 balanced order。根据 NVML 排除与外部 4-GPU 作业重叠、显存峰值异常的 runs。

## 3. 性能结果

| Arm | Clean runs | Measured wall | Median | CV |
| --- | ---: | --- | ---: | ---: |
| Native default | 1 | 1006.062s | 1006.062s | N/A |
| Native tuned | 2 | 827.069 / 821.370s | **824.220s** | 0.49% |
| V3 parent-bound | 4 | 923.198 / 845.112 / 915.020 / 846.146s | **880.583s** | 4.82% |
| V3 elastic | 4 | 965.754 / 955.337 / 933.517 / 909.886s | **944.427s** | 2.63% |

```text
Native tuned vs default          1.221×
Native tuned vs V3 parent        1.068×
Native tuned vs V3 elastic       1.146×
V3 parent vs elastic median      1.073×
```

四个 paired `parent / elastic`：

```text
0.956× / 0.885× / 0.980× / 0.930×
paired median = 0.943×
```

即 elastic 4/4 次都慢，paired median 表示约 **6.1% slowdown**。不能挑选受污染或单轮
正结果进入论文。

## 4. Packing 与 Stage 归因

| 指标 | Parent | Elastic |
| --- | ---: | ---: |
| page-stage RPC | 1,040 | 884–885 |
| pages/RPC | 6.8 | ~7.99 |
| all-stage RPC | ~3,855 | ~3,340 |
| all-stage fill | ~0.842 | ~0.987 |

四轮中位 Stage 数据：

| Stage | Parent busy/span | Elastic busy/span |
| --- | ---: | ---: |
| Layout | 397.1 / 860.6s | 387.0 / 928.6s |
| OCR | 2027.0 / 860.0s | 2124.8 / 928.4s |
| Table/assemble | 1395.4 / 873.8s | 1423.2 / 937.4s |

源码边界：

- Layout 真实调用 `predict_batch(images)`；
- RapidOCR 仍逐 page、逐 OCR rect 调 reader；
- TableStructureModel 仍逐 page、逐 table 调单 box prediction；
- logical batch 填满不等于 OCR/Table kernel batch 变大。

因此：

> V3 elastic 正确实现了 cross-parent packing，但该 Docling workload 的主导 kernels
> 不具备相应 batch efficiency；同步和尾部延迟抵消了 Layout 的小幅收益。

## 5. Correctness

以 Native default 为参考，选取干净 V3 pair：

```text
Markdown token Jaccard median      1.0
minimum                            0.99968354
mean                               >0.99999
structure exact:
  Native tuned                     344 / 368
  V3 parent                        346 / 368
  V3 elastic                       338 / 368
```

没有缺失文档。结构计数差异来自 layout batch peers 的细微数值变化；正式论文报告分布，
不宣称 bitwise/structure 368/368 exact。

## 6. 论文结论

支持：

1. Docling `doc_batch_concurrency=4` 相对 default 提升约 `1.221×`；
2. V3 能把动态 9–53 page fan-out 的 batch 从平均 6.8 填到约 8；
3. 但 elastic 在该 workload 上稳定慢约 6%，V3 split-stage 也没有超过 Native tuned；
4. 主要差距来自 CPU OCR、逐 table 内核和 immutable Stage transport，而不是 driver commit。

不支持：

- “elastic 对所有 dynamic fan-out 都加速”；
- “RPC 更少或 fill 更高必然等于 wall 更快”；
- “V3 已全面快于 Docling”。

MinerU/vLLM 继续作为 batch-efficient model 上 elastic 正收益的主证据；Docling 作为边界
案例，说明系统需要识别 UDF/model 的 batchability。

## 7. 复现产物

```text
/tmp/mgv3-docling-368/manifest.json
/tmp/mgv3-docling-368/full-four-gpu/
```

每个 arm 保存 `summary.json / documents.json.gz / gpu_samples.jsonl / run.log`。详细参数
推导、污染 run 剔除规则和 pilot 过程见 `docling_368_experiment_plan.md`。
