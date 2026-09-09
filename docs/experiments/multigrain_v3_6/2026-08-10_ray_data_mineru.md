# V3.6 × Ray Data MinerU 受控实验

日期：2026-08-10
性质：单机论文 feasibility；Ray Data 两臂完成三次交错重复，V3.6 对比仍是单次结果。

## 1. 结论

在相同 Ray 2.50.0、同一组 368 PDFs / 7,072 pages、4×H20、同一 MinerU UDF 和
batch cap 64 下：

| Arm | Run wall（mean ± sample SD） | Hash-regroup bytes | OCR batch calls / fill |
| --- | ---: | ---: | ---: |
| Ray Data natural `full_value` | 976.545 ± 53.015s | 79.892GB | 196 / 0.5638 |
| Ray Data expert `reference_only` | 837.839 ± 11.529s | 2.022MB | 196 / 0.5638 |
| V3.6 elastic（单次，仅作 feasibility） | 617.222s | 不执行通用 full-value groupby | 121 / 0.9132 |

Ray Data 的受控干预只改变进入同一个 hash groupby 的值表示。Reference-only 平均缩短
138.706s（14.2%；arm-mean ratio 1.166×），配对差样本标准差 41.548s，n=3 的 t 区间为
`[35.494s, 241.918s]`。因此在该 workload 上，完整 page/content 参与通用 hash regroup
具有可重复的端到端成本。

最直白的技术解释是：

- full-value 在最后 OCR 后平均等 200.108s，第一份 document 才能开始 assemble；
- reference-only 的对应 handoff 只有 0.207s；
- 但外挂 owner protocol 让 assemble span 从 76.306s 增至 130.013s；
- 两项相抵后，terminal tail 仍从 277.765s 降至 130.292s，净省约 147.5s；
- reference 的最后 OCR 反而平均晚 8.767s，所以收益不是模型变快。

这也解释了 V3.6 的设计价值，但不能把剩余差距归给单一 feature：V3.6 同时有更完整的
cross-parent packing、运行时原生 ownership、persistent actors 和 per-parent online Reduce。

## 2. 控制变量

两个 Ray Data arm 共用同一个 runner 和物理算子链：

```text
from_items
  → flat_map(RenderPdf)
  → map_batches(OcrPages, 4 GPU actors, batch cap 64)
  → HashShuffle(parent_id, 4 partitions)
  → map_groups(AssemblePdf, 4 actors)
```

唯一干预点：

| Arm | 进入 HashShuffle 的行 | 完整 page/content |
| --- | --- | --- |
| `full_value` | parent/ordinal + `image_rgb` + content | Dataset row |
| `reference_only` | parent/ordinal + store/block/row selector | 4 个 owner actors 的 coarse ObjectRef blocks |

固定不变量：

- 输入为相同排序后的 368 个 PDF，DPI=200，严格得到 368 documents / 7,072 pages；
- `MinerUPdfToPages`、`MinerUVlmOcrPage`、`MinerUAssembleDoc` 完全复用同一实现；
- 32 CPUs、4×NVIDIA H20、100 GiB object store、4 render/4 OCR/4 assemble actors；
- model、GPU memory utilization、batch cap 64、parent key、ordinal 和输出语义相同；
- run wall 从 Dataset materialization 开始，包含 actor/model startup、OCR、shuffle、完整
  assemble 和 collect，不含 `ray.init()` / `ray.shutdown()`；
- 顺序为 Full R1 → Ref R1 → Full R2 → Ref R2 → Full R3 → Ref R3；
- 每轮独立启动/关闭 Ray，开始前 GPU 空闲，结束后核验无 Ray instance 和 GPU process。

Reference-only 是因果控制，不是 Ray Data 推荐 API，也不是一个免费的优化：同步
`store`、`acquire/get/release` 和引用生命周期全部计入 wall time。

## 3. 前置门禁与调参

4-PDF CPU smoke 证明两臂物理计划相同、4 documents / 48 pages 及 ordinal 完全一致。
即使没有真实 OCR，full/reference shuffle-map output 已分别为 538,575,540B / 14,004B。

48-PDF real-model gate 只扫描 source blocks，其他变量不动：

| Source blocks | Run wall | OCR batch calls | Fill | Last OCR | Post-OCR tail |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | **160.146s** | 27 | 0.5741 | 116.661s | 43.485s |
| 8 | 161.073s | 27 | 0.5741 | 118.237s | 42.836s |
| 16 | 164.441s | 29 | 0.5345 | 120.689s | 43.752s |

P4/P8 差 0.58%，选择更简单的 P4；P16 已出现 batch fragmentation。相同 P4 的
reference gate 为 152.332s，shuffle 从 11.136GB 降到 272KB，结构输出完全一致。

## 4. 368-PDF 重复结果

### 4.1 每轮原始结果

| Repeat | Arm | Wall | Last OCR | Post-OCR tail | Shuffle-map bytes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Full | 973.311s | 673.311s | 300.000s | 79,891,901,972 |
| 1 | Ref | 835.910s | 705.328s | 130.582s | 2,021,526 |
| 2 | Full | 925.221s | 659.803s | 265.418s | 79,892,058,655 |
| 2 | Ref | 827.396s | 696.009s | 131.386s | 2,021,526 |
| 3 | Full | 1031.102s | 763.224s | 267.879s | 79,892,225,640 |
| 3 | Ref | 850.211s | 721.303s | 128.908s | 2,021,526 |

| Metric | Full mean ± SD | Ref mean ± SD | Ref intervention |
| --- | ---: | ---: | ---: |
| Run wall | 976.545 ± 53.015s | 837.839 ± 11.529s | −138.706s |
| Last OCR | 698.779 ± 56.217s | 707.547 ± 12.792s | +8.767s |
| Post-OCR tail | 277.765 ± 19.295s | 130.292 ± 1.265s | −147.473s |
| Shuffle-map output | 79.892GB | 2.022MB | 39,521× smaller |

三组 paired wall differences 为 137.401/97.825/180.891s，全部同向。Full 的较大方差
同时来自模型主段和 terminal finalize；shuffle bytes 本身在三轮几乎不变。

这里的 196 是 actor 内真实模型 UDF batch calls，不是 Ray RPC 数。Ray Data stats 显示
OCR operator 每轮执行 99 个远程 actor tasks；一个 task 会连续调用 UDF，典型形成
`64 + 8` 两个 model batches。V3.6 的 121 则同时是 Worker RPC 和 model batch call。
因此可比较的是 packing/fill，不能把 `196 vs 121` 直接写成 RPC 开销。

### 4.2 Ray operator 证据

Raw Ray Data stats 与时间线相互吻合：

| Operator evidence | Full | Reference-only |
| --- | ---: | ---: |
| Shuffle finalize remote wall（3-run mean） | 201.557s/partition | 0.014s/partition |
| Last OCR → first Assemble | 200.108 ± 27.057s | 0.207 ± 0.014s |
| Assemble remote wall（3-run mean） | 51.150s/task | 122.323s/task |
| First → last Assemble span | 76.306 ± 11.803s | 130.013 ± 1.278s |

Full 每个 finalize partition 约 19–22GB，只有 finalize 后 group 才交给 assemble。
Reference manifest 的 finalize 只有约 0.5MB/partition，因此最后 OCR 后几乎立即 assemble；
代价是 assemble 必须恢复 196 个 coarse payload blocks，故自身更慢。不能把 Ray stats 中
`Shuffle executed in ...` 当作隔离的纯 shuffle 时间，因为 streaming operator wall 与上游
重叠；上表使用 suboperator remote wall 和显式 UDF 边界。

### 4.3 两臂都确实使用四卡

在“四个 OCR actor 均 ready → 最后 OCR batch”窗口，1s GPU 采样的三轮均值为：

| Metric | Full | Reference-only |
| --- | ---: | ---: |
| Mean GPU utilization | 55.90% | 53.35% |
| Mean active GPUs（util ≥10%） | 3.286 / 4 | 3.101 / 4 |
| All four active samples | 52.45% | 41.40% |
| All four idle samples | 1.13% | 1.23% |

Reference 的同步 publish 降低了 GPU 活跃度，进一步说明它不是人为偏快的控制臂。

### 4.4 Correctness

六轮都严格得到同样的 368 document identities、每个 parent 的连续 ordinal 和总计 7,072
pages；owner stores 结束时 `live_blocks/live_rows/live_payload_bytes` 全部为零。Full/Ref
Markdown token Jaccard：

| Repeat | Min | Mean | Median | ≥0.95 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9300 | 0.9939 | 0.9979 | 363/368 |
| 2 | 0.9281 | 0.9938 | 0.9969 | 366/368 |
| 3 | 0.7488 | 0.9911 | 0.9949 | 364/368 |

R3 的低分 outlier 与既有 V3.6/Daft/V3.5 自漂移中出现的复制文档 outlier 同量级；这是
模型调度相关非确定性，不能归因于 Ray Data representation。

## 5. 与 V3.6 的写作口径

同日 V3.6 单次完整结果为 617.222s。相对 Ray Data 三轮均值：

- natural full-value：`976.545 / 617.222 = 1.582×`，V3.6 时间少 36.8%；
- expert reference-only：`837.839 / 617.222 = 1.357×`，V3.6 时间少 26.3%；
- expert control 只收回 natural gap 的 138.706/359.323 = 38.6%，仍有 220.617s residual。

Residual 不能直接命名为一个 feature。已观测的复合差异包括：Ray Data 为 196 model
batch calls、fill 0.5638（99 remote actor tasks），而 V3.6 为 121 Worker RPC/model calls、
fill 0.9132；expert control 还有同步 owner RPC；Ray Data
groupby 在所有 map 输入 terminal 后才交付 partition，而 V3.6 按 parent completion 在线
Reduce。论文应分别做 batching scope、online Reduce 和 ownership ablation。

可以写的 takeaway：

> 在同一 Ray substrate 上，Ray Data 的自然 `flat_map → map_batches → groupby` 正确但会
> 把约 79.9GB page/content 带入 hash regroup。只把 groupby 输入换成 manifest，就把
> terminal handoff 从约 200s 降到 0.2s，并使端到端平均快 138.7s；但用户态引用恢复又
> 增加约 54s assemble。这说明 V3.6 的优势不是“少一次 RPC”，而是把 lineage、引用
> ownership、跨 parent batching 和 online ordered Reduce 作为一个运行时协议共同实现。

不能写：

- “Ray Data 不能表达 1:M/M:1”——它能自然表达并保持正确性；
- “79.9GB 都经过网络”——本轮单节点，stats 是 shuffle block bytes，不是 NIC bytes；
- “reference-only 等于 V3.6”——它是用户态专家控制，恢复和失败语义均不同；
- “V3.6 结果已有统计显著性”——当前 V3.6/Daft 仍各为单次 feasibility。

## 6. 复现与限制

Runner：
[`mineru_ray_data.py`](../../../rayorch/experimental/multigrain_v3_6/benchmark/mineru_ray_data.py)。
测试：
[`test_mineru_ray_data.py`](../../../test/experimental/multigrain_v3_6/benchmark/test_mineru_ray_data.py)。

每个正式 arm 使用：

```bash
python -m rayorch.experimental.multigrain_v3_6.benchmark.mineru_ray_data \
  --limit 368 --source-blocks 4 --regroup-mode {full_value,reference_only} \
  --output-dir OUT --artifact-dir ARTIFACT --result-jsonl RESULTS.jsonl
```

原始临时产物位于 `/tmp/rayorch_vldb_raydata368_{full,ref}_r{1,2,3}/artifact/`，包含
`summary.json`、`ray_data_stats.txt`、`gpu_samples.jsonl` 和 correctness hashes；正式
artifact 必须迁入有 checksum 的归档路径。

限制：当前 commit `813347a5` 外加未提交 runner，不是 frozen paper commit；单节点；
368 paths 只有 22 个不同 PDF checksum，是 replicated throughput/skew workload；模型权重
digest 尚未记录；未采集 NIC bytes；V3.6/Daft 尚未在同一交错三轮矩阵中复跑。因此本报告
支持机制与 feasibility，不替代最终论文主表。
