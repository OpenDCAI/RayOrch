# MinerU multi-Arena overlap 与 OCR 组批实验

日期：2026-07-24

分支：`codex/multigrain-recovery-tiers`

multi-Arena 实现 commit：`d275740`

本轮开发前基线 commit：`a1fde04`

## 1. 实验问题

本轮实验用于区分低吞吐可能来自的三类原因：

1. 多个 microbatch/Arena 没有真正 overlap；
2. driver 或 Ray actor 调度导致 OCR actor 空转；
3. OCR RPC 太小，反复支付 MinerU two-step UDF 的内部阶段开销。

同时验证：

> V2.5 当前的 elastic streaming scheduler，是否可以在不恢复旧 MG
> sealed-domain sharding 策略的前提下，回归到旧 MG 的吞吐水平。

## 2. 实验环境

```text
GPU                       4× NVIDIA H20
单卡显存                  97,871 MiB
系统内存                  约 2.2 TiB
Ray                       2.50.0
Python                    3.12
PDF 数量                  368
页面数量                  7,072
模型                      MinerU2.5-2509-1.2B
```

仓库和模型路径：

```text
RayOrch
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/RayOrch

Flash-MinerU
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru

模型
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/model/MinerU2.5-2509-1.2B
```

计时方法：

- 所有 persistent actors 通过 readiness barrier 后，才开始
  `measured_wall`；
- 模型构造和 vLLM warmup 单独计入 `startup`；
- 性能对比主要使用 `measured_wall`。

## 3. Pipeline 与调度结构

V2.5 使用公开 Pipeline API：

```text
synthetic Source
→ Expand(MinerUPdfToPages)
→ Map(MinerUVlmOcrPage)
→ Reduce(anchor=pdf, members=contents, pages=aligned_pages)
```

单次 run 内通过以下配置建立 bounded Arenas：

```python
Executor(
    pipeline,
    microbatch_size=<每个 Arena 的 PDF 数>,
    max_inflight_arenas=<同时活跃的 Arena 数>,
)
```

运行时合同：

- 每个 Arena 拥有独立的 GrainTable、indexes、barriers 和 generation；
- 多个 Arena 共享同一组 persistent Ray actors；
- Source position 在切分 Arena 前按 run-global ordinal 确定；
- Arena 完成后生成 detached `RunResult` 并立即 reclaim；
- 当前不跨 Arena 合成 Dispatch。

OCR node 的 `batch_size` 属于另一个层次，定义为：

> 一次发送给一个 OCR actor replica 的 RPC，最多包含多少个 page
> logical grains。

它不是每个 Arena 的 PDF 数量。

## 4. 核心实验命令

与旧 MG 宏观参数对齐的性能回归使用：

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
RAY_PROFILING=1 \
RAY_task_events_report_interval_ms=100 \
python -u -m rayorch.experimental.multigrain_v2_5.benchmark.mineru_cli \
  --mode elastic \
  --limit 368 \
  --replicas 4 \
  --microbatch-size 24 \
  --max-inflight-arenas 3 \
  --batch-size 128 \
  --max-batch-wait-ms 10 \
  --render-replicas 4 \
  --reduce-replicas 1 \
  --num-cpus 32 \
  --object-store-gb 60 \
  --rss-interval-s 1 \
  --output-dir \
    /apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/outputs_v2_5_regression_full_368_bs128_mb24_i3 \
  --timeline-dir \
    /apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/v2_5_timeline_regression_full_368_bs128_mb24_i3
```

本机 Ray 2.50 的实际行为是：

- `RAY_task_events_report_interval_ms=0` 时，legacy profile events 不会上报；
- 设置为正值 `100ms` 后，`ray.timeline()` 可以得到非空 trace。

因此本轮 timeline 实验使用 `100ms`。

## 5. 实验结果

### 5.1 368 PDF 全量对比

所有结果均使用：

```text
368 PDFs
7,072 pages
4×H20
4 个 persistent OCR actors
```

| 配置 | Arena 策略 | OCR batch | measured wall | pages/s | OCR bubble |
|---|---|---:|---:|---:|---:|
| V2.5 大 batch | 单 Arena | 256 | 491.738 s | 14.3816 | 7.29% |
| 旧 MG LPT | 24 PDF，inflight 3 | 动态 shard | 519.950 s | 13.6000 | 6.53% |
| V2.5 性能回归 | 24 PDF，inflight 3 | 128 | 521.836 s | 13.5522 | 8.82% |
| V2.5 中等 batch | 单 Arena | 64 | 597.149 s | 11.8429 | 5.40% |
| V2.5 小 batch | 24 PDF，inflight 4 | 16 | 898.394 s | 7.8718 | 2.07% |
| 当前原生 Flash-MinerU | doc-grain，inflight 4 | 8 PDFs | 818.001 s | 8.6455 | 未记录 |
| 历史 Flash-MinerU baseline | doc-grain，inflight 4 | 8 PDFs | 891.980 s | 7.9284 | 未记录 |

V2.5 性能回归与旧 MG LPT 的差异：

```text
绝对差异                   1.886 s
相对差异                   0.36%
```

该差异处于真实模型运行的正常波动范围内。实验表明，只要 OCR RPC
规模相近，V2.5 elastic streaming scheduler 可以回归到旧 MG 的吞吐水平。

当前原生 Flash-MinerU commit `7246a35` 的重新实测结果为 `818.001s`，
因此当前版本的原生对照应以该数字为准。原生实现和外层 PDF batch 对照见：

`docs/experiments/multigrain_v2_5/2026-07-24_native_flash_mineru_baseline.md`。

### 5.2 性能回归实验中的 OCR Dispatch

```text
OCR logical grains             7,072
OCR RPC 数量                       65
旧 MG 理论 RPC 数量                64（16 chunks × 4 replicas）
完整 128-page RPC 数量             46
120-page RPC 数量                   4
其余 bounded tail RPC              15
active Arena high watermark         3
OCR actor-capacity utilization   91.18%
同时存在 4 个 OCR RPC            OCR 区间的 82.60%
```

此前 `batch_size=16` 的全量实验产生了 447 个 OCR RPC；改为
`batch_size=128` 后只有 65 个，与旧 MG 每个 24-PDF chunk 产生四个
GPU shards 的物理调用数量接近。

### 5.3 控制面时延

性能回归实验中的 OCR Dispatch：

```text
worker finish → driver 收到 manifest    平均 9.44 ms
driver 收到 manifest → commit           通常约 2–3 ms
首个 render 完成 → 首个 OCR submit      约 254 ms
```

Ray timeline 和 V2.5 Dispatch timeline 中没有发现足以解释
`batch_size=16` 与 `batch_size=128` 之间数百秒差异的 driver 调度空泡。

### 5.4 48 PDF OCR batch sweep

48 PDFs / 992 pages，`microbatch_size=24`：

| OCR batch | max inflight Arenas | measured wall | pages/s |
|---:|---:|---:|---:|
| 16 | 4 | 104.780 s | 9.4675 |
| 32 | 4 | 86.988 s | 11.4038 |
| 64 | 3 | 77.985 s | 12.7204 |
| 128 | 单 Arena | 74.247 s | 13.3608 |
| 256 | 单 Arena | 73.104 s | 13.5697 |

尽管小 batch 实验中的 actor scheduling bubble 已经很低，提高 OCR UDF
单次调用规模仍然显著提升吞吐。因此主要损失来自反复进入 MinerU
`batch_two_step_extract()` 的内部阶段，而不是 multi-Arena admission 没有生效。

## 6. Timeline 证据

48-PDF、`batch_size=16`、`microbatch_size=24`、inflight 4 的 trace：

```text
Ray timeline events              1,338
RayWorker.run events               153
benchmark RPC count                153
worker lanes                        12
最大并发 actor run spans            11
```

联合 Ray timeline 和 V2.5 Dispatch timeline，可以把 worker lanes 映射为：

```text
4 个 render actors
4 个 OCR/vLLM actors
4 个 Reduce actors
```

Perfetto 中约 39–41 秒的四条 `__init__` span 对应四个 vLLM actors。
初始化后的可见间隔主要是首批 PDF render；首个 render 完成后约 254ms
即提交了首个 OCR RPC。

## 7. 内存观测

部分全量实验结果：

| 配置 | driver RSS peak | worker RSS peak | live blocks high watermark |
|---|---:|---:|---:|
| 单 Arena，OCR 256 | 596 MB | 35.32 GB | 506 |
| 单 Arena，OCR 64 | 596 MB | 37.60 GB | 618 |
| multi-Arena，OCR 16 | 593 MB | 33.04 GB | 172（跨活跃 Arenas） |
| 性能回归，OCR 128 | 584 MB | 27.42 GB | 91（跨活跃 Arenas） |

这些数字只是观测结果，不构成通用 bounded-memory 证明。

worker RSS 包含 vLLM 进程及其映射；更准确地区分 RSS/PSS/USS、allocator
fragmentation 和 Object Store mapping 仍属于后续 TODO。

## 8. 输出正确性

V2.5 全量 368-document 输出与 Flash-MinerU baseline 对比：

```text
matched documents             368/368
missing/extra                   0/0
token Jaccard median           0.995982
token Jaccard mean             0.992361
documents >= 0.98              331/368
```

该对比将不同 batching 引起的 vLLM 输出变化视为非字节级模型抖动，
不宣称 deterministic byte equality。

## 9. 外部实验产物

性能回归实验：

```text
日志
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/v2_5_regression_full_368_bs128_mb24_i3.log

输出
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/outputs_v2_5_regression_full_368_bs128_mb24_i3

Ray/V2.5/GPU timelines
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/v2_5_timeline_regression_full_368_bs128_mb24_i3
```

其他 trace：

```text
368 PDFs，OCR 16，inflight 4
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/v2_5_timeline_full_368_bs16_i4

48 PDFs，OCR 16，inflight 4
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/v2_5_timeline_48_bs16_i4_ray

48 PDFs，OCR 32，inflight 4
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/v2_5_timeline_48_bs32_i4
```

这些大体积产物不提交到 Git。

## 10. 结论与后续

1. bounded multi-Arena overlap 已经生效，并共享 persistent actors；
2. actor scheduling bubble 很低，并不代表小 OCR UDF 调用具有高吞吐；
3. `batch_size=128` 时，V2.5 与旧 MG 性能只差 0.36%；
4. 当前不需要为了修复 MinerU 吞吐立即实现第二套 sealed-domain scheduler；
5. elastic streaming 与 sealed-domain sharding 仍是两种不同的物理策略，
   可选模式开关已单独记录为 TODO；
6. 下一阶段应优先验证 grain-level recovery 和 failure containment，
   而不是继续增加 batching 策略。
