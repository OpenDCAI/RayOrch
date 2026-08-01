# Multigrain V3 368-PDF 四系统对比（2026-08-01）

## 1. 实验问题

在完全相同的：

```text
368 PDFs
7,072 pages
4× NVIDIA H20
MinerU2.5-2509-1.2B
```

输入上比较：

1. V3 elastic；
2. V3 parent-bound；
3. 裸 Ray Data；
4. 当前 Flash-MinerU native DAG pipeline。

主比较使用 startup-inclusive end-to-end wall。Ray Data 没有稳定公开的 actor readiness
barrier，因此不能把 V3/native 的 post-readiness measured wall 与 Ray Data 的
startup-inclusive wall 混作公平速度比较。

## 2. 系统配置

### V3 elastic / parent-bound

```text
microbatch_size            24 PDFs
max_inflight_arenas         3
OCR replicas                4
OCR batch cap              64 pages
```

两个模式唯一关键差异：

```text
elastic       pages 可跨 PDF parent 组 batch
parent_bound  OCR batch 不跨最近 parent
```

### Ray Data

```text
flat_map(PDF → page tensor)
map_batches(MinerU OCR, 4 GPU actors, batch cap 64)
groupby(parent_id)
map_groups(ordered document assembly)
```

应用显式维护 `parent_id/page_ordinal`。

### Native Flash-MinerU

```text
PDF batch size              8
max batches inflight        4
render/OCR/convert replicas 4
```

native OCR actor 逐 PDF 调用 `batch_two_step_extract`，不同 PDF 的 pages 不会合成同一个
vLLM batch。

## 3. 结果

本轮先完成一个完整四模式 paired set：

| System | Measured wall | Startup | End-to-end | Pages/s |
| --- | ---: | ---: | ---: | ---: |
| V3 elastic | 587.781 s | 39.100 s | **626.890 s** | **12.0317** |
| V3 parent-bound | 795.071 s | 39.613 s | 834.699 s | 8.8948 |
| Ray Data | 982.549 s | included | 982.549 s | 7.1976 |
| Flash-MinerU native | 822.476 s | 45.184 s | 867.659 s | 8.598* |

`*` native pages/s 由 `7,072 / 822.476` 推导，仅用于说明 measured throughput。

End-to-end speedup：

```text
V3 elastic vs parent-bound    1.331×
V3 elastic vs Ray Data        1.567×
V3 elastic vs native          1.384×
```

Measured-wall elastic vs parent-bound：

```text
795.071 / 587.781 = 1.353×
```

这与 V2.5 的三次 paired median `1.400×` 方向一致。

## 4. Packing 证据

### V3

| Mode | all-stage RPC | Grains/RPC | Batch fill |
| --- | ---: | ---: | ---: |
| elastic | 642 | 12.74 | 89.73% |
| parent-bound | 1,117 | 7.32 | 31.58% |

这里是全 DAG 指标，不是只统计 OCR。尽管如此，唯一改变的核心调度策略是 OCR child
是否允许跨 parent packing，RPC 和 fill 差异与预期一致。

### Ray Data

Ray Data 的 OCR stage 能形成较大 batch：

```text
OCR tasks                  99
rows/task              60–72
```

但 ordered document assembly 通过 `groupby(parent_id)` 完成，导致完整 page payload 被
hash shuffle：

```text
page/image/OCR columns       ~79.9 GB
shuffle total stage          973.67 s
shuffle finalize blocks      19.1–21.5 GB / block
observed host used memory    briefly ~1.0 TiB
```

Ray Data 最终成功，没有 OOM；但恢复 parent group 需要对大业务 payload 做全量 shuffle。
V3 Reduce 使用 lineage receipts 和 coarse block row references，不需要按 parent 重分区整张
page image 和 OCR payload。这是 V3 相对裸 Ray Data 的主要系统差异，而不只是 API 更短。

## 5. Correctness

以 V3 elastic 输出为左侧：

| Compared with | Documents | Missing | Jaccard median | Mean | Min | ≥0.98 | ≥0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 parent-bound | 368 | 0 | 0.99315 | 0.98968 | 0.92775 | 317 | 363 |
| Ray Data | 368 | 0 | 0.99538 | 0.99164 | 0.92777 | 327 | 361 |
| Native | 368 | 0 | 0.99304 | 0.98858 | 0.73489 | 312 | 362 |

MinerU/vLLM 多次运行并非 bitwise deterministic，因此使用 token Jaccard。所有系统均产生
368 个文档，无缺失/额外输出。

## 6. 结论边界

本轮证据支持：

1. V3 clean-slate rewrite 保留了 MinerU 主性能路径；
2. 在 V3 内部保持模型、batch cap、GPU、输入一致时，cross-parent elastic 相对
   parent-bound 约 `1.35×` measured speedup；
3. V3 elastic 相对当前 native pipeline 的 end-to-end speedup 为 `1.384×`；
4. Ray Data 可以完成同一语义和大 OCR batch，但 ordered regroup 的通用 hash shuffle
   带来大 payload movement 和高 host-memory 水位；
5. V3 lineage-aware Reduce 避免了这类全量 regroup shuffle。

本轮不能支持：

- 所有 workload 都能从 elastic 得到 wall-time speedup；
- Ray Data 在所有写法中都必须产生同样 shuffle；
- 通用 bounded-memory theorem；
- 统计显著性结论。

目前四系统各一轮。V2.5 已有三次 paired elastic/parent 结果，本轮 V3 单轮与其一致；正式
论文表格仍应再补至少两轮 V3 elastic/parent。Ray Data full 每轮会产生约 80GB shuffle 和
接近 TiB 级 host-memory 瞬时水位，是否重复三次应根据实验成本和论文审稿需求决定。
