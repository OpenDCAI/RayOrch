# Multigrain V2.5 TODO

状态：用于记录已明确推迟、需要证据触发的 V2.5 工作。权威语义仍以
`docs/multigrain_v2_5_architecture.md` 为准。

## 当前原则

- 先完成单次 run 内的 Ray pipeline parallelism、elastic rebatching 和可复现实验；
- microbatch arena 必须在 DELIVERED/ABORTED 后统一 reclaim；
- 不因“未来可能需要”提前增加 checkpoint、通用 lease、recovery graph 或分布式 metadata；
- TODO 只有在对应 benchmark、真实 workload 或故障复现提供证据后才升级为实现任务。

---

## P1：Benchmark methodology 与实验矩阵

- [ ] 完整 manual matrix：
  - fan-out：uniform / log-normal / Pareto / Zipf-like；
  - service time：constant / uniform / log-normal；
  - actors：1 / 2 / 4；
  - batch size：1 / 4 / 16；
  - max batch wait：0 / 2 / 5 / 10 ms；
  - paired parent-bound / elastic repetitions。
- [ ] 保存 raw JSONL、summary JSON/CSV 和运行环境信息。
- [ ] 检查 final output digest parity。
- [ ] 明确 per-node bubble ratio，避免把正常的跨 stage idle 误报为 scheduler bubble。
- [ ] 所有论文数字重新在 V2.5 上运行，不复用旧实现数字。

## P1：Document-like workload

- [x] document/PDF → pages → real vLLM OCR Map → document Reduce。
- [x] 使用真实 Ray persistent actors、4×H20 和 coarse ObjectRef transport。
- [x] Reduce 同时消费 ordered contents fiber 与 aligned original page fiber。
- [ ] 覆盖真实 workload 的 bad-child/fail-closed recovery。
- [ ] 运行同版本 parent-bound full-scale ablation。

### 2026-07-24：真实 Flash-MinerU 368-PDF 结果

设置：

```text
368 PDFs
7072 pages
4×NVIDIA H20
MinerU2.5-2509-1.2B
4 persistent vLLM actors
OCR dispatch batch_size=256
max_batch_wait_ms=20
```

V2.5 public path：

```text
Pipeline
→ Expand(MinerUPdfToPages)
→ Map(MinerUVlmOcrPage)
→ Reduce(anchor=pdf, members=contents, pages=aligned_pages)
→ Executor.run
```

单次 run 的 bounded stream path 已实现：

```text
microbatch_size=24
max_inflight_arenas=3
多个独立 Arena 共享同一 persistent actor pool
source positions 在切 Arena 前按 run-global ordinal 冻结
完成 Arena 立即 detached delivery + reclaim
```

以下 491.74 s 结果来自此前的单 Arena、OCR batch 256 实验。模型加载/readiness
与 measured wall 分开：

```text
startup                          41.98 s
measured wall                   491.74 s
end-to-end incl. startup        533.73 s
throughput                      14.38 pages/s
OCR bubble                      7.29 %
RPC count                       505
grains/RPC                      15.46
batch fill ratio                84.58 %
tail RPC fraction               5.94 %
live blocks high watermark      506
driver RSS start/peak           537 MB / 596 MB
worker RSS peak                 35.3 GB
per-GPU memory peak             ~83.4 GB
outputs                         368/368 markdown
```

对比：

```text
Flash-MinerU measured baseline  891.98 s → V2.5 1.814×
previous V2 multigrain          519.95 s → V2.5 1.057×
```

输出逻辑一致性（V2.5 vs Flash-MinerU baseline）：

```text
matched                         368/368
missing/extra                   0/0
token Jaccard median            0.995982
token Jaccard mean              0.992361
token Jaccard min               0.933066
token Jaccard >= 0.98           331/368 (89.95 %)
```

该分布与既有 vLLM batching jitter 范围一致；不宣称 byte-identical。
Benchmark CLI：`python -m rayorch.experimental.multigrain_v2_5.benchmark.mineru_cli`。

### 小 OCR batch + microbatch inflight 复测

48 PDFs / 992 pages / 4×H20，`microbatch_size=24`、
`max_inflight_arenas=3`：

| OCR batch | measured wall | pages/s | OCR bubble | worker RSS peak |
|---:|---:|---:|---:|---:|
| 16 | 104.19 s | 9.52 | 5.09 % | 11.51 GB |
| 64 | 77.99 s | 12.72 | 8.75 % | 10.12 GB |

与单 Arena 对照几乎相同（batch16 104.10 s；batch64 75.33 s）。说明 V2.5
单 Arena 已经能在 grain 级别让 render/OCR/reduce 流水线重叠；multi-arena 的主要收益是
bounded live set 和每 microbatch reclaim，而不是修复 actor starvation。

Full 368-PDF batch64 单 Arena 结果为 597.15 s、OCR bubble 5.40 %、worker RSS peak
37.60 GB；GPU actors 已经较忙，但 vLLM 小调用效率显著低于 batch256 的 491.74 s。
因此暂不增加 per-actor queued pending/concurrency，后续优先精确区分 actor/EngineCore
RSS/PSS/USS。

使用旧 MG 的宏观参数重新运行 V2.5：

```text
microbatch_size                 24 PDFs
max_inflight_arenas             3
render replicas                 4
OCR replicas                    4
OCR batch_size                  128 pages/RPC/replica
Reduce replicas                 1
```

全量 368-PDF 结果：

```text
V2.5 measured wall              521.836 s
previous MG LPT                 519.950 s
difference                        1.886 s / 0.36 %
OCR RPC count                       65
active Arena high watermark          3
```

这表明此前 batch16 的 898.394 s 主要是 OCR UDF 调用粒度回归，不是
multi-Arena overlap 或单 driver event loop 失效。完整实验方法、timeline、
GPU/RSS 采样和 artifact 路径见：

`docs/experiments/multigrain_v2_5/2026-07-24_mineru_overlap_and_batching.md`。

---

## P2：Arena 内 coarse block early release（证据触发）

当前允许采用 **arena-scoped coarse-block ownership**：

- 中间 ObjectRefs 最迟在 microbatch arena DELIVERED/ABORTED 后统一释放；
- `max_grains_per_arena`、`max_fanout_per_grain` 和
  `max_pending_dispatches` 提供硬边界；
- 该简化保证跨 runs 不保留历史引用，但不声称单个超大 arena 的 peak
  object-store/RSS 已最优。

只有出现以下证据时才启动 early release：

- [ ] `live_blocks_high_watermark` 随已完成 stages/历史 emissions 显著累积；
- [ ] bounded microbatch 仍触发 object-store pressure/OOM；
- [ ] delivery 后可以回收，但 run 内 peak RSS 明显限制吞吐或 fan-out scale。

若触发，首选最小实现：

```text
(Dispatch, output port) coarse block
AND 全部 compiled consumers terminal
AND 无 active/retry dispatch 引用
AND 不是 final output
→ 删除 Arena 对 ObjectRef 的引用
→ 删除对应 ValueIndex locators
```

明确不预先实现：

- ObjectHoldLedger；
- lease protocol；
- 通用 reference-count service；
- byte-credit state machine；
- RSS feedback controller；
- actor rotation。

## P2：Memory/RSS soak

- [ ] 单次大 microbatch 的 live block count/high watermark。
- [ ] delivery/abort 后 Arena block/index 数归零。
- [ ] driver RSS start/end/peak。
- [ ] worker RSS start/end。
- [ ] 区分 pinned ObjectRefs、plasma mapping、allocator fragmentation 和 metadata leak。
- [ ] RSS 异常前不加入 jemalloc、actor recycle 或复杂 profiler。

---

## P2：Timeout/cancel fencing

- [ ] dispatch timeout 使用当前 AttemptToken/generation fencing。
- [ ] timeout 后 grain 回 READY 或耗尽预算后 arena abort。
- [ ] late success/error callback 必须 stale no-op。
- [ ] mixed current/stale 继续视为 executor invariant violation。
- [ ] 不增加 Grain lifecycle 状态。

## P3：Failure/Ray robustness

- [ ] 多 replica actor failure/replacement soak。
- [ ] completion storm 下 bounded manifest processing。
- [ ] accepted ObjectRef 丢失仍按 arena abort 处理。
- [ ] driver crash recovery、checkpoint/resume 继续不做。

---

## P3：完整 Relate

当前只保留 bounded、sealed-port、显式 int-key integration prototype。

- [ ] canonical typed-key codec；
- [ ] bool/int、bytes/str、Unicode、NaN 边界 golden；
- [ ] key projection failure → arena abort；
- [ ] per-node/per-arena relation cardinality guard；
- [ ] chunked Cartesian enumeration；
- [ ] production JoinIndex；
- [ ] 是否进入论文核心实验由 workload 结果决定。

明确不做：

- distributed join；
- spill；
- streaming window；
- hot-key distributed execution。
