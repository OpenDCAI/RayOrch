# Video B：MSR-VTT SmolVLM 四卡实验（2026-08-04）

## 1. 配置

复用 Video A 的固定 MSR-VTT manifest：

```text
1.3GiB / 6,236 independent videos
duration 10.0–30.67s
```

Pipeline：

```text
Video
→ Expand(最多4个key frames)
→ Map(SmolVLM-256M greedy caption)
→ Reduce(ordered captions)
```

```text
model                     HuggingFaceTB/SmolVLM-256M-Instruct
revision                  7e3e67edbbed1bf9888184d9df282b700a323964
4×H20
stride                     12
max frames/video            4
sampled frames/captions    23,060
caption actors              4 × 1 GPU
caption batch cap          16
max_new_tokens             12
microbatch                 32 videos/Arena
max inflight Arenas         4
```

由于单 pair 已耗时约 78 分钟，本实验只作为规模化 feasibility，不作为多次统计主表。

## 2. 结果

| Mode | Measured | E2E |
| --- | ---: | ---: |
| Parent-bound | 2371.924s | 2379.136s |
| Elastic | 2263.400s | 2270.577s |

```text
elastic measured speedup     1.048×
elastic E2E speedup          1.048×
```

Packing：

| 指标 | Parent | Elastic |
| --- | ---: | ---: |
| Caption RPC | 6,236 | 1,550 |
| Frames/RPC | 3.70 | 14.88 |
| Full batch RPC | 0 | 1,348 |
| All-stage fill | 0.272 | 0.901 |
| Caption busy sum | 9476.3s | 9039.4s |
| Caption span | 2371.8s | 2263.3s |

Elastic 大幅减少 RPC，但 generation 的 wall 收益只有约 4.8%。原因包括：

- 每条 caption generation 较短，GPU 平均利用率仅 14–16%；
- generation batch padding/KV 代价高于纯 encoder；
- parent 每视频只有3–4帧，虽严重 underfill，但多请求合批不按比例降低 decode 时间。

显存：

```text
Parent     ~3.9–4.1GB/GPU
Elastic    ~11.8–13.6GB/GPU
```

说明 generation batching 用显存换取约 5% wall 收益。

## 3. Correctness

```text
video/frame/source-index structure exact
caption mismatches          191 / 23,060 frames
mismatch rate               0.83%
normalized mismatch         177 / 23,060 = 0.77%
affected videos             189 / 6,236
```

虽然使用 greedy `do_sample=False`，不同 batch padding/GEMM shape 仍会改变少量生成 token。
因此不能声称 captions bitwise exact；论文应报告结构 exact 和文本漂移率。首个例子：

```text
CONVERSE.  vs  CONVERSATION.
```

## 4. 结论

Video B 支持：

- V3 能对动态 frame captions 做 cross-video elastic packing；
- 在真实 generation workload 上获得约 `1.048×` 单 pair speedup；
- packing 收益低于纯 ViT encoder 的 `1.159×`，且显存代价明显更高。

边界：

- 只有一组 parent/elastic pair，不能作为稳定统计主结论；
- captions 有约0.8% batch-shape文本漂移；
- 当前没有 Ray Data caption baseline。

因此 Video B 放在 feasibility/敏感性实验，Video A 仍是正式视频性能主证据。
