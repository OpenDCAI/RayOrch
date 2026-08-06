# Multigrain V3.5 MinerU 回归（2026-08-06）

## 目的

验证 V3.5 引入 `LogicalProgram → analysis → RuntimePlan` compiler 边界后，没有丢失
V3 已验证的真实 workload 性能来源：

- PDF→Page 动态 `1:M`；
- 跨 PDF elastic page batching；
- 多 Arena overlap；
- ordered Page→Document Reduce；
- 4×H20 persistent OCR actor pool。

V3.5 runner 直接复用 V3 的 `MinerUPdfToPages`、`MinerUVlmOcrPage`、
`PdfMetadata` 和 `MinerUAssembleDoc`，只替换 Pipeline authoring、compiler、Arena 和
Executor。因此模型、render、assemble 和输出逻辑不是本轮变量。

## 固定配置

```text
Input                  368 PDFs / 7,072 pages
GPU                    4× NVIDIA H20 96 GB
OCR replicas           4
OCR batch cap          64 page Grains/RPC
Arena size             24 PDFs
max in-flight Arenas   3
render/reduce replicas 4 / 4
GPU memory utilization 0.8
batch policy           immediate work-conserving
```

历史性能 gate：V3 measured wall 587.781s，允许 ±5%。较新的 V3.3 clean full run
为 592.975s。它们是独立历史运行，不是与本次同一时刻交错执行的 paired trial。

可复现入口：

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
python -u -m rayorch.experimental.multigrain_v3_5.benchmark.mineru \
  --mode elastic \
  --limit 368 \
  --replicas 4 \
  --microbatch-size 24 \
  --max-inflight-arenas 3 \
  --batch-size 64 \
  --gpu-memory-utilization 0.8 \
  --render-replicas 4 \
  --reduce-replicas 4 \
  --num-cpus 32 \
  --object-store-gb 100 \
  --output-dir /tmp/rayorch_v35_mineru_368_elastic_20260806/outputs \
  --artifact-dir /tmp/rayorch_v35_mineru_368_elastic_20260806/artifacts \
  --result-jsonl /tmp/rayorch_v35_mineru_results_20260806.jsonl
```

## 分级 gate

| Gate | Pages | Measured | Throughput | OCR RPC | Pages/RPC |
|---|---:|---:|---:|---:|---:|
| 4 PDF | 48 | 18.266s | 2.6278 pages/s | 4 | 12.000 |
| 48 PDF | 992 | 78.863s | 12.5788 pages/s | 20 | 49.600 |
| 368 PDF | 7,072 | 602.760s | 11.7327 pages/s | 121 | 58.446 |

4-PDF gate 与历史 V3.3 的 17.889s 相差 +2.1%。48-PDF gate 比历史 V3.3 的
83.098s 快 5.1%，其 OCR RPC 数、平均 batch 和完整 histogram 均完全一致。

## 全量性能结果

```text
model/actor startup        40.172 s
measured pipeline wall    602.760 s
end-to-end wall           642.934 s
throughput                 11.7327 pages/s
outputs                    368 / 368
active Arenas high water     3
driver RSS peak            588,087,296 bytes
released value bindings     16,352
```

相对 V3 golden：

```text
V3 golden                 587.781 s
V3.5                      602.760 s
V3.5 / V3                  1.02548
slowdown                     2.55%
gate                        PASS (≤5%)
```

OCR packing：

```text
page Grains                7,072
OCR RPC                      121
average pages/RPC          58.4463
batch fill ratio             91.32%
full 64-page RPC              102
```

OCR RPC 数与平均 pages/RPC 和 V3.3 clean full run 完全一致。V3.5 虽然没有
定时 batch wait，而是 actor 空闲时发送当前可见 Grain，但在固定 24-PDF Arena / 3-Arena
配置下保留了相同 packing 行为。

## Correctness

V3.5 full output 与 V3.3 clean full artifact 比较：

```text
matched documents          368 / 368
missing / extra              0 / 0
token Jaccard median       0.99596774
token Jaccard mean         0.99332145
token Jaccard minimum      0.93085106
Jaccard >= 0.95            367 / 368
Jaccard >= 0.98            337 / 368
```

该分布落在历史 MinerU/vLLM 独立运行的非确定性范围内；没有 framework-level 文档或
页面丢失。4-PDF 与 48-PDF gates 同样无 missing/extra，且所有文档 Jaccard ≥0.98。

仓库外 artifacts：

```text
/tmp/rayorch_v35_mineru_4_gate_20260806
/tmp/rayorch_v35_mineru_48_gate_20260806
/tmp/rayorch_v35_mineru_368_elastic_20260806
/tmp/rayorch_v35_mineru_results_20260806.jsonl
```

## 结论

V3.5 compiler 化没有改变真实 MinerU 的动态 fan-out、elastic packing 或 ordered Reduce
结果。368-PDF measured wall 相对 V3 golden 退化 2.55%，通过既定 ±5% gate；结构性
证据（OCR RPC 数与 batch 分布）也与 V3.3 full run 一致。

这个实验不证明所有 workload 都能从即时聚批获得相同性能，也不是同一时刻的 paired
统计实验。它证明固定 MinerU golden workload 在当前机器、当前依赖和同一业务 UDF 下，
V3.5 仍保持 V3 级别的吞吐与输出语义。
