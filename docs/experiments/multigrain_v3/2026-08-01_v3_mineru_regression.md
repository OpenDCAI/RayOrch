# Multigrain V3 MinerU 回归（2026-08-01）

## 目的

验证 clean-slate V3 在保留四组件模型和更清晰数据结构后，仍能复现：

- 细粒度 PDF→Page `1:M`；
- 跨 parent elastic rebatching；
- 多 Arena in-flight 流水线；
- ordered Page→PDF Reduce；
- 4×H20 persistent OCR actor pool。

## 环境

```text
GPU                 4× NVIDIA H20, 97,871 MiB
Flash-MinerU        /apdcephfs_zwfy10/share_304380933/hunyuan/
                    sunnyhazema/workspace/Flash-mineru
Model               MinerU2.5-2509-1.2B
Input               368 PDFs / 7,072 pages
OCR replicas        4
OCR batch cap       64 logical page Grains/RPC
microbatch_size     24 PDFs/Arena
max_inflight        3 Arenas
batch wait          20 ms
```

可复现入口：

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
python -u -m rayorch.experimental.multigrain_v3.benchmark.mineru \
  --mode elastic \
  --limit 368 \
  --replicas 4 \
  --microbatch-size 24 \
  --max-inflight-arenas 3 \
  --batch-size 64 \
  --max-batch-wait-ms 20 \
  --gpu-memory-utilization 0.8 \
  --render-replicas 4 \
  --reduce-replicas 4 \
  --num-cpus 32 \
  --object-store-gb 100 \
  --output-dir /path/to/outputs \
  --artifact-dir /path/to/artifacts \
  --result-jsonl /path/to/results.jsonl
```

## 结果

```text
model/actor startup             39.100 s
measured pipeline wall         587.781 s
end-to-end wall                626.890 s
throughput                      12.0317 pages/s
outputs                         368 / 368
active Arenas high watermark     3
driver RSS peak                579.9 MB
```

OCR Stage：

```text
page Grains                    7,072
OCR RPC                          120
average pages/RPC               58.93
full RPC                         101
timeout/tail RPC                  19
OCR bubble ratio                3.87%
```

V2.5 参考：

```text
V2.5 elastic median            581.509 s
V3                             587.781 s
V3 / V2.5 slowdown               1.08%
```

V3 达到设计 gate 的 `±5%` 性能回归范围，同时保持接近满载的 OCR packing。

## Correctness

V3 与 V2.5 `full_r3_elastic_bs64` Markdown 输出逐文档比较：

```text
V3 documents                   368
V2.5 documents                 368
matched                        368
missing/extra                  0 / 0
token Jaccard median           0.99766
token Jaccard mean             0.99477
token Jaccard minimum          0.93333
Jaccard >= 0.98                340 / 368
Jaccard >= 0.95                365 / 368
```

该差异范围与已有 V2.5 多次 vLLM paired runs 的非确定性相符；未发现缺失或额外文档。

## 48-PDF 预回归

```text
48 PDFs / 992 pages
measured wall                  74.743 s
throughput                     13.2722 pages/s
OCR RPC                        19
average pages/RPC              52.21
OCR bubble ratio               6.51%
active Arenas high watermark    2
```

当输入只有两个 24-PDF chunks 时，high watermark 为 2，符合输入上限而不是配置上限 3。

## 结论

V3 的结构重写没有丢失 V2.5 的核心性能来源：

```text
dynamic fan-out
→ page Grain cross-parent packing
→ 4-GPU coarse OCR RPC
→ ordered Reduce
→ bounded multi-Arena overlap
```

本实验只证明该固定 MinerU 配置的性能与 correctness 回归，不证明通用 bounded-memory
或所有 workload 的性能结论。
