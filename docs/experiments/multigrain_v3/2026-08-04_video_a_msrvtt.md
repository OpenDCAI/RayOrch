# Video A：MSR-VTT 1.3GiB 四卡实验（2026-08-04）

## 1. 输入与模型

本地完整 MSR-VTT：

```text
7,010 independent videos
local corpus size       1.5GB
```

正式固定子集：

```text
videos                  6,236
input bytes             1,395,938,505（1.3GiB）
duration                10.0–30.67s
duration median         13.67s
manifest SHA-256:
cbec4f3a2effdc7c77e9c03fcc6f3b5326effc065b0b9ff2b6dc8803780d0c99
```

模型：

```text
google/vit-base-patch16-224-in21k
revision b4569560a39a0f1af58e3ddaf17facf20ab919b0
4×H20
```

## 2. Pipeline 与参数

```text
Video
→ Expand(sampled frames)
→ Map(ViT embedding)
→ Reduce(ordered video summary)
```

```text
stride                    4
max_frames/video          32
sampled frames            76,116
decode/reduce replicas    4 / 4
ViT actors                4 × 1 GPU
batch cap                 16
model_repeats             20
microbatch                32 videos/Arena
max inflight Arenas       4
```

`model_repeats=20` 是固定 compute-intensity sensitivity，用于模拟更深 frame encoder；
V3 parent、V3 elastic 和 Ray Data 使用完全相同的业务 UDF 和重复次数。

## 3. 结果

| Engine | Runs | Startup-inclusive wall | Median | CV |
| --- | ---: | --- | ---: | ---: |
| V3 parent-bound | 2 | 768.107 / 767.844s | **767.975s** | 0.02% |
| V3 elastic | 2 | 661.791 / 663.265s | **662.528s** | 0.16% |
| Ray Data | 2 | 679.384 / 682.526s | **680.955s** | 0.33% |

```text
V3 elastic vs parent paired speedup median     1.159×
V3 elastic vs Ray Data median                  1.028×
```

四卡平均利用率：

```text
V3 parent     ~89–91%
V3 elastic    ~88–89%
Ray Data      ~85–88%
```

## 4. Packing 证据

首轮 transform Stage：

| 指标 | Parent | Elastic |
| --- | ---: | ---: |
| transform RPC | 7,265 | 4,858 |
| frames/RPC | 10.48 | 15.67 |
| full batch RPC | 1,196 | 4,669 |
| all-stage fill | 0.603 | 0.872 |
| transform busy sum | 3031.6s | 2609.8s |
| transform span | 761.4s | 655.0s |

ViT batch curve已证明 batch 16 前吞吐随 batch 增长，因此这组 workload 能把 logical packing
转化为真实 kernel efficiency。与 Docling 的负结果形成互补：

```text
batch-efficient ViT/vLLM      → elastic 正收益
逐 page/逐 table Docling      → elastic 负收益
```

## 5. Correctness

所有 runner：

```text
video count                 6,236 exact
frames                      76,116 exact
source frame indices        exact
mean edge density           exact
```

不同物理 batch shape 下，ViT GEMM 末位浮点可能改变 top-k embedding index signature：

```text
digest mismatch frames      4 / 76,116
mismatch rate               0.0053%
affected videos             4 / 6,236
```

V3 elastic 与 Ray Data 相对 parent 均为相同数量级。正式 gate 是结构 exact + 报告 digest
mismatch rate，不要求 raw embedding/JSON bitwise equality。

## 6. 结论

该实验支持：

1. 在 6,236 个独立视频、自然 frame-count skew 和 4×H20 下，V3 elastic 稳定获得约
   `1.159×` parent-bound speedup；
2. 收益来自 transform batch 从平均 10.48 提到 15.67，并直接减少 ViT worker busy/span；
3. V3 elastic 比自然 Ray Data pipeline 快约 `1.028×`，差距较小但两轮稳定；
4. elastic 的收益依赖底层模型 batchability，而不只是 RPC 数量。

边界：

- 数据总量为 1.3GiB，不是原计划中的 4GiB；但已覆盖完整本地 MSR-VTT 的大多数视频；
- `model_repeats=20` 是 compute-intensity sensitivity，论文必须显式标注；
- 当前没有实现专家优化的 Ray Data reference-only Reduce；Video summary payload较小，
  因而自然 Ray Data 已是合理 baseline。

## 7. 产物

```text
/tmp/mgv3-msrvtt/manifest-1.3g.json
/tmp/mgv3-video-a/full/
```

每个 arm 保存 summary、compressed outputs、GPU samples 和日志。
