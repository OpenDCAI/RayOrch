# 视频三类 workload 实验计划

## 1. 共同语义

所有方案共享：

```text
Video
→ dynamic Clips/Frames
→ heavy models
→ ordered Clip/Video Reduce
```

统一比较：

```text
V3 parent-bound
V3 elastic
Ray Data natural
Ray Data optimized（若需要 regroup）
native/batched loop
```

数据必须是多个独立视频，不用复制同一文件作为正式结果。UCF101 subset 仅用于 smoke；
正式输入至少数百 clips，并报告 frame-count skew。

## 2. 方案 A：GPU frame encoder

### 模型

优先：

```text
openai/clip-vit-base-patch32   ~605MB PyTorch
```

备选：

```text
google/vit-base-patch16-224-in21k   ~346MB safetensors
```

### DAG

```text
Video
→ Expand(sampled frames)
→ Map(CLIP/ViT image embedding, GPU)
→ Reduce(ordered frame embeddings → video summary)
```

### 原因

- 最小、稳定；
- embedding 输出 grain-separable；
- GPU batch efficiency 容易测；
- 可直接做 batch curve 1/4/8/16/32/64；
- 最适合验证纯 cross-parent elastic rebatching。

首个正式视频主实验应选择该方案。

若单次 ViT forward 相对 decode/startup 太轻，可使用 `model_repeats` 固定重复同一 encoder
forward，模拟现实中更深的 frame model。V3/Ray Data 必须使用相同 repeats；报告应明确这是
compute-intensity sensitivity，不伪装成不同模型。

Correctness signature 不使用 raw embedding bytes。不同物理 batch shape 的 GEMM 可能产生
末位浮点差异；使用绝对值最大的 embedding dimensions 的 canonical top-k index signature，
并另外比较 summary statistics。

## 3. 方案 B：Frame caption/VLM

### 模型

候选：

```text
HuggingFaceTB/SmolVLM-256M-Instruct
```

PyTorch safetensors 约 513MB。

### DAG

```text
Video
→ Expand(key frames)
→ Map(VLM caption, GPU)
→ Reduce(captions → video narrative)
```

### 价值

- 更贴近多模态数据治理；
- generation 长度变化带来显著 skew；
- 可测试 bad frame/generation failure。

### 风险

- 输出非 bitwise deterministic；
- generation batching 和 KV cache 影响复杂；
- correctness 需要语义/结构 gate；
- 模型 latency 较高。

作为第二视频 case，不阻塞方案 A。

## 4. 方案 C：ASR + frame 双分支

### 模型

```text
openai/whisper-tiny   ~151MB
```

### DAG

```text
Video
├── Expand(audio chunks) → Map(Whisper)
└── Expand(frames/clips)  → Map(CLIP/ViT)

→ Clip-level aligned Map/Reduce
→ Video Reduce
```

### 价值

- 展示 General DAG、多 Port lineage；
- 两个不同 fan-out domain；
- 同一 video 的 audio/frame branches；
- 更贴近真实多模态 pipeline。

### 约束

- 需要 ffmpeg/torchaudio decode；
- audio chunk 和 frame clip 的时间对齐必须显式；
- 当前 V3 无 keyed M:N，首版使用固定时间窗、position-aligned clips；
- 不实现 streaming window join。

作为架构多样性 case，性能主结论仍由方案 A/MinerU 提供。

初步 smoke 已跑通：

```text
4 logical videos
audio chunks/video       2
frames/video             7–8
ASR                      Whisper tiny
frame model              ViT Base
GPU sharing              0.5 + 0.5 H20
E2E                      5.15s
measured after startup   0.51s
```

输出按 source video 对齐合并，验证：

- 两个独立 Expand scopes；
- Whisper/ViT 两个 persistent GPU actors；
- 两个 ordered Reduce；
- document-scope aligned Map merge。

当前只作架构 smoke，不作性能结论。

## 5. 正式数 GB 数据集与 manifest 合同

### 5.1 选择原则

正式 A/B 不再使用 UCF101 的两个 sample 重复输入。建议主数据采用：

```text
完整 UCF101（公开视频 action clips）
```

原因：

- 官方数据集含 13,320 个视频、101 个动作类别和约 27 小时视频，适合从原始独立 clips
  中裁出数 GB 固定 workload；
- 动作场景多样，能避免把同一个 codec/画面重复成“调度数据集”；
- 官方提供原始视频 archive；比 Kinetics 这类 URL/约十秒 clips 的集合更容易冻结为本地、
  可复现的文件 manifest；
- 视频时长通常较短，但可以通过真实 frame count、分辨率、codec 和 decode 速度形成自然 skew；
- 若完整 UCF101 本地 archive 不可用，可退化到同样有大量独立 clips 的公开动作/多媒体数据集；
  **不得**用文件复制补足正式样本。

不在仓库内下载、镜像或提交视频数据。数据许可、下载账户和本地路径由实验环境负责。

### 5.2 Manifest-first

正式 run 必须由有序 JSON manifest 驱动，而不是 shell glob：

```text
dataset
split
source_id
local path
label（若有）
file bytes
OpenCV-probed duration
```

新工具：

```bash
python -m rayorch.experimental.multigrain_v3.benchmark.video.manifest \
  --input-root /datasets/UCF-101 \
  --dataset ucf101 \
  --split all \
  --target-gib 4 \
  --min-count 300 \
  --seed 20260803 \
  --output /tmp/mgv3-video/ucf101-4g-manifest.json
```

它稳定枚举可 decode 的本地视频，按 duration bucket：

```text
<=5s / (5,15]s / (15,45]s / >45s
```

轮转取样直到同时满足：

```text
至少 300 个独立 source
实际视频文件总大小 >= 4 GiB
```

并写入相邻 `.report.json`，记录候选数、总 bytes、duration min/max/sum 和四个 bucket 的
计数。最后一个文件可以使总大小略超过目标；绝不截断文件或以逻辑重复凑容量。

实验前 `video.compare --manifest ...` 会重 probe bytes/duration，若文件被替换或损坏则拒绝
运行。正式 report 应记录 manifest 文件的 SHA-256。

### 5.3 A 的正式矩阵

同一个 manifest、同一 GPU、同一 ViT revision、同一 frame stride/max-frames 和 heavy-stage
batch cap 下：

```text
native/batched loop（若实现）
Ray Data natural
V3 parent-bound
V3 elastic
```

首先完成 V3 parent-bound 与 elastic；Ray Data/native 只在不改变业务 UDF 的前提下补入。
每个模式：

```text
warmup = 1
measured repeats = 3
记录 startup-inclusive wall 和 post-readiness measured（若 runner 支持）
记录 frame count distribution、V3 RPC/fill、GPU utilization/timeline、RSS
```

V3 parent/elastic 两臂只允许改变 `batch_scope`。若容量过大，先用 1 GiB / 100 clips 验证
正确性与容量，再升至 4 GiB / 300 clips；不得改模型或 batch cap 来“修”某一臂。

### 5.4 当前状态

已完成：

- UCF101/tiny video decode smoke；
- A 的 ViT batch curve 与小规模 parent/elastic 机制验证；
- B caption、C audio+frame General DAG smoke；
- 本地视频目录 → long-tail manifest 工具和无模型测试。

已完成：

- 完整本地 MSR-VTT 7,010 独立视频；
- 1.3GiB / 6,236-video manifest；
- Video A 4×H20 parent/elastic/Ray Data 两轮正式实验；
- elastic vs parent `1.159×`，elastic vs Ray Data `1.028×`；
- 结构 exact，ViT top-k digest mismatch `4/76,116`。

未完成：

- 尚未确认本机已有完整 UCF101 或其他数 GB 公开视频镜像；
- Video B 尚未在该完整 manifest 上规模化运行；
- Video A 数据量为1.3GiB，未达到原计划4GiB，但使用了本地MSR-VTT大多数独立视频；
- 尚未实现 native/batched-loop 视频 baseline。

正式 Video A 结果见 `2026-08-04_video_a_msrvtt.md`。

Video B 后续已完成同一 1.3GiB manifest 的单组四卡 pair，见
`2026-08-04_video_b_msrvtt.md`；由于单 pair 约78分钟，只作为 feasibility。

此外已下载官方完整 UCF101，并生成 4.000GiB / 8,448-video / 101-class manifest。
Video A 单组四卡 pair 获得 `1.024×` elastic speedup；详见
`2026-08-04_video_a_ucf101_4g.md`。

## 6. 执行顺序

1. 下载固定 revision 的 CLIP/ViT；
2. 单 GPU batch curve；
3. 用 1 GiB / 100 independent clips 生成并验证 manifest；
4. V3 parent-bound vs elastic；
5. Ray Data 对比；
6. 扩大到 4 GiB / 300 independent clips；
7. SmolVLM caption；
8. Whisper + frame 双分支。

每步先证明：

```text
输出 parity
实际 batch histogram
model worker cumulative time
RPC/task count
GPU bubble/utilization
```

再讨论 wall-time speedup。

## 7. 方案 A 初步结果

H20 ViT batch curve：

```text
batch 1      220 frames/s
batch 4      477 frames/s
batch 8      588 frames/s
batch 16     648 frames/s
batch 32     553 frames/s
```

模型在 batch 16 前有明显收益。

32 logical videos、每视频 7/9 frames、batch cap 16：

```text
                         parent-bound      elastic
V3 wall median           10.847 s          9.919 s
RPC                      96                66
batch fill               47.62%            86.02%
Ray Data wall median     10.204 s         10.262 s
```

该 workload 使用 `model_repeats=10` 表示更深的 frame encoder compute-intensity。
Elastic 相对 parent-bound 约 `1.094×`，packing 改善明显但 wall 收益温和。主要原因：

- video decode/actor startup 仍占比不小；
- 只有一张 GPU；
- 输入是两个公开 AVI 的逻辑重复，不是正式独立视频集。

这证明机制可工作，但正式论文仍需数百独立 clips 和 stage-specific GPU worker time。

## 8. 方案 B 初步结果

SmolVLM-256M greedy caption batch curve：

```text
batch 1       7.6 frames/s
batch 4      16.8 frames/s
batch 8      20.1 frames/s
batch 16     21.6 frames/s
```

8 logical videos、每视频 4 sampled frames：

```text
                         parent-bound      elastic
V3 E2E                   14.84 s           14.48 s
caption RPC               8                 4
all-stage fill           46.15%            85.71%
captions                  exact parity
```

Elastic 减半 caption RPC，但小输入下 startup/decode 占比仍高，wall 仅改善约 2.5%。正式实验
需要更多独立 clips、warmup 和三次重复。

## 9. 方案 A 正式结果

MSR-VTT 1.3GiB / 6,236 independent videos / 76,116 sampled frames / 4×H20：

```text
V3 parent E2E median        767.975s
V3 elastic E2E median       662.528s
Ray Data median             680.955s
elastic vs parent           1.159×
elastic vs Ray Data         1.028×
```

Transform RPC `7,265 → 4,858`，frames/RPC `10.48 → 15.67`，四卡利用率约
88–91%。详细参数和 correctness 见 `2026-08-04_video_a_msrvtt.md`。

## 10. 方案 B 规模结果

MSR-VTT 1.3GiB / 6,236 videos / 23,060 captions / 4×H20：

```text
Parent E2E                 2379.136s
Elastic E2E                2270.577s
speedup                    1.048×
Caption RPC                6,236 → 1,550
Frames/RPC                 3.70 → 14.88
caption mismatch           191/23,060 = 0.83%
```

详细结果见 `2026-08-04_video_b_msrvtt.md`。

## 11. UCF101 4GiB Video A

```text
8,448 independent videos / 240,204 frames / 4×H20
Parent E2E                  2145.538s
Elastic E2E                 2095.791s
speedup                     1.024×
frames/RPC                  14.45 → 15.87
correctness                 structure/digest exact
```

UCF101 parent 本身已接近 batch cap，因此 cross-parent 空间和收益均小于 MSR-VTT。
