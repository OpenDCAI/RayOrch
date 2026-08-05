# Multigrain V3 论文实验索引

本文是论文写作的唯一论点/配置入口；详细运行记录保留在分项文档，不在这里重复。

## 1. 核心论点

### MinerU：elastic rebatching 主证据

```text
PDF → dynamic Pages → GPU OCR → ordered PDF
```

同一 V3 runtime、模型、batch cap 和输入，仅切换：

```text
parent-bound vs cross-parent elastic
```

现有 368-PDF 证据：

```text
V3 elastic measured        587.781s
V3 parent-bound measured   795.071s
speedup                      1.353×
OCR pages/RPC              58.93 vs 19.22
OCR RPC                      120 vs 368
```

它是 elastic 净收益的主图。详见 `2026-08-01_full_system_comparison.md`。

### Ray Data：自然模型与专家优化

三条 baseline：

```text
Ray Data full-value groupby
Ray Data expert/reference-only
V3
```

368-PDF：

```text
full-value Ray Data         982.549s, shuffle ~79.9GB
reference-only Ray Data     889.164s, shuffle ~1.13MB
V3 elastic                  626.890s
```

reference-only 需要应用手写 owner actor、block token、row selector、consumer credit 和
release；V3 将其变成 lineage/failure-aware runtime contract。详见
`ray_data_reference_baseline.md`。

### Docling：core-stage disaggregation

不是 per-page `DocumentConverter`，而是复用：

```text
PDF backend/preprocess
LayoutModel
OcrAutoModel
LayoutPostprocess
TableStructureModel
PageAssembleModel
ReadingOrderModel
```

12 个多样短 PDF / 43 pages / 单 H20 初步结果：

```text
Native default             35.49s
V3 parent-bound            22.11s
V3 elastic                 21.88s
Markdown Jaccard           12/12 = 1.0
```

当前收益主要是跨文档 stage pipeline、persistent actors 和 parse/OCR data parallelism，不是
已证明的 elastic 净收益。正式矩阵必须包含：

```text
Native default
Native tuned doc concurrency
V3 parent-bound
V3 elastic
```

最新 tuned-native 结果：

```text
Native concurrency 1     33.33s
Native concurrency 2     22.75s
Native concurrency 4     18.76s
V3 parent-bound          22.11s
V3 elastic               21.88s
```

因此 Docling 当前只作为架构/集成 case，不能写成 V3 性能胜出。

后续专项优化修正了 eager 3-scale page image 传输，并保留原生高分辨率 render。旧的
单轮 `23.77s vs 17.08s` 只作为优化过程记录，不进入主表，因为 cache/startup 口径不够稳定。

```text
Native: 1 layout / 1 OCR / 1 table，最大并发4/4/2
V3:     1 layout / 1 OCR / 1 table actor，并发4/4/2
batch cap: layout/OCR/table 均为4
```

2026-08-02 严格模型实例、batch cap、输入和 concurrency 对齐后的 3-run 结果：

```text
Native measured median          17.495s
V3 parent-bound median          13.208s
steady speedup                   1.325×
V3 parent-bound E2E median      19.559s
startup                         Native 6.303s；V3约6.45s
Markdown/structure parity       exact on 12/12 docs
```

V3 每次 run 仍重建 actors，所以 first-run E2E 不快于 warm Native measured。论文同时报告
startup-inclusive 和 post-readiness，不混用。

Elastic 消融：

```text
parent-bound measured median    13.208s
elastic measured median         13.866s
RPC / fill                      72 / 0.671 → 64–65 / 0.750–0.773
```

elastic 提高 fill、减少 RPC，但此 workload 上慢约 5%；Docling 不能作为 elastic 正收益
证据。当前主要专项短板是 per-run actor/model startup 和 immutable DTO transport，不是
driver commit。

详见 `docling_core_stage_feasibility.md` 和 `docling_internal_stage_design.md`。

#### Docling 当前未完成项（2026-08-03）

为避免把小规模调试结果写成最终结论，以下仍是论文前 gate：

```text
[ ] 单一四臂 CLI：Native default / Native tuned / V3 parent-bound / V3 elastic
[ ] 固定 input manifest、统一 JSON 指标 schema、统一 correctness gate
[ ] 更大公开多样 PDF corpus，1 warmup + 3 measured
[ ] V3 reusable ExecutionPool，用于区分 cold-start 与 persistent-service 口径
[ ] elastic 仅在出现正净收益时作为 Docling 主论点；否则报告负结果
```

2026-08-03 已验证统一四臂 runner 的端到端 smoke；其单次结果仅证明 runner 和 strict
correctness gate 可用，不替代 `warmup=1 + repeats=3` 的正式统计。

下一项主实验已冻结为 368 PDF / 7,072 pages 的独立进程四臂矩阵，使用四次 Latin-rotation
repeats、逐臂即时落盘、CV 和同 repeat paired speedup。详细输入 identity、命令和结论 gate
见 `docling_368_experiment_plan.md`。

48-PDF 四卡配置 sweep 已冻结 `1 Layout GPU + 3 Table GPUs, cap=8`。两轮 balanced
parent/elastic paired speedup 为 `1.084× / 1.054×`，median `1.069×`，且 48/48 strict
parity。该结果只作为 368 full 的 entry gate，不写入最终主表。

368-PDF 四卡正式结果见 `2026-08-03_docling_368_four_gpu.md`。完成四组无污染的
2×2 balanced pairs 后，elastic 4/4 次慢于
parent，paired speedup median `0.943×`；但 RPC/fill 稳定改善到约 `884 / 0.987`。原因是
Docling 只有 Layout 使用真实 `predict_batch`，RapidOCR/Table 内核仍逐 page/table。
正式论文将其报告为“rebatching mechanism succeeds but model is not batch-efficient”的
负结果，不使用受外部4卡任务污染的首轮正收益。

```text
Native default              1006.062s（1 run）
Native tuned                 824.220s（2 runs, CV 0.49%）
V3 parent                    880.583s（4 clean runs）
V3 elastic                   944.427s（4 clean runs）
```

### Video：三类多模态验证

```text
A. Video → Frames → GPU ViT → Video
B. Video → Frames → SmolVLM captions → Video
C. Video → Audio chunks/Frames → Whisper/ViT → aligned merge
```

当前均已完成 smoke；正式 A/B 使用几 GB 公开独立视频集和长尾 frame counts。详见
`video_three_case_plan.md`。

Video A 已完成正式 MSR-VTT 规模实验：

```text
1.3GiB / 6,236 independent videos / 76,116 sampled frames / 4×H20
V3 parent median          767.975s
V3 elastic median         662.528s
Ray Data median           680.955s
elastic vs parent         1.159×
elastic vs Ray Data       1.028×
```

详见 `2026-08-04_video_a_msrvtt.md`。

Video B 在同一 1.3GiB manifest 上完成单组 SmolVLM 四卡 pair：

```text
Parent E2E             2379.136s
Elastic E2E            2270.577s
speedup                1.048×
Caption RPC            6236 → 1550
caption text drift     0.83%
```

由于只有一组 pair 且 generation 文本受 batch shape 影响，Video B 仍定位为 feasibility；
详见 `2026-08-04_video_b_msrvtt.md`。Video C 仍是架构 smoke。

数 GB scale gate 已由官方 UCF101 完成：

```text
4.000GiB / 8,448 videos / 240,204 frames / 101 classes
V3 parent E2E          2145.538s
V3 elastic E2E         2095.791s
speedup                1.024×
```

详见 `2026-08-04_video_a_ucf101_4g.md`。其收益小于 MSR-VTT，因为 parent frames/RPC
本身已达14.45/16，剩余rebatching空间有限。

#### Video 当前未完成项（2026-08-03）

```text
[ ] 选择许可清晰、可公开下载的数 GB 视频数据集
[ ] 固定 clip manifest 和长尾 frame/audio-duration 分布
[ ] A/B 至少数百独立 clips 的 full benchmark
[ ] 记录下载校验、抽样规则、模型 revision、GPU/CPU 资源和完整日志
```

## 2. 公平配置合同

所有对比固定：

- 相同有序 source list；
- 相同模型 revision/权重；
- 相同 GPU 数和每 Stage 资源；
- 相同 heavy-stage batch cap；
- startup-inclusive E2E 为跨系统主口径；
- parent/elastic 只改变 batch scope；
- correctness 比较最终业务输出和结构，不只比较数量；
- 至少 1 warmup + 3 measured runs 才进入正式表。

每次保存：

```text
wall/startup
stage RPC/tasks
items per heavy RPC
batch fill
worker cumulative time
GPU bubble/utilization
driver/worker/object-store memory
correctness
```

## 3. 数据和规模

```text
MinerU      368 PDFs / 7,072 pages
Docling     公开/可复现多样 PDF shards；正式规模待 tuned-native 后冻结
Video A/B   几 GB 主流公开视频集；至少数百独立 clips
Video C     带音频公开视频；固定时间窗对齐
```

禁止把复制同一 PDF/video 的结果作为正式 diversity 数据；复制只用于调度 smoke。

## 4. 结论边界

- MinerU 已支持 elastic 性能结论；
- Ray Data 支持“自然写法代价 + 专家可优化 + 编程模型复杂度”结论；
- Docling 当前支持 V3 core-stage 系统加速潜力，不支持 elastic 净收益结论；
- 视频 A 有温和 elastic 收益，B/C 当前是 feasibility；
- 不声称通用 bounded-memory theorem 或所有模型都从大 batch 获益。
