# Multigrain V3 论文实验矩阵

## 1. 实验要回答的问题

本轮实验不追求罗列尽可能多的框架，而是用少量、可复现的 workload 分别回答：

1. **性能来源**：提升是否来自跨 parent 的 elastic rebatching，而不只是增大 UDF batch。
2. **系统差异**：相同 `1:M → M:1` 负载在裸 Ray Data 中需要多少显式 lineage、排序和
   regroup 代码。
3. **负载多样性**：该机制是否同时适用于文档 page、文档 region 和视频 frame/clip。
4. **语义价值**：动态 fan-out、ordered Reduce、嵌套 Reduce 和坏记录定位是否在真实负载中
   可观测，而不只是 dummy test。

实验必须分开报告：

- **framework scheduling**：packing、RPC、流水线并行、driver/worker memory；
- **model/application work**：模型吞吐、业务输出和准确性；
- **failure behavior**：定位/隔离额外耗时及成功记录是否需要重算。

不能把不同模型、不同输入或不同有效 model batch 的结果直接称为框架加速。

## 2. 公平对比合同

每个正式 case 至少包含下列模式中的前三项：

| 模式 | 含义 |
| --- | --- |
| `v3_elastic` | V3 Grain 可跨 parent/microbatch 打包；保持 ordered lineage |
| `v3_parent_bound` | 同一 V3 runtime，但重型 Stage 的 batch 不跨 parent |
| `ray_data` | 裸 `ray.data.flat_map → map_batches → groupby/map_groups` |
| `native` | 应用已有的生产/官方 pipeline；存在时才测 |

固定不变量：

- 相同 source 清单和顺序；
- 相同预处理参数、模型权重、精度和 GPU 数；
- 相同 heavy-stage logical batch cap；
- correctness 比较业务输出，不只比较条数；
- model/actor startup 与 measured pipeline wall 分开；
- 预热策略相同；
- 原始 artifacts 写入未跟踪目录，Git 只提交 runner、配置和精简报告。

Ray Data baseline **允许且必须**显式增加：

```text
parent_id
ordinal path
```

否则无法证明 ordered regroup。该字段是 baseline 为补足语义所需的应用代码，应在报告中记录，
不能把它隐藏成 Ray Data 原生 lineage。Ray Data 的 `groupby().map_groups()` 要求单个 group
能放入单节点内存，因此只用于本项目的 bounded document/video group，不据此声称支持无界
group。

最初的 4-PDF CPU smoke 观察到：包含 `PIL.Image` 与页面 metadata 的整体 `page` 列不能
直接编码为 Arrow，Ray Data 会退回 pickled Python-object column。为保证对比公平，正式
baseline 已改为：

```text
image_rgb: Arrow variable-shaped tensor
page_id/scale/width/height/pdf_len: scalar columns
```

优化后的真实 PDF smoke 未再出现 Arrow conversion/object-fallback warning。旧对象布局结果
只作为实现审计记录，不进入正式性能表。

## 3. 统一指标

### 3.1 性能与调度

- measured wall time、items/s、pages/s 或 frames/s；
- heavy-stage RPC/task count；
- logical items/RPC、batch fill ratio；
- tail/recovery RPC fraction；
- heavy actor/GPU bubble ratio；
- in-flight Arena high watermark；
- 端到端 startup time。

### 3.2 内存与控制面

- driver RSS peak；
- worker RSS peak；
- Ray object-store peak（能可靠采到时）；
- V3 live coarse blocks high watermark；
- baseline 中 lineage columns 的序列化字节数或近似值。

### 3.3 正确性和失败

- source 数、leaf 数、最终 group 数；
- 每个 parent 的 ordinal path 是否连续且顺序一致；
- 输出 digest/文本相似度/模型 label 一致率；
- 注入一个确定性坏 leaf 后的 attributed coordinate；
- 正常 run 与定位 run 的额外 wall time。

## 4. Case A：MinerU PDF → Pages → OCR → PDF

### 4.1 目的

这是主性能 case，直接覆盖动态 page fan-out、跨 PDF OCR packing、ordered document Reduce
和 4×H20 persistent actors。

拓扑：

```text
PDF
├── Map(metadata)
└── Expand(render pages)
      └── Map(VLM OCR, heavy)
            └── Reduce(by PDF, ordered pages)
```

### 4.2 输入与规模

- 当前固定语料：Flash-MinerU 仓库中的 `368 PDFs / 7,072 pages`；
- smoke：4 PDFs；
- feasibility：48 PDFs；
- full：368 PDFs；
- 现有 V3 full 结果：`587.781 s` measured wall，`12.0317 pages/s`。

### 4.3 对比

1. `v3_elastic`；
2. `v3_parent_bound`；
3. 裸 Ray Data：
   `from_items → flat_map(render) → map_batches(OCR actor pool) → groupby(parent_id)
   → map_groups(ordered assemble)`；
4. Flash-MinerU `dag_pipeline` native；
5. 可选旧 sequential native，只作为流水线并行消融。

Ray Data OCR 必须保留 `parent_id/page_ordinal`，OCR actor 输出也必须回传它们。正式满载前按
4 → 48 → 368 PDFs 递进，先确认结果与 V3 使用同一 OCR/assemble 逻辑。

### 4.4 关键消融

- skew：按 PDF page count 排序、反序、固定乱序；
- heavy batch cap：16/32/64；
- V3 microbatch：8/16/24；
- in-flight：1/2/3/4；
- GPU：1/2/4；
- 固定 OCR batch cap 比较 elastic 与 parent-bound。

`batch cap=64` 下 V3 elastic 对 parent-bound 的差异是论文中判断“elastic 而非纯大 batch”
的主要证据。

## 5. Case B：嵌套文档解析 Page → Regions

### 5.1 目的

在第二个文档 case 中展示两层动态 fan-out，而不是重复一次 MinerU：

```text
Document
→ Expand(Pages)
→ Map(Layout detection)
→ Expand(Regions)
→ Filter(region kind)
→ Map(OCR/table/picture branch)
→ Reduce(Regions → Page)
→ Reduce(Pages → Document)
```

该拓扑覆盖 V3 的 nested ordinal lineage、branch/Filter 和两次 ordered Reduce。首选真实
Docling 或 PaddleOCR stage；在依赖未确认前，不用 dummy workload 冒充正式结果。

### 5.2 数据候选

首选小规模 smoke：

- `opendatalab/OmniDocBench`：按文件下载 8/32/128 张公开页面图像及对应 JSON；
- 现有 MinerU PDF 中抽取 8/32 个 page，作为无需下载的环境 smoke。

不把整个 `DocLayNet-v1.2` 作为第一轮下载：其 parquet shards 单个约
`267–566 MB`，完整训练集过大。后续需要 layout ground truth 时只取 test shard 或支持
server-side filtering 的小 split。

### 5.3 框架候选与可行性 gate

| 候选 | 当前环境 | Gate |
| --- | --- | --- |
| Docling | 隔离环境已通过 CPU 单 PDF smoke | 拆出可复用 page/region 中间 stage |
| PaddleOCR | 隔离环境可安装/import，单页 pipeline 尚未通过 | 解决官方 GUI OpenCV/libGL 硬依赖后再测 |
| MinerU pipeline backend | 当前 Flash-MinerU VLM 路径可用 | 明确其内部 stage 与 V3 对接粒度 |

选择规则：

1. 能以 callable stage 暴露 page/region 中间结果；
2. 模型能被 persistent actor 初始化一次；
3. 32-page smoke 在当前 4×H20 节点可稳定复现；
4. 不需要修改第三方框架 correctness 逻辑；
5. 如果 Docling/PaddleOCR 只能提供整体黑盒 API，则把它作为 `native` baseline，不伪装成
   V3 多 Stage 实现。

隔离环境实测记录：

- Docling `2.117.0` 支持 Python 3.12；默认安装拉入 GUI OpenCV，在当前无 `libGL.so.1`
  容器中会失败。移除 GUI wheel、保留 headless OpenCV 后，现有
  `1838_reformer_the_efficient_transfo.pdf` CPU 转换成功：
  `23.112 s / 39,775 Markdown chars`。因此 Docling 是当前优先文档候选。
- 页面级 V3 prototype 已实际跑通同一 12-page PDF：

  ```text
  PDF → Expand(PNG pages) → Map(Docling image converter) → Reduce(Markdown)
  ```

  `page_batch_size=4` 时 wall `93.570 s`、`5 RPC`、平均 `2.8 grains/RPC`、
  batch fill `93.33%`，输出 `12 pages / 38,881 chars`。`page_batch_size=1` 的前一轮为
  `97.402 s / 14 RPC`。这只证明 page-level persistent actor、packing 与 ordered
  Reduce 可行；Docling CPU 内部仍主要逐 page 计算，当前不能声称有显著速度提升。
- 同一 page-level workload 的裸 Ray Data baseline 已实现并实际对比：

  ```text
  V3 wall             94.148 s
  Ray Data wall      114.426 s
  Markdown Jaccard     1.000
  V3 RPC                   5
  V3 grains/RPC        2.800
  ```

  两边复用相同 PDFium render、Docling `convert_all` 和 Markdown assembly。该数据仍是
  单文档、单次、CPU smoke，主要证明公平对比链路和 exact correctness；还不能作为正式
  speedup。
- Docling 的 `PictureItem/TableItem.get_image()` API 确实可表达 Page→Region 第二层
  fan-out，但在当前论文 PDF 前两页和一个公开 cTDaR table-detection image 上，模型均未
  产生 Picture/Table item。因此暂不把二层 region case 写成“已验证”，要先选择能稳定触发
  region 的公开样本。
- PaddleOCR `3.7.0` + Paddle CPU `3.3.1` 支持 Python 3.12，headless OpenCV 下 import
  成功；但 PaddleX OCR extra 将 `opencv-contrib-python==4.10.0.84` 写成硬依赖并按
  distribution name 检查。该 GUI wheel 在当前容器缺少 `libGL.so.1`，替换成 headless
  wheel后又无法通过 PaddleX extra 检查。因此本轮不把 PaddleOCR 当作已可运行 case。

## 6. Case C：Video → Frames/Clips → Video

### 6.1 目的

证明 `1:M` 不依赖 PDF 数据类型。首个视频 case 保持模型轻量，先测 scheduling，再决定是否
加入昂贵 VLM/ASR：

```text
Video
→ Expand(sampled frames)
→ Map(image embedding/classification, heavy)
→ optional Filter
→ Reduce(ordered frames → video summary)
```

第二阶段可扩展为：

```text
Video → Clips → Frames → frame model → Clip Reduce → Video Reduce
```

### 6.2 数据与 smoke

- 连通性 smoke：`hf-internal-testing/tiny-video-dataset`，1 个约 `266 KB` MP4；
- 实验小集：`sayakpaul/ucf101-subset`，约 `172 MB`，另含两个独立 AVI 示例；
- skew 扩展：对公开 clips 采用不同采样 stride/截断长度，产生可控的 4–128 frame fan-out。

当前环境已有 OpenCV，可先用 `cv2.VideoCapture` 解码；`decord`、PyAV、MoviePy 未安装。
正式性能实验要先验证 OpenCV 能解码选定 MP4/AVI，并把 decode 与模型 Stage 分开计时。

### 6.3 heavy stage

优先级：

1. 本地或可缓存的小型 image classifier/encoder；
2. 若无合适本地权重，先下载一个固定公开模型并记录 revision；
3. 不用 `sleep` 作为论文正式性能数据，只保留为调度回归测试。

Ray Data baseline 使用 `flat_map(frames)` 和 `map_batches(model)`，再按
`video_id/frame_ordinal` regroup。正式结论必须同时给出均匀 fan-out 与 skew fan-out。

预训练 ResNet18 baseline 的 correctness signature 使用 top-5 class IDs，而不 hash 原始
float logits。CPU GEMM 在不同物理 batch shape 下可能产生末位浮点差异；把 raw bytes 当作
业务语义会错误地把合法 rebatching 判为输出变化。top-5 classes 仍是模型业务输出，并满足
本实验所需的 grain-separable parity 检查。

UCF101 两视频的 ResNet18 feasibility：

```text
source frames                 170 / 125
stride                        20
sampled leaves                 9 / 7
model              ImageNet ResNet18
batch cap                     16

V3 elastic RPC                 4
V3 elastic grains/RPC          5.00
V3 elastic fill               90.91%

V3 parent-bound RPC            6
V3 parent-bound grains/RPC      3.33
V3 parent-bound fill          47.62%

V3/Ray Data top-5 parity       exact
```

单次 wall 中 elastic `2.987 s`、parent-bound `2.689 s`，输入太小且 actor/model startup
占主导，不能据此声称 elastic 更慢或更快。该消融目前只可靠证明跨视频 rebatching 将模型
RPC 从 6 降为 4，并把 fill 从 47.62% 提高到 90.91%。正式 wall-time 结论需要更多视频、
预热和重复。

把同一两个公开 AVI 重复成 8 个独立 logical videos 后：

```text
                         elastic    parent-bound
V3 wall                    3.994 s       3.845 s
all-stage RPC                   18            24
all-stage grains/RPC         4.444         3.333
all-stage fill              74.07%        47.62%
Ray Data wall               5.600 s       6.286 s
semantic parity              exact         exact
```

这里 `all-stage RPC` 包括 render/reduce，不是纯 ResNet RPC；仍是单次且重复输入，所以只作为
调度机制 smoke。它继续支持“elastic 改善 packing/RPC”的结论，但 wall 差异仍不足以写成
正式性能结论。

在同一 8-video 配置下增加 `1 warmup + 3 measured paired trials`：

```text
                         elastic             parent-bound
V3 wall                  4.248 median        3.975 median
Ray Data wall            5.587 median        4.278 median
paired V3/Ray Data       1.341× median       1.076× median
all-stage RPC            18                  24
all-stage fill           74.07%              47.62%
semantic parity          exact               exact
```

这个结果可以支持“V3 对该小型 real-model workload 的控制/数据组织开销低于裸 Ray Data”
以及“elastic 显著改善 packing”。但 elastic 的 V3 wall 没有优于 parent-bound：当前 CPU
ResNet18 对 batch size 8–16 的吞吐收益不足以覆盖等待/大 batch 代价。因此该 case **不能**
被用于声称 elastic 普遍提速；它是一个重要的负结果，说明论文应将收益条件明确限定为
batch-efficient、昂贵的 GPU/model stage，并用 MinerU 主实验验证。

## 7. 实现目录

不把所有 runner 继续堆进单个 `mineru.py`。新增 case 采用：

```text
rayorch/experimental/multigrain_v3/benchmark/
├── common/
│   ├── artifacts.py        # JSONL、timeline、RSS/GPU observation
│   └── lineage.py          # baseline parent/ordinal helpers
├── mineru/
│   ├── v3.py
│   ├── ray_data.py
│   ├── native.py
│   └── configs/
├── document_nested/
│   ├── v3.py
│   ├── ray_data.py
│   ├── native.py
│   └── data.py
└── video/
    ├── v3.py
    ├── ray_data.py
    ├── native.py
    └── data.py
```

现有 `benchmark/mineru.py` 在新 baseline smoke 通过前保持兼容，不为目录美化先重构。

测试：

```text
test/experimental/multigrain_v3/benchmark/
├── test_ray_data_lineage.py
├── test_document_nested_data.py
└── test_video_data.py
```

- deterministic 数据和 lineage tests 使用 pytest；
- 需要 Ray 的 4–16 item smoke 使用 pytest integration；
- 真实模型、下载和 4×H20 full run 使用手工 CLI，不进入默认 pytest。

实验记录：

```text
docs/experiments/multigrain_v3/
├── experiment_matrix.md
├── mineru/
├── document_nested/
└── video/
```

## 8. 执行顺序与 go/no-go

1. **MinerU Ray Data correctness smoke（4 PDFs）**
   ordered regroup 与 V3 输出匹配后才跑 48 PDFs。
2. **MinerU Ray Data feasibility（48 PDFs）**
   无 driver OOM、actor pool 可复用、OCR batch 实际达到目标后才跑 368 PDFs。
3. **MinerU full matrix**
   至少 V3 elastic/parent-bound/Ray Data/native 各三次，报告中位数与离散度。
4. **HF video decode smoke**
   下载 tiny MP4，验证 frame count、ordinal 和 OpenCV decode。
5. **Docling/PaddleOCR 隔离环境 smoke**
   先记录安装解和单页结果，再选一个作为正式 nested document case。
6. **一个新 case full run**
   优先 video（依赖更少）；文档框架 smoke 成功后补 nested document。

当前已知：

- Ray `2.50.0`、Ray Data `flat_map/map_batches/groupby` 可用；
- Hugging Face 元信息访问可用；
- `datasets 4.0.0`、OpenCV `4.13.0` 已安装；
- tiny MP4 已实际下载并由 OpenCV 解码：`49 frames / 8 FPS / 720×480`，以 stride 6
  得到 8 个连续 logical ordinals；
- V3 与裸 Ray Data 已在两个不同长度 MJPG 视频上跑通相同 workload：
  stride 2 后分别产生 `3` 和 `6` 个 frame leaves，ordered summary 完全一致；
- Hugging Face `sayakpaul/ucf101-subset` 的两个独立 AVI 也已跑通 compare CLI：

  ```text
  source frames        170 / 125
  stride               10
  sampled leaves       16 / 13
  V3 wall              1.383 s
  Ray Data wall        3.478 s
  V3 grains/RPC        4.125
  V3 batch fill        78.57%
  output parity        exact
  ```

  这是轻量 CPU transform 的单次 feasibility run，只证明公开数据、动态 fan-out 和两套
  runner 的公平 parity；未预热、未重复，也没有 GPU model，因此不能作为论文 speedup
  数字。
- MinerU Ray Data columnar CPU smoke 已实际跑通：`4 PDFs / 48 pages / 4 ordered
  groups`；未加载 vLLM 的单次 measured wall 为 `17.331 s`，该数字包含首次 actor/codec
  路径，只证明数据链路，不作性能比较；
- MinerU Ray Data 真实 vLLM：

  ```text
  4 PDFs / 48 pages
  wall                     61.666 s
  V3/Ray Data Jaccard      0.9946 median

  48 PDFs / 992 pages
  wall                    165.419 s
  pages/s                   5.997
  V3/Ray Data Jaccard      0.9952 median
  all 48 documents >= 0.98
  driver RSS peak         638 MB
  ```

  48-PDF Ray Data stats 显示 14 个 OCR tasks、约 60–72 pages/task，但后续
  `groupby(parent_id)` 对约 11.1GB page/image columns 做 shuffle，shuffle 总阶段约
  `155 s`，其中 finalize blocks 约 `2.5–3.1GB`。这说明裸 Ray Data 能获得大 OCR batch，
  但为恢复 ordered document group 复制/重分区了大 image payload；V3 Reduce 依赖 lineage
  receipt，无需全量业务 payload hash shuffle。该差异是本论文系统价值的重要证据。
- Docling/PaddleOCR 只安装在 `/tmp` 隔离环境，未污染仓库主 Python 环境；
- decord、PyAV 未安装；
- 受限 shell 内 `nvidia-smi` 不可见；通过沙箱外执行已确认 4×H20 可用并完成 full runs。

这些是 feasibility 结论，不是性能结果。

后续通过沙箱外 GPU 执行确认 4×H20 可见，并完成 368-PDF 四系统各一轮。完整结果见：

```text
2026-08-01_full_system_comparison.md
```
