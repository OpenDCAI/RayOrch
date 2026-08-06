# Docling core-stage V3 重组设计

## 1. 目标

目标不是把每个 PDF page 当成独立 `DocumentConverter.convert(image)` 调用。该写法会重复
创建 image document context，并绕过 Docling 已有的 PDF backend、segmented cells 和内部
模型 batch 接口。

正确目标是：

```text
复用 Docling 数据模型与模型权重
抽取 preprocessing/layout/OCR/table/page-assemble 核心 stage
由 V3 跨 documents 组织 page/region batches
复用 Docling reading-order/document assembly
与原生 StandardPdfPipeline 做端到端对比
```

不 fork 或修改已安装 Docling；adapter 只调用现有公开类或稳定模型方法。

## 2. 原生 Docling 现状

`StandardPdfPipeline` 的主要 page stages：

```text
PDF backend
→ PagePreprocessingModel
→ LayoutModel
→ OcrAutoModel / RapidOcrModel
→ LayoutPostprocessingModel
→ TableStructureModel
→ PageAssembleModel
→ ReadingOrderModel
```

原生每个 document 创建一套 threaded queues。`ThreadedPipelineStage._process_batch()` 会先按
`run_id` 分组，再调用模型，因此 page batch 不跨 documents。

默认：

```text
layout_batch_size    4
ocr_batch_size       4
table_batch_size     4
doc_batch_size       1
```

对当前 12-page PDF 的官方 profiling：

```text
wall                     23.28 s
pipeline_total           18.27 s
page_parse                1.16 s
layout                   11.42 s
OCR                       5.75 s
layout postprocess        0.22 s
table structure          10.80 s
page assemble             0.006 s
document assemble         0.15 s
```

stage timings 因流水线重叠不能直接相加。最值得优化的是 layout 和 table；OCR 次之。

## 3. V3 数据边界

### 3.1 Source/Expand：Document → Page

Driver-side PDF adapter 使用 Docling 的：

```text
InputDocument
DoclingParseDocumentBackend
PdfPageBackend
Page
ConversionResult
```

为每个 PDF 创建一个 document context，并把每个 page 规范化为可序列化
`DoclingPageState`：

```text
DoclingPageState
├── document_key
├── page_no
├── size
├── image_72dpi
├── image_ocr_scale（按需）
├── segmented_page
└── page confidence/predictions
```

不跨 actor 传 `PdfPageBackend`。backend 只在 source/preprocess adapter 内存在；需要的
segmented cells、bitmap rects 和 page images 被提取成值。

### 3.2 Layout Map

每个 persistent actor 持有 Docling `LayoutModel` 或其
`layout_predictor`。

核心批处理接口：

```python
layout_predictor.predict_batch(list[PIL.Image])
```

输出转换回 Docling：

```text
LayoutPrediction
Cluster
BoundingBox
DocItemLabel
```

V3 可跨 PDF 把 pages 合成比原生 `layout_batch_size=4` 更大的 batch。首轮测试 batch
curve：

```text
1 / 2 / 4 / 8 / 16 pages
```

如果 GPU/CPU layout model 在 4 以上无吞吐收益，就不宣称 elastic layout 加速。

### 3.3 OCR Map

当前默认 `OcrAutoModel` 在该环境选择 RapidOCR/ONNX Runtime。其 `__call__` 内仍逐
page、逐 OCR rect 调用 reader，没有跨 page batch inference。

首版保持 Docling 的：

```text
get_ocr_rects
RapidOCR reader
post_process_cells
```

但只把 OCR 放到独立 persistent actor pool，获得 stage pipeline overlap。不能预先宣称
OCR elastic batching。

后续若 RapidOCR backend 暴露真正 batch API，再单独增加，不在 adapter 中伪造。

### 3.4 Layout postprocess Map

复用：

```text
LayoutPostprocessingModel
```

它依赖：

- raw layout clusters；
- segmented text cells；
- page size。

属于轻量 CPU stage，可以与 OCR 的结果对齐后执行。为了保持原生语义，stage 顺序仍为：

```text
layout prediction
→ OCR 更新 cells
→ layout postprocess/cell assignment
```

### 3.5 Table：两种实现阶段

#### 第一阶段：Page Map

直接复用 `TableStructureModel.predict_tables()`，验证端到端 parity。

#### 第二阶段：Page → Table regions → Page

从 layout clusters 中 Expand：

```text
TABLE / DOCUMENT_INDEX clusters
```

每个 table record 包含：

```text
cropped table image
cluster
tokens
page coordinate transform
```

table actor 调用 Docling 已有 TableFormer 内核：

```text
TFPredictor.multi_table_predict
```

随后按 page Reduce 恢复 `TableStructurePrediction.table_map`。

这是真正的第二层动态 fan-out：

```text
Document → Pages → Tables → Pages → Document
```

同时允许不同 page/document 的 table regions 跨 parent 组 batch。前提是先验证
`multi_table_predict` 能把多个 table boxes 放进同一次模型调用；若内核仍逐 table，则只展示
pipeline parallelism，不声称 table rebatching 收益。

### 3.6 Page/Document assembly

复用：

```text
PageAssembleModel
ReadingOrderModel
HeadingHierarchyModel
```

Page Reduce 按 page ordinal 恢复 pages；Document Reduce 构造与原生一致的
`ConversionResult`，再执行 reading order 和 Markdown export。

最终 correctness 比较：

- document count；
- Markdown token Jaccard；
- DoclingDocument JSON structure；
- tables/pictures/text item counts；
- page provenance。

## 4. 公平对比

### Native Docling

使用同一版本、模型、device、线程数和 pipeline options：

```text
StandardPdfPipeline
```

### V3 core-stage

必须保持：

- 相同 PDF backend；
- 相同 layout/OCR/table 权重；
- 相同 model options；
- 相同 image scales；
- 相同 reading-order/assembly；
- 相同输出 correctness gate。

只改变：

- page/region 作为 Logical Grain；
- 模型 actors 的 persistent 生命周期；
- 跨 documents elastic batching；
- bounded multi-Arena overlap。

### 消融

```text
Docling native
V3 parent-bound
V3 elastic
```

如果只比较 native 与 elastic，无法把收益归因到跨 document rebatching。

## 5. 实施顺序

1. 单 PDF、12 pages：core-stage adapter 与 native structure parity；
2. 4–8 PDFs：layout batch curve，确认模型是否 batch-efficient；
3. V3 parent-bound 与 elastic：相同 stage actors 和 batch caps；
4. 加入 Table region Expand/Reduce；
5. 公开多文档数据集 full run；
6. 最后再加入 picture classifier/VLM description 分支。

首个 gate 不是“比 native 快”，而是：

```text
不调用 per-page DocumentConverter
核心模型各初始化一次
DoclingDocument parity 达标
layout/table batch 确实跨 document
```

只有 gate 达成后，wall-time 比较才有意义。

Elastic 实验必须同时报告 `layout_batch_wait_ms`。默认 V3 的 2ms tail timer 对 PDF parse
这种秒级 producer 太短，会导致 page 一到就 dispatch，表面配置 elastic 但实际上仍接近
每文档一次 RPC。首轮 sweep：

```text
2 / 20 / 50 / 100 ms
```

以 layout RPC、pages/RPC、fill 和 wall 共同选择，不只按 wall 调参。

仅增加 layout wait 仍可能无法跨 document 聚合：若上游 PDF Expand
`parse_batch_size=1`，每次 completion 只发布一个 parent 的 pages，且不同 PDF parse
completion 间隔远大于 layout wait。应同时 sweep：

```text
parse_batch_size     1 / 2 / 4
layout_batch_size    8
```

parse batch 是普通 Stage coarse RPC 配置，不是 Docling-specific 调度飞线。

## 6. 当前审计/原型证据

### 6.1 原生 stage profiling

当前 12-page PDF、CPU：

```text
native E2E               23.28 s
layout total             11.42 s
table total              10.80 s
OCR total                 5.75 s
```

layout/table 是主要优化对象。

### 6.2 Layout batch efficiency

CPU layout batch 越大吞吐越低：

```text
batch 1     1.39 pages/s
batch 4     1.22 pages/s
batch 8     1.02 pages/s
batch 12    0.95 pages/s
```

H20 GPU：

```text
batch 1     37.5 pages/s
batch 4     64.8 pages/s
batch 8     70.0 pages/s
batch 12    63.6 pages/s
```

因此 Docling V3 的潜在增速来自：

- GPU layout actor；
- 多文档 pages 填满 batch 8；
- 多 GPU replicas；
- OCR/Table/parse 与 layout 的 pipeline overlap。

CPU elastic layout 不应作为性能主实验。

公平的单 H20 V3 配置：

```text
Layout actor       device=cuda, num_gpus=0.5
Table actor        device=cuda, num_gpus=0.5
RapidOCR actor     device=cpu,  num_gpus=0
```

这样 layout/table 与原生一样共享一张物理 GPU；不能给 V3 两张 GPU 再对比 native
一张 GPU。

### 6.3 原生 batch 边界和数值稳定性

原生单文档真实 layout batches 不是固定 `4/4/4`，而是受 producer/queue 时序影响：

```text
1 / 4 / 4 / 3
```

同一个 page 在不同 batch peers 下可能产生不同 cluster 数。因此：

- page-level Layout Map 的 cardinality 仍是稳定的 1:1；
- layout prediction value 允许数值小差异，以最终 Markdown/structure gate 验证；
- 不能未经稳定化就把 raw layout clusters 直接当作 Expand cardinality。

Table region 二层 fan-out 必须在 layout 输出 canonicalization 方案冻结后再做。

### 6.4 Core-stage prototype

当前已实现：

```text
Docling PDF backend/preprocess
→ immutable DoclingPageSource
→ LayoutPrediction value Map
→ OCR segmented-page value Map
→ postprocess/table/page-assembly value Map
→ document Reduce/ReadingOrder
```

Stage 之间不传可变 Page，不共享 mutable predictions；每个 actor 从 immutable source/value
重建临时 Page，符合 grain-separable 边界。

单 PDF CPU prototype：

```text
V3 E2E                  33.48 s
native E2E              23.5 s
Markdown chars          39,771 vs 39,775
tables                  5 vs 5
pictures                5 vs 5
texts                   351 vs 354
```

剩余轻微结构差异来自 layout batch 时序/数值差异，需要以多文档 correctness 分布评估，而
不是要求 bitwise equality。

## 7. Table / OCR 的安全批处理演进

2026-08-04 对当前依赖版本的源码审计：

```text
RapidOCR 3.9.2
Docling IBM models 3.13.3
```

### 7.1 TableFormer

当前 `TableStructureModel.predict_tables()` 是：

```text
for page in pages:
  for table in page.tables:
    tf_predictor.multi_table_predict(page_input, [one_box])
```

而 `multi_table_predict()` 内仍对每个 box 调：

```text
predict(..., table_image)
```

且 `_prepare_image()` 固定构造 batch=1。底层 encoder 能接收 batch dimension，但当前
`TableModel04_rs.predict()` decoder 使用 scalar `.item()`，不是安全的 batch>1 decode。

因此 P0 不直接冒充“stack 后调用当前 predict 即可加速”。正确分层：

```text
Page/Table cluster
→ TableJob(page/table coordinate, crop, tokens)
→ stable TableJob batch planner
→ verified batched kernel / reference kernel
→ semantic compare
→ per-job batch=1 fallback
→ restore Page TableStructurePrediction
```

V1 默认仍走 reference kernel，历史 `decoder_accelerated` 路径和 368-PDF 证据保持不变。
当前新增的 V2 不修改 V1，而是独立实现：

```text
TableJobV2
→ public encode_images / forward
→ per-row EOS batch decoder
→ lineage bbox split
→ deterministic OTSL/cell postprocess
```

它不访问 V1 的 `_prepare_image/_encoder/_tag_transformer`，也不 monkeypatch model method；
只有通过 `OTSL/cells/spans/bbox/Markdown` gate 后，`v2_batch` 才能作为显式实验配置使用。

### 7.2 RapidOCR

RapidOCR 内部 classifier/recognizer 已对**一个 rect 内**的 crops 使用 `cls_batch_num` /
`rec_batch_num`，但 Docling 外层仍：

```text
for page:
  for ocr_rect:
    rapidocr(rect_image)
```

P1 保持 detector 不变，只将 detector 产出的 text crops 做跨 rect/page recognition batch。
早期低层 kernel 版本物化归一化 tensor 并直接调用 session；当前生产 adapter 只保留
normalized-width compatibility key，模型执行仍调用 RapidOCR facade：

```text
crop
→ stable cross-page gather
→ derive normalized width（不物化 tensor）
→ recognize_txt(list[compatible crops])
→ shadow 单条 compare
→ per-job fallback
→ original Docling post_process_cells
```

V3 只读取 `text_rec.rec_image_shape` 这项版本钉死的只读 compatibility metadata，避免
极端长宽比 crop 放大同一 RapidOCR minibatch；不访问 session、预处理或 decoder。
RapidOCR 自己完成排序、`rec_batch_num` 切分并恢复输入顺序。文本严格比较，score 使用
显式 tolerance；shadow 中任何 decode 漂移默认回退原单条结果。

V3 的生产 adapter 直接持有 `RapidOcrModel`，不再经由
`OcrAutoModel._engine`。跨 rect/page gather 以后直接调用 RapidOCR 已有的：

```text
preprocess_img / detect_and_crop / cls_and_rotate
→ recognize_txt(list[crop])
→ build_final_output
→ Docling post_process_cells
```

每个 compatibility group 调用一次 recognizer facade；RapidOCR 自己按宽高比排序，并按
`rec_batch_num` 构造 minibatch。V3 只拥有跨 page 的 gather、lineage scatter 和
fallback；recognizer 的 resize/normalize、session、CTC decode、RTL 恢复和结果对象
构造继续由 RapidOCR 实现。这样 UDF 合同不会绑定 V3 的
`Pipeline/Map/Expand/Reduce` 写法，迁移到 `RayModule + F.*` 时只需改 authoring/lowering。

### 7.3 输出合同

“一致”冻结为：

```text
OCR: page/rect/crop order、text、cell structure exact；
     score/bbox 记录 tolerance；不稳定 job fallback。

Table: page/table cluster binding、OTSL、cell text/span/count exact；
       bbox tolerance；不稳定 job fallback。
```

不承诺不同 GPU batch shape 下的模型 float score bitwise 相同。

### 7.4 当前实验开关

adapter 公开两个默认关闭的实验开关：

```text
table_batch_mode:
  reference        默认，原 TableStructureModel
  encoder_shadow   batch TableFormer encoder + 原单表 decoder + reference fallback
  v2_batch         独立 TableFormerV2 batch decoder 与 TableJobV2 contract

ocr_batch_mode:
  reference            默认，原 RapidOCR rect 路径
  recognition_shadow   detector/classifier 原样 + compatible recognizer shadow
  recognition_accelerated  同一 facade/kernel，不重复执行 reference
```

两个 shadow 模式首先是 correctness/audit 工具，不是性能开关：

```text
candidate batch
→ independent reference
→ semantic compare
→ per job / per rect fallback
```

在严格 shadow 打开时通常不会产生净加速，因为 reference 仍会执行。只有收集充分
`exact/fallback fraction/Markdown` 数据并冻结可接受风险后，才可引入明确命名的
non-shadow accelerate mode；该模式不应默认开启。`v2_batch` 与
`recognition_accelerated` 只在独立实验配置显式启用；其当前实现、证据和全量 gate 见
`2026-08-05_docling_direct_v2.md`。

full368 gate 的最终边界是：direct RapidOCR + V1 在前 48 PDF 对历史 artifact 48/48
Markdown byte-exact；TableV2 的真实 batch 与 singleton 也 byte-exact，证明 adapter/kernel
合同成立。但 V2 full368 比 V1 median 慢 10.09%，且复杂表出现 512-token runaway 与明显
行合并，所以 `v2_batch` 保持 opt-in，不能替代 V1 golden。这个结果说明“可替换 kernel”
不等于“任一新模型自动晋升”；模型质量 gate 与 batch 抽象 gate 必须分开。

### 7.5 跨版本不变量

Docling workload 固定为以下 UDF 数据边界：

```text
Document → Page → OCR/Layout values → TableJob → Page → Document
```

V3 使用 `Pipeline + Expand/Map/Reduce` 表达；V3.1+ 可用 `RayModule + F.*` 表达，但
Page/TableJob DTO、模型 adapter、batch kernel、lineage key 和输出合同不应随 authoring
API 复制一份。TableFormer V1/V2 也只能替换 `TableJob → Table` kernel，不改变图结构。
