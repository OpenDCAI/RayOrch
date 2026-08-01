# Multigrain V3 多负载可行性实验（2026-08-01）

## 1. 目的

在 MinerU 主实验之外，验证 V3 的 `1:M → M:1` 语义和 elastic rebatching 能否用于：

- 视频 frame pipeline；
- 通用文档 page pipeline；
- 裸 Ray Data 对比；
- 第三方 native pipeline 对比。

本文只记录已经实际运行的结果。未预热单次结果只用于 feasibility；正式性能结论必须使用
warmup、重复和中位数。

## 2. 视频：UCF101 → Frames → ResNet18 → Video

### 2.1 Workload

公开数据来自 Hugging Face `sayakpaul/ucf101-subset` 中两个独立 AVI：

```text
BabyCrawling       170 frames, 320×240, 25 FPS
BasketballDunk     125 frames, 320×240, 25 FPS
```

业务拓扑：

```text
Video
→ Expand(OpenCV sampled frames)
→ Map(ImageNet ResNet18)
→ Reduce(ordered top-5 signatures)
```

V3 和 Ray Data 复用同一个 decode、预处理、ResNet18 和 summary 实现。Ray Data 显式维护
`video_id`、`frame_ordinal` 和 `source_frame_index`。

为了让短公开样本形成稳定调度负载，两个 AVI 被重复为 8 个独立 logical videos；数据只下载
一次。每次运行采样 64 frame leaves。

配置：

```text
stride                 20
model                   torchvision ResNet18 ImageNet1K V1
model batch cap         16
transform replicas       2 CPU actors
torch threads/actor      1
warmup                   1 paired trial
measured                 3 paired trials
```

### 2.2 结果

```text
                         V3 elastic   V3 parent-bound
V3 wall                  4.248 s       3.975 s
Ray Data wall            5.587 s       4.278 s
paired V3/Ray Data       1.341×        1.076×
all-stage RPC            18            24
grains/RPC               4.444         3.333
batch fill               74.07%        47.62%
semantic parity          exact         exact
```

各次 measured wall：

```text
elastic V3       4.248, 4.271, 4.028
elastic RayData  4.259, 5.728, 5.587

parent V3        4.009, 3.940, 3.975
parent RayData   4.291, 4.252, 4.278
```

以上每次 trial 都重新创建 actors，属于 startup-inclusive wall。最初采集固定先跑 V3、后跑
Ray Data；runner 已修正为 paired trial 交替执行顺序，避免把 OS page cache 顺序优势误认为
框架收益。所以上表仍可作为已有 feasibility 记录，但正式数字需用修正后的 runner 重采。

### 2.3 判断

可以支持：

- V3 和裸 Ray Data 在相同真实模型 workload 上输出一致；
- V3 在本配置中的控制/数据组织开销低于裸 Ray Data；
- elastic 将 all-stage RPC 从 24 降至 18；
- elastic 将 fill 从 47.62% 提高到 74.07%。

不能支持：

- elastic 在所有模型上都更快；
- 该 CPU ResNet18 case 有 elastic wall-time speedup。

实际上，elastic wall 比 parent-bound 略高。原因是该 CPU model 对 8–16 frame batch 的
吞吐收益不足以覆盖等待/大 batch 成本。这是有价值的负结果：elastic 的收益条件应限定为
昂贵且 batch-efficient 的 stage，MinerU GPU OCR 才是主性能证据。

Correctness signature 使用 top-5 class IDs，不使用 raw float logits bytes。不同物理 batch
shape 的 CPU GEMM 可能产生末位差异，但 top-5 业务结果稳定。

## 3. Docling：PDF → Pages → Markdown

### 3.1 环境和 workload

隔离环境：

```text
Docling        2.117.0
Python         3.12
device         CPU
```

容器没有 `libGL.so.1`，因此必须移除 GUI OpenCV wheel 并使用
`opencv-python-headless`。

拓扑：

```text
PDF
→ Expand(PDFium-rendered PNG pages)
→ Map(persistent Docling image converter / convert_all)
→ Reduce(ordered page Markdown)
```

Ray Data baseline 使用相同业务函数，并显式维护 `document_id/page_ordinal`。

输入：

```text
1838_reformer_the_efficient_transfo.pdf
12 pages
```

### 3.2 V3 packing

```text
                         batch=1      batch=4
wall                     97.402 s     93.570 s
RPC                      14            5
grains/RPC                1.0          2.8
batch fill                  -         93.33%
Markdown chars           38,881       38,881
```

### 3.3 V3 与 Ray Data

```text
V3 wall                  94.148 s
Ray Data wall           114.426 s
Markdown token Jaccard    1.000
```

这是单文档、单次 CPU smoke，不能当正式 speedup；它证明了：

- page-level persistent actor 可运行；
- 跨 page coarse batching 生效；
- ordered Reduce 与 Ray Data regroup exact parity；
- V3/Ray Data/native 三种 runner 的比较边界已具备。

Docling native whole-PDF runner也已实现。正式表需要多个文档、预热、三次重复。

## 4. Page → Region 二层 fan-out

Docling 提供：

```text
PictureItem.get_image()
TableItem.get_image()
```

因此可自然形成：

```text
Document → Pages → Picture/Table regions → region model
         → Page Reduce → Document Reduce
```

但当前论文 PDF 前两页和一个公开 cTDaR table detection image 都没有稳定产生
`PictureItem/TableItem`。本轮不虚构二层实验；需要先找到能稳定触发 region 的公开样本。

## 5. PaddleOCR

隔离环境验证：

```text
PaddleOCR   3.7.0
Paddle CPU  3.3.1
Python      3.12
```

Python wheels 和 headless OpenCV 可以导入，但 PaddleX 的 OCR extra 按 distribution name
硬依赖 `opencv-contrib-python==4.10.0.84`。该 GUI wheel 在当前容器缺少
`libGL.so.1`；换成 headless wheel后 PaddleX extra 又判定依赖缺失。

因此 PaddleOCR 当前是环境阻塞的备用 case，不为了实验加入脆弱 monkey patch。

## 6. 当前结论

目前已有两个 MinerU 之外的真实 workload：

1. UCF101 + pretrained ResNet18；
2. Docling page pipeline。

二者均具备 V3 和裸 Ray Data runner、ordered output parity 和清晰的 `1:M → M:1`
拓扑。视频 case 已有 warmup + 三次重复；Docling 仍需要扩大输入和重复。

主性能结论仍等待 4×H20 恢复后完成 MinerU：

```text
V3 elastic
V3 parent-bound
Ray Data
Flash-MinerU native pipeline
```

当前会话 `torch.cuda.is_available() == False`，不能伪造该结果。

> 后续更新：受限 shell 内 CUDA 不可见，但 GPU benchmark 以沙箱外执行后确认 4×H20
> 可用。MinerU Ray Data 已完成 4/48-PDF smoke；full matrix 尚待运行。

## 7. MinerU 裸 Ray Data 预实验

Ray Data 使用：

```text
PDF
→ flat_map(page tensor + parent_id/page_ordinal)
→ map_batches(MinerU vLLM)
→ groupby(parent_id)
→ map_groups(ordered assembly)
```

真实结果：

```text
4 PDFs / 48 pages
wall                     61.666 s
V3/Ray Data Jaccard      0.9946 median

48 PDFs / 992 pages
wall                    165.419 s
pages/s                   5.997
V3/Ray Data Jaccard      0.9952 median
all documents >= 0.98
driver RSS peak         638 MB
```

Ray Data stats：

```text
OCR tasks                     14
rows/task                  60–72
page/image columns          ~11.1GB
groupby shuffle stage       155.13s
shuffle finalize block       2.5–3.1GB
```

Ray Data 可以跨 PDF 形成较大的 OCR batches，但为了恢复 document group，默认
`groupby(parent_id)` 会 shuffle 完整 page images、page metadata 和 OCR content。V3 的
lineage-aware Reduce 不需要用业务 payload 做 hash shuffle。这是比单纯“V3 API 更方便”
更强的系统差异，但需要 368-PDF full run 进一步观察 boundedness。
