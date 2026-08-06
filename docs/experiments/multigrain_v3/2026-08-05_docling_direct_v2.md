# Docling V3：RapidOCR 直接适配与 TableFormerV2 batch

日期：2026-08-05；状态：实现、真实模型 gate 与 368-PDF 回归已完成。RapidOCR direct
adapter 通过；TableFormerV2 batch 抽象通过，但 V2 模型质量/性能不满足替代 V1 的门槛。

## 结论边界

这次只替换 Docling workload 的模型 UDF，不改变 V3 图：

```text
Document → Page → Layout/OCR values → TableJob → Page → Document
```

- RapidOCR adapter 直接持有 `RapidOcrModel`，通过 RapidOCR facade 组织 detector、
  classifier 和 recognizer，不再依赖 `OcrAutoModel._engine`，也不复刻 ONNX session、
  CTC decode、RTL 恢复或结果对象构造。
- TableFormerV2 使用独立 `TableJobV2 → batch kernel → Table` 实现；V1 的
  `reference/decoder_accelerated` 路径及其 368-PDF 结果原样保留。
- Pipeline authoring 仍是 V3 `Pipeline + Expand/Map/Reduce`。DTO、模型 adapter、batch
  kernel 和 lineage scatter 不依赖该 authoring API；迁移到 `RayModule + F.*` 时只替换
  图表达和 lowering。

因此这不是 Docling 私有控制流的另一份长期 fork，而是“稳定 workload dataflow + 可替换
模型 kernel”的验证版本。

## 实现边界

### RapidOCR

跨 page/rect 按稳定 lineage 收集 detector 产生的 crops，按 normalized width 隔离极端
长宽比，再调用：

```text
preprocess_img / detect_and_crop / cls_and_rotate
→ recognize_txt(list[crop])
→ build_final_output
→ Docling post_process_cells
```

adapter 只读取 `text_rec.rec_image_shape` 计算 compatibility key，不物化 normalized tensor；
resize/normalize、按宽高比排序、`rec_batch_num` 切分、inference 和 decode 都属于
RapidOCR。每个结果按 `(page, rect, crop)` lineage scatter；facade 异常时只把对应 group
涉及的 rect 标为 fallback。`recognition_shadow` 继续提供严格 reference gate；
`recognition_accelerated` 不重复执行 reference。

### TableFormerV2

Table job 直接携带 2× page crop、table cluster 和 OCR text cells。kernel 批量执行：

```text
encode_images(batch)
→ autoregressive forward with per-row EOS mask
→ split bbox predictions by row lineage
→ OTSL decode / cell build / text match
→ page-local table map
```

batch 内某一行结束后强制保持 EOS，不能让短序列继续生成；bbox flat output 依据每行截断后的
cell token 数量拆分。OCR cell 先转换为轻量坐标记录，再执行与 Docling 等价的 overlap 匹配，
避免在内循环反复构造对象。

当前只隔离保留两个版本钉死的确定性 postprocess helper：`_decode_otsl_sequence` 和
`_build_table_cells`。adapter 不访问 `TFPredictor._prepare_image`、`model._encoder`、
`model._tag_transformer`，也不通过 `object.__setattr__()` 临时替换 encoder/predict。

## 已有证据

### V1 历史性能证据（保留，不覆盖）

同一 4×H20、368 PDF / 7,072 pages 的最终 V3 V1 两轮：

```text
measured wall       520.032 / 500.116 s
median              510.074 s
Native tuned median 824.220 s
throughput speedup  1.5957×
OCR jobs            60,624，零错误/零 fallback
Table jobs           3,312，零错误/零 fallback
```

历史 artifacts：

```text
/tmp/mgv3-docling-tablejob/full368-fair-parse16-ocr6-pending1-layout2-table2-r1
/tmp/mgv3-docling-tablejob/full368-fair-parse16-ocr6-pending1-layout2-table2-r2
```

V1 两次独立运行自身并非 Markdown byte-exact：358/368 exact；token-set Jaccard 最低
0.99968354、中位数 1.0，且没有文档低于 0.99。最低差异主要是算法块前后多/少 Markdown
fence。这组分布是审查 V2 的噪声基线，不能把所有非 byte-exact 都归因于新 adapter。

### V2 局部正确性证据

在 H20 上对两个不同尺寸的 synthetic table 同时推理：

```text
batch token ids == upstream serial token ids
bbox counts       4 / 12，与 serial 一致
bbox max abs diff 8.6e-5 / 9.3e-5
input batch       (2, 3, 448, 448)
```

Ray-free gate 覆盖 per-row EOS、lineage bbox split、空 batch、text-cell match、V3 pipeline
选择和 CLI/config 矩阵。它证明 batch decoder 的合同成立，但不能替代真实 PDF 的端到端
语义与吞吐回归。

真实 PDF page 的 CPU RapidOCR gate：29 个 OCR crops，reference 0.800s，direct
accelerated 0.740s，5 个 RapidOCR minibatches、零 error/fallback；最终 251 cells 的文本
与 reference 完全一致。shadow 同样 exact，但耗时 2.091s，因此只作为 correctness 模式。

### 真实 PDF 与全量结果

direct RapidOCR + V1 `decoder_accelerated` 在 manifest 前 48 PDF 上与历史 V1 artifact：

```text
Markdown byte-exact 48 / 48
structure exact     48 / 48
OCR jobs             7,408，零 error / fallback
measured wall       88.465 s
```

这证明去掉 `OcrAutoModel._engine`、改为 facade adapter 没有改变该组真实输出。V2 的
5-table batch 与五次 singleton、以及包含 512-token 长表的 `2411.10741v1` batch 与
singleton，Markdown 均 byte-exact；因此 V2 的 EOS fencing、bbox split 和 lineage scatter
没有引入 batch-shape 语义漂移。

同一历史公平配置的 V2 full368（4×H20，368 PDF / 7,072 pages）：

```text
startup                         7.671 s
measured wall                 561.519 s
E2E                           569.190 s
vs Native tuned 824.220 s       1.468× / wall -31.87%
vs V1 median 510.074 s         +10.09% wall
OCR jobs                       60,624，零 error / fallback
TableV2 jobs                    3,312，零 error / fallback
jobs reaching 512-token limit     144（4.35%）
```

Artifact：

```text
/tmp/mgv3-docling-tablev2/full368-v2-fair-r1
/tmp/mgv3-docling-tablev2/full368-v1r1-vs-v2r1-diff.json
```

V1→V2 的 368-document diff：structure exact 346，token Jaccard median 0.993798、最低
0.910688；96 份低于 0.99，16 份低于 0.95。`hyper-connection` 的一张 V2 表跑满 token
上限，使 Markdown 从约 95K 膨胀到约 1.316M 字符；`2410.19313v1` 等复杂表也出现明显
行/表头合并。相同输入的 singleton V2 会产生相同结果，故这是当前上游 V2 模型/权重的
质量边界，不是 batch adapter 的语义错误。

最终裁决：保留 `v2_batch` 作为独立、清晰、可继续更换模型的 opt-in kernel；默认和 golden
性能路径继续使用 V1。V4 应继承 adapter/kernel 的抽象方式，不应默认继承这次 V2 模型选择。

## 已发现并修正的性能陷阱

1. 直接复用上游 `_match_text` 会让 table actors 长时间停在逐 cell 对象匹配；改为一次性
   准备 OCR cell 坐标，保持同一 overlap 阈值与 fallback 语义。
2. 早期所谓“单 PDF 超过 2 分钟、OCR 重复”不是 ONNX/Ray replay：命令同时传入
   `--manifest` 与 `--limit 1`，旧 `_read_paths()` 在 manifest 分支忽略 limit，实际顺序处理
   368 个复制 PDF。CLI 已改为 manifest、显式 paths 和目录输入统一应用正数 limit，并增加
   path/metadata 同步裁剪测试。单 PDF 真实回归中 OCR UDF 只调用一次、约 4.3s。
3. RapidOCR 仍按 normalized-width compatibility group 调用 facade；这是避免极端长宽比在
   内部 minibatch 中扩大 padding 的保守物理策略，但不再作为上述误诊的解释。

中止的误配置运行不计入性能结论；上面列出的 one-PDF、48-PDF 与 full368 artifacts 才是
有效证据。

## Gate 结果

- [x] one-PDF direct OCR、V2 batch/singleton 与 V1 artifact 对照。
- [x] 48 PDF table-heavy 递进；检查 token limit、cell count、error/fallback。
- [x] 4×H20 full368；同时报告 measured 与 E2E。
- [x] V1/V2 逐文档 token Jaccard、结构 counts 与最低差异人工审查。
- [x] promotion gate 作出否决：V2 比 V1 median 慢 10.09%，且存在明确复杂表语义退化；无需
      为一个已否决候选再消耗第二轮 full368。V1 两轮历史波动证据继续作为 golden baseline。

输出差异工具：

```bash
python -m \
  rayorch.experimental.multigrain_v3.benchmark.document_docling.core_output_diff \
  /path/to/v1/documents.json.gz \
  /path/to/v2/documents.json.gz \
  --output /path/to/diff.json
```
