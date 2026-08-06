# Multigrain v3.5 真实 workload 迁移与性能验证

日期：2026-08-06；状态：MinerU full gate 已有结果，Docling/Video authoring 已迁移，
Docling GPU gate 待明确授权，Kinetics-400 约 50GB 数据正在准备。

## 目标与证据边界

本轮不是再写一组看起来相似的 demo，而是固定三类真实关系图，复用 V3 的同一业务 UDF，
只替换 authoring、compiler、RuntimePlan、Arena 和 Executor：

1. MinerU：`PDF → Page → Document`，验证单层动态关系和 aligned reduction；
2. Docling：`Document → Page → TableJob → Page → Document`，验证两层嵌套关系；
3. Video：单 frame relation、caption relation，以及 audio/frame 两个 sibling relation。

因此输出或性能变化可归因于框架层，而不是悄悄换模型、预处理或组装 kernel。所有性能结论
必须来自同一有序 manifest、同一 UDF、同一模型、同一 actor 资源和交替执行顺序。

## 当前实现

### MinerU

入口：`rayorch.experimental.multigrain_v3_5.benchmark.mineru`。

已有 368-PDF 结果为 368 documents / 7,072 pages，measured `602.760s`；V3 golden
为 `587.781s`，差异 `+2.55%`，落在预先使用的 `±5%` 回归带内。完整记录见
`2026-08-06_mineru_regression.md`。

### Docling

入口：

- `benchmark.document_docling.core_v35`：V1/V2/reference 三个 kernel adapter；
- `benchmark.document_docling.paired`：V3↔V3.5 4/48/368 paired gate。

编译后的图固定为 8 个业务 Call、2 个 Expand、2 个 Group 和 3 层 Domain；optimized 与
unoptimized 均为 `Document(root) → Page → TableJob`。V3 的 timed batch wait、actor
concurrency 和 shallow outstanding window 不会被 v3.5 接受或静默忽略；paired artifact
会列出唯一被投影掉的 V3-only 选项，并明确记录：

- V3：timed wait + shallow outstanding；
- V3.5：immediate work-conserving。

主性能口径是两臂一致的 outer wall，包含 actor startup、最终输出物化和 teardown。旧 V3
内部 measured 不含 `RunResult.get()`，v3.5 measured 含物化，所以二者只作为诊断字段，
不能直接作为 paired 主结论。

Worker 新增一次 run 尾部的只读 `WorkerObservation`：只包含 calls、RSS 和标量 audit。
它不进入 LogicalProgram、RuntimePlan、Arena、routing 或 retry 决策。Docling gate 会拒绝
任意非零 OCR/Table error 或 fallback。

真实 GPU gate 当前没有启动：此前“回归只用 MinerU、不用 Docling”的指令与后续迁移
Docling 目标冲突，需要用户明确允许后再占用 4×H20。

### Video

入口：

- `benchmark.video.v35`：`Video → Frame → Feature → Video`；
- `benchmark.video.caption_v35`：`Video → Frame → Caption → Video`；
- `benchmark.video.multimodal_v35`：audio/frame 两个 sibling child Domain，分别 Reduce
  回 Video 后做 root-level merge；
- `benchmark.video.paired`：V3↔V3.5 单 relation paired gate；
- `benchmark.video.model_paired`：SmolVLM caption 与 Whisper+ViT paired gate；
- `benchmark.video.kinetics50`：约 50GB Kinetics 数据准备。

真实的两个 MSR-VTT MP4 CPU gate 已逐字段相等：frame count、source frame ordinal、
digest 和 edge density 完全一致；两版物理 RPC 都是 6。旧 V3 必须通过其公开 `get()`
物化 `BlockSlice` 后再比较，不能直接拿 ObjectRef wrapper 与 v3.5 业务对象比较。

Caption gate 沿用历史语义：video/frame/source-index 必须 exact，但 greedy generation 会因
batch padding/GEMM shape 有少量 token 漂移，因此单独报告 raw/normalized mismatch，默认
normalized 上限为 `2%`，不错误要求文本 byte exact。Whisper+ViT smoke 的最终结构化摘要
仍要求 exact。

## 约 50GB 数据合同

采用 CVDF 官方托管的 Kinetics-400 S3 snapshot，而不是重复现有 UCF101/MSR-VTT 文件：

- validation 全 20 shards；
- train shards `0..12`；
- 33 archives；
- 2026-08-06 HEAD 核验总计 `50,607,623,297` bytes（`47.132 GiB`）；
- 官方说明每个 tar 约 1,000 个视频，clip 约 10 秒。

官方来源：

- https://github.com/cvdfoundation/kinetics-dataset
- https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt
- https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt
- https://s3.amazonaws.com/kinetics/400/annotations/val.csv
- https://s3.amazonaws.com/kinetics/400/annotations/train.csv

Kinetics 原 clip 来自第三方视频；这里只做本地研究回归，不获得也不主张再分发权。

数据准备 CLI 默认只打印 plan。只有显式 `--action download/extract/all` 才写正文；下载使用
可恢复 `.part`、冻结 Content-Length 校验，解压使用 Python safe tar filter，并把每个 shard
放进独立目录。解压后仍必须用 V3 已有 manifest builder 对每个文件重新 decode/probe，
按实际视频 bytes 选择接近 50GB 的独立 source，并记录 source identity；archive bytes 不能
冒充 workload bytes。

```bash
python -m rayorch.experimental.multigrain_v3_5.benchmark.video.kinetics50 \
  --root /tmp/mgv35-kinetics50 \
  --plan-output /tmp/mgv35-kinetics50/plan.json \
  --action all --download-workers 8

python -m rayorch.experimental.multigrain_v3.benchmark.video.manifest \
  --input-root /tmp/mgv35-kinetics50/videos \
  --output /tmp/mgv35-kinetics50/manifest-50g.json \
  --dataset kinetics400-cvdf --split val+train \
  --target-gib 45 --min-count 10000 --seed 20260806
```

`45 GiB` 是约 `48.3 GB` 的实际解压视频下限；如果解压后的可 decode 总量足够，会在正式
run 前把 target 上调并冻结最终 manifest/report。不得通过复制路径或重复 source_id 达标。

## 冻结 gate

### Docling

1. 4 PDF：结构 exact、Markdown Jaccard minimum `≥0.99`、零 error/fallback；
2. 48 PDF：相同 gate，并诊断 batch/RPC/GPU；
3. 368 PDF：至少两次交替 paired trial；主 outer-wall 中位数不得比 V3 慢超过 `5%`；
4. 368/368 documents、7,072 pages、3,312 tables 应与冻结 manifest/golden 对齐。

### Video

1. 2/48 video correctness smoke：V3/V3.5 完整业务输出 exact；
2. 现有 1.3GB MSR-VTT 与 4GB UCF101：确认历史 workload 不回退；
3. Kinetics 约 50GB：manifest 的 source identity 唯一、实际 bytes 达标、全量 decode 无坏片；
4. V3↔V3.5 使用交替顺序，主 startup-inclusive wall 中位数回归带为 `±5%`；
5. caption 与 multimodal 先做小型真实性 gate，再决定是否值得对 50GB 全量执行昂贵模型；
6. 报告吞吐、RPC/batch、Arena high watermark、RSS/GPU 和最终输出 digest，不只报 wall time。

## 当前结论

v3.5 的编译器化不是为了这几个 benchmark 临时加 special case。三个 workload 已覆盖：

- 单层、嵌套和 sibling Domain；
- Expand/Group 的完整状态迁移；
- 同 Domain 多输入与 Reduce 后 root merge；
- compiler/runtime 分层、无 Origin 进入运行时；
- observation-only 物理诊断，不反向污染语义。

这些结构在 authoring 中都只由 `RayModule + F.expand/F.reduce` 表达，没有 `driven_by`、
手写 entity key 或跨层查表飞线。性能优势仍必须由未完成的 Docling/Video paired runs 证明；
在真实数据 gate 完成前，只能说设计行为符合预期，不能宣称普遍加速。
