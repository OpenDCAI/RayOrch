# 当前原生 Flash-MinerU 全量基线与外层 batch 对照

日期：2026-07-24

Flash-MinerU commit：`7246a35`

RayOrch commit：`56787bd`

## 1. 实验问题

本实验直接运行当前原生 Flash-MinerU，回答：

1. 当前 commit 处理 368 PDFs 的端到端性能是多少；
2. V2.5 相对当前原生版本仍有多少收益；
3. 将原生 Flash-MinerU 的外层 PDF `batch_size` 调大，是否足以获得
   V2.5 的性能；
4. 当前证据可以把收益归因到什么程度，还需要什么严格 ablation。

## 2. 原生 Flash-MinerU 的 batch 语义

原生公开入口：

```python
MineruEngine(
    batch_size=8,
    replicas=4,
    num_gpus_per_replica=1.0,
    inflight=4,
)
```

其中：

```text
batch_size = 一个 pipeline microbatch 中的 PDF 数
```

它不是一次 vLLM 调用的 page 数。

原生 `ProcessImagesOp.run()` 的核心逻辑是：

```python
for images_list in images:
    contents = client.batch_two_step_extract(
        images=该单个 PDF 的全部 pages,
    )
```

所以即使外层 PDF batch 增大，一个 actor 仍然逐 document 调用
`batch_two_step_extract()`，不同 PDF 的 pages 不会合并进同一个 vLLM
调用。本数据集每个 PDF 分别包含 12、16 或 34 页，因此原生有效 OCR
batch 通常也是 12、16 或 34 pages。

V2.5 则把 page 表达为独立 Logical Grain，并允许同一 Arena 中来自不同
PDF/parent 的 pages 形成一个最多 128-page 的 OCR Dispatch。

## 3. 实验环境与计时

```text
GPU                       4× NVIDIA H20
单卡显存                  97,871 MiB
系统内存                  约 2.2 TiB
Ray                       2.50.0
Python                    3.12
模型                      MinerU2.5-2509-1.2B
```

输入：

```text
按文件名排序的前 368 PDFs
总页面数                  7,072
```

计时方法：

1. 构造 `MineruEngine`；
2. 对所有 render/OCR/convert actors 调用 `__ray_ready__`；
3. readiness barrier 完成后开始 `measured_wall`；
4. `engine.run(pdfs)` 返回后停止计时；
5. 模型构造和 actor readiness 单独计入 `startup`。

## 4. 当前原生 368-PDF 全量结果

配置：

```text
PDF batch_size             8
max batches inflight       4
render replicas            4
OCR replicas               4
convert replicas           4
GPU/OCR replica            1×H20
```

结果：

```text
startup                    42.367 s
measured wall             818.001 s
end-to-end                860.368 s
PDF/s                       0.4499
pages/s                     8.6455
pipeline result batches        46
Markdown outputs             368/368
layout.json outputs           368/368
```

此前记录的历史原生 baseline 为 891.980 s。本轮当前 commit 比历史记录快：

```text
绝对差异                   73.979 s
相对改善                    8.29%
```

后续对当前代码版本做性能比较时，应使用本轮 `818.001s`，历史
`891.980s` 只保留作历史参考。

## 5. 与 V2.5 全量结果比较

### 5.1 对齐旧 MG 宏观参数的 V2.5

```text
microbatch_size            24 PDFs
max_inflight_arenas         3
OCR replicas                4
OCR batch_size            128 pages/RPC/replica
measured wall             521.836 s
pages/s                    13.5522
```

相对当前原生：

```text
speedup                     1.568×
wall 减少                 296.165 s
耗时降低                   36.21%
```

### 5.2 当前 V2.5 最快已记录结果

```text
单 Arena
OCR batch_size            256
measured wall             491.738 s
pages/s                    14.3816
```

相对当前原生：

```text
speedup                     1.663×
wall 减少                 326.263 s
耗时降低                   39.89%
```

## 6. 原生外层 PDF batch 对照

使用同一批 48 PDFs / 992 pages：

| 原生配置 | result batches | measured wall | pages/s |
|---|---:|---:|---:|
| PDF batch 8，inflight 4 | 6 | 98.359 s | 10.086 |
| PDF batch 24，inflight 3 | 2 | 131.873 s | 7.522 |

把原生外层 PDF batch 从 8 调到 24：

```text
wall 增加                 33.515 s
耗时增加                  34.07%
```

原因与原生代码结构一致：

- 外层 batch 变大不会合并不同 PDF 的 pages；
- 每个 OCR actor 仍然逐 PDF 调用 vLLM；
- microbatch 数减少，使跨 microbatch pipeline overlap 机会变少；
- 更大的 document-level barrier 和长尾会影响完成时间。

因此本实验排除了以下解释：

> V2.5 只是把原生 Flash-MinerU 的外层 PDF batch 调大，所以变快。

## 7. 当前可以成立的性能归因

### 已经可以成立

1. 当前原生 Flash-MinerU 在相同 368-PDF 数据上的 measured wall 为
   `818.001s`；
2. V2.5 `521.836s` 相对当前原生为 `1.568×`；
3. 原生外层 PDF batch 从 8 增加到 24 并不会获得 V2.5 的收益，48-PDF
   实验中反而慢 34.07%；
4. V2.5 的直接物理收益来自更大的有效 page-level vLLM batch；
5. 这种大 page batch 来自：

```text
PDF
→ dynamic page Logical Grains
→ 跨 parent READY queue
→ elastic page rebatching
→ 大 OCR Dispatch
```

### 仍然不能完全拆开的贡献

当前“原生 vs V2.5”同时改变了：

- document grain → page grain；
- 原生 per-document OCR → V2.5 page-level OCR；
- 是否允许跨 parent 组 batch；
- execution/manifest/Reduce 实现。

因此当前还不能仅凭原生对照，把全部 `1.568×` 收益都声明为
cross-parent elastic rebatching。

要得到 elastic rebatching 的净收益，仍需在 V2.5 内部做：

```text
parent_bound
vs
elastic
```

并保持其他参数完全一致。

## 8. 外部实验产物

### 368-PDF 当前原生基线

```text
结果
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/native_full_368_results.jsonl

日志
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/native_full_368_pdfbs8_i4_20260724.log

输出
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/outputs_native_full_368_pdfbs8_i4_20260724
```

### 48-PDF 原生外层 batch 对照

```text
结果
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/native_batch_ablation_results.jsonl

batch8 日志
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/native_control_48_pdfbs8_i4.log

batch24 日志
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/native_control_48_pdfbs24_i3.log

batch8 输出
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/outputs_native_control_48_pdfbs8_i4

batch24 输出
/apdcephfs_zwfy10/share_304380933/hunyuan/sunnyhazema/workspace/Flash-mineru/outputs_native_control_48_pdfbs24_i3
```

这些大体积产物不提交到 Git。

## 9. 实验限制

- 每个配置目前只有一次运行，没有报告多次重复的方差；
- 48-PDF batch8/batch24 对照同时改变了外层 batch 和 inflight，目的是复现
  两种实际 macro configurations，不是单变量理论实验；
- 原生执行没有与 V2.5 使用完全相同的 page-grain UDF ABI；
- 仍需 V2.5 内部 `parent_bound vs elastic` paired ablation；
- 不据此宣称所有 workload 都能获得相同 speedup。

## 10. 下一步

下一步实验应固定：

```text
同一 V2.5 Pipeline
同一 page Logical Grains
同一 microbatch_size
同一 max_inflight_arenas
同一 OCR replicas
同一 OCR batch_size hard cap
同一 max_batch_wait_ms
同一 transport/manifest/Reduce
同一输入顺序
```

唯一改变：

```text
batch_scope="parent_bound"
vs
batch_scope="elastic"
```

该实验用于测量 cross-parent elastic packing 的净收益。
