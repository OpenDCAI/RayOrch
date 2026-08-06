# Docling V3：TableFormer V1 clean batch kernel

日期：2026-08-06；状态：实现、准确率与性能双 gate 均通过；`v1_batch` 晋升为推荐的
TableFormer V1 实验路径。

## 目标与边界

保持 V1 weights、crop、OTSL decoder 语义、cell matching 和 Page lineage 不变，只把
TableFormer 的执行边界改为显式：

```text
TableJob batch
→ prepare images
→ encoder + active-mask autoregressive decoder
→ per-row class/bbox scatter
→ matching/postprocess
→ Table values
```

候选模式命名为 `v1_batch`。`decoder_accelerated` 继续作为历史 golden 和可复现对照；
通用 CLI 的兼容默认值也不静默改变。新实验应显式选择 `v1_batch`。

实现不调用 `object.__setattr__()`，不替换 `model.predict`/encoder，也不通过 replay queue
把 batch 结果伪装成 singleton inference。由于上游 V1 没有 public batch API，神经模型
内部绑定与确定性 postprocess helper 被集中在钉版本 port；当前只接受
`docling-ibm-models==3.13.3`，依赖合同漂移时 fail fast。

## 预先冻结的晋升门槛

### 准确率

1. Ray-free contract：预处理 tensor 与上游 singleton 逐元素一致；per-row classes、bbox、
   顺序与 chunk scatter 有测试覆盖；完整 Docling benchmark tests 全绿。
2. 真实模型 smoke：singleton/golden 与 `v1_batch` 的最终 Table/Markdown exact；若 raw bbox
   只存在浮点尾差，必须报告 max absolute difference。
3. manifest 前 48 PDF：相对 current-head golden，Markdown byte-exact `48/48`，结构 exact
   `48/48`，Table/OCR 均零 error、零 fallback。
4. full 368 PDF：文档数、页数、表数、图片数完整；Table/OCR 零 error、零 fallback；语义
   差异不得差于既有 V1 独立重复运行噪声：token Jaccard 不得低于 `0.99`，并报告 exact、
   minimum、median 与 structure exact 分布。

### 性能

current-head `decoder_accelerated` 与 `v1_batch` 使用同一 manifest、UDF、4×H20 配比和冷
进程配置：

```text
microbatch_size=24, max_inflight_arenas=4
stage_batch_size=16, table_core_batch_size=4
parse/layout/ocr/table/reduce replicas=16/2/6/2/3
layout/table num_gpus=1/1
max_pending_per_actor=1, ray_num_cpus=128
OCR recognition_accelerated
```

- candidate measured wall 不得比 current-head golden 慢超过 `3%`；
- A/B 落在 `±3%` 噪声带时补跑，不凭单次结果晋升；
- 同时报告 startup、measured wall、E2E、Table batch histogram、GPU utilization 与 RPC；
- 任一错误、fallback、OOM 或输出缺失均直接否决该 run。

## Ray-free 证据

- `v1_batch` 使用独立 Table core stage，DAG 仍为
  `Document → Page → TableJob → Page → Document`；
- V1 image preprocessing 与上游 `_prepare_image` 的 tensor 逐元素相等；
- postprocess 保留 class argmax、bbox 转换、bbox sync、cell matcher、matching postprocess、
  response merge 和 row/column normalization；
- 测试保护候选实现不得重新引入 runtime method replay；
- 默认 `decoder_accelerated` 和 V2 opt-in 路径未被替换。

- Docling benchmark tests：`63 passed`；V1 kernel 定向 tests：`8 passed`；
- `git diff --check` 通过；
- 实测环境钉定 `docling==2.117.0`、`docling-ibm-models==3.13.3`、
  `ray==2.50.0`、`torch==2.7.1`。

## 真实模型与 PDF correctness

所有 run 均使用 direct RapidOCR、TableFormer V1 weights 和相同的 V3 elastic UDF/DAG。

| gate | 结果 | artifact |
| --- | --- | --- |
| 1 PDF smoke | Markdown/structure exact `1/1`；5 tables，零 error/fallback | `/tmp/mgv3-docling-tablev1-clean/v1-batch-onepdf-r1` |
| 前 48 PDF | Markdown/structure exact `48/48`；512 tables，零 error/fallback | `/tmp/mgv3-docling-tablev1-clean/v1-batch-48-r1` |
| full-368 r1 | 368 documents / 7072 pages / 3312 tables；零 error/fallback | `/tmp/mgv3-docling-tablev1-clean/full368-v1-batch-r1` |
| full-368 r2 | 368 documents / 7072 pages / 3312 tables；零 error/fallback | `/tmp/mgv3-docling-tablev1-clean/full368-v1-batch-r2` |

full-368 相对两次历史 V1 golden 的 token Jaccard minimum 均为 `0.99968354`、median
为 `1.0`、`<0.99` 为 `0/368`。候选 r1↔r2 自比较的 minimum/median 同样为
`0.99968354/1.0`，Markdown exact `360/368`，structure exact `345/368`；这与历史
V1 重复运行已有的少量 Docling 后处理非确定性同量级，没有新增语义退化。

完整差分证据：

```text
/tmp/mgv3-docling-tablev1-clean/full368-v1-batch-r1/vs-historical-v1-r1.json
/tmp/mgv3-docling-tablev1-clean/full368-v1-batch-r1/vs-historical-v1-r2.json
/tmp/mgv3-docling-tablev1-clean/full368-v1-batch-r2/vs-v1-batch-r1.json
```

## 性能 A/B

固定配置为 4×H20、`microbatch=24`、`max_inflight=4`、stage batch `16`、Table core
batch `4`、replicas `16/2/6/2/3`、OCR recognition batch `6`、每 actor 最多一个 pending
RPC。每个 full run 都是独立冷进程。

| 路径 | run 1 measured | run 2 measured | 双轮中位数 | 相对历史 V1 |
| --- | ---: | ---: | ---: | ---: |
| historical `decoder_accelerated` | 520.032s | 500.116s | 510.074s | baseline |
| clean `v1_batch` | 518.886s | 497.210s | 508.048s | **-0.40%** |

候选 startup 为 `7.694s / 7.793s`，E2E 为 `526.581s / 505.003s`；按 7072 pages
计算 measured throughput 为 `13.629 / 14.223 pages/s`。48-PDF 的同日 current-head
A/B 为 `89.262s` 对 `88.465s`，候选慢 `0.90%`。两组结果都在预先冻结的 `±3%`
噪声带内，因此结论是**性能等价、无回退**，不是可宣称的吞吐加速。

full r1/r2 分别把 3312 个 TableJob 聚合为 `976 / 974` 个物理 RPC，平均
`3.393 / 3.400 jobs/RPC`；batch histogram 分别为
`{1:99, 2:114, 3:67, 4:696}` 与 `{1:90, 2:124, 3:66, 4:694}`。Table GPU
（devices 2/3）平均利用率为 `28.07%/27.06%` 与 `29.48%/28.35%`，峰值 `94%`，
最大显存约 `1.87 GiB`，与历史 golden 同量级。

## 后续控制面调参

在 clean kernel 晋升后，以同一 48-PDF manifest 做单变量筛选；除表中参数外，其余均保持
上述 baseline。每个 run 均为 48/48 Markdown byte exact、structure exact，且零
error/fallback。

| 候选 | measured | 相对初始 89.262s | 结论 |
| --- | ---: | ---: | --- |
| `max_inflight=6` | 90.377s | +1.25% | 淘汰；RPC 略少但 admission 成本更高 |
| `max_inflight=8` | 93.790s | +5.07% | 淘汰 |
| `table_core_batch=8` | 91.173s | +2.14% | 淘汰；RPC 151→91，但 decoder straggler 更重 |
| 1 Layout GPU + 3 Table GPU | 91.757s | +2.80% | 淘汰；Table GPU 利用率被进一步摊薄 |
| `ocr_replicas=8` | 87.891s | -1.54% | 噪声带内，不晋升 |
| `reduce_replicas=4` r1/r2 | 80.533s / 79.333s | 双轮中位数 -12.0% | 进入 full gate |
| `reduce=4, ocr=8` | 78.987s | 相对 reduce r2 -0.44% | 无可证实增量，保留 OCR=6 |

筛选 artifact 位于 `/tmp/mgv3-docling-v1-tuning/48-*`。`reduce_replicas` 同时控制
postprocess、table expand、page reduce 和 document reduce 的轻量 CPU actor pools；增加到
4 没有改变 Table kernel、模型 batch、GPU 配比、Arena admission 或 shallow outstanding
window。

full-368 最终使用三次独立冷进程，并在前两轮差异较大后增加一组相邻
baseline→candidate tie-breaker：

| 配置 | run 1 | run 2 | run 3 | 三轮中位数 |
| --- | ---: | ---: | ---: | ---: |
| clean baseline，reduce=3 | 518.886s | 497.210s | 521.443s | 518.886s |
| tuned，reduce=4 | 483.991s | 497.809s | 478.165s | 483.991s |

三轮中位 measured wall 降低 **6.73%**，吞吐从 `13.629` 提升到
`14.612 pages/s`（+7.21%）；候选 startup 中位数 `7.725s`、E2E 中位数
`491.716s`。同序号 run 2 基本持平，说明系统仍有明显运行波动；但新增的相邻 tie-breaker
为 `521.443s → 478.165s`（-8.30%），且三轮中位数越过预设 3% 晋升线。

三个 tuned full run 均完成 368 documents / 7072 pages / 3312 tables，OCR 与 Table
零 error/fallback。相对 baseline 及 tuned 自身重复运行的 token Jaccard minimum/median
均为 `0.99968354/1.0`，`<0.99` 为 `0/368`。Table 物理 RPC 三轮为
`958/956/961`，GPU utilization 与原 baseline 同量级；收益来自 CPU 汇合 stage 缩短供给
长尾，而不是更激进的 GPU batching。

最终 artifact：

```text
/tmp/mgv3-docling-v1-tuning/full368-baseline-r3
/tmp/mgv3-docling-v1-tuning/full368-reduce4-r1
/tmp/mgv3-docling-v1-tuning/full368-reduce4-r2
/tmp/mgv3-docling-v1-tuning/full368-reduce4-r3
```

## 裁决

准确率与性能门槛同时通过。`v1_batch` 证明 TableFormer V1 能自然写成
`TableJob 1:M → batch kernel → Table M:1`：编排层只见稳定 DTO 和 batch callable；所有
不可避免的上游私有模型绑定集中在一个版本钉定 port，依赖清晰、无跨层飞线。

kernel 替换本身带来的是**抽象和维护性晋升，性能持平**；独立的控制面消融进一步证明
`reduce_replicas=4` 能在 full-368 上稳定改善三轮中位数。后续 V3/V3.4 Docling 实验推荐
显式使用 `--table-batch-mode v1_batch --reduce-replicas 4`，其余 golden 配置保持不变。
保留 `decoder_accelerated` 仅用于历史复现，不再在其 monkeypatch/replay 设计上继续叠加
功能。
