# V3.6 MinerU 单页投毒、`GroupFailure` 与竞品 API 审核

日期：2026-08-27
代码状态：`codex/multigrain-recovery-tiers`，基于 `ff3a1b865765` 的未提交工作树
环境：4×GPU、32 CPU、Ray 2.50.0、Daft 0.7.21、160 PDFs / 3,584 pages

## 结论

1. `GroupFailure` 的**结果隔离合同**在真实 MinerU 中成立。READY sibling 可以被
   admission 门禁封存；已经领取的同父 page 会继续计算，但其晚到结果不会提交；
   健康父级不受影响。所有实测 arm 的最终 report 和计数守恒均通过。
2. 当前 4-actor work-conserving 调度下，没有找到“粒度足够且无毒性能只小幅下降”的
   单一 OCR batch setting。`B=48` 仅慢 6.1%但不剪枝；`B=16` 已慢 49.8%仍不
   剪枝；`B=12` 慢 64.2%，随机 poison 阶梯只剪掉 2.8% / 6.2% / 3.1% 的潜在
   siblings。
3. `B=8, root microbatch=4` 对一篇有利的 53 页长 PDF 可把 50/52 siblings
   挡在 actor 外，但无毒 measured wall 慢 93.8%。它证明机制有效，不能代表随机
   workload 的通用甜点。
4. 因此推荐把 `B=64` 保留为吞吐 preset：它保证精确结果隔离和审计，但不承诺节省
   已进入投机窗口的 OCR。只有 workload 的父级明显更长、坏数据代价远高于正常吞吐时，
   才应选择小 batch isolation preset；不能为了论文图表改生产默认值。
5. 同 seed、64 poison、B64 natural full-value 对照中，Ray Data 和 Daft 都能在
   regroup 后丢弃完整毒父并保持管线继续，但 3,584 页全部进入 OCR actor，runtime
   没有在线 parent barrier。RayOrch B64 也因窗口过大而未节省 sibling；它的差异是
   typed lineage-scoped contract，而不是本次配置下凭空获得算力收益。
6. 为回答“坏一页是否必须丢整篇 PDF”，又完成了三引擎共同的 `skip_page` 矩阵。
   4/16/64/80 个 PDF 各毒 page 0 后，三者都保留 160 篇文档，输出严格为
   `3,584−N` 页，批次结构不随投毒量变化。这个能力依赖 UDF 显式返回 marker 和
   tolerant reducer，不是竞品自动识别，也不是 RayOrch 的 typed parent barrier。
7. Ray Data 的 item-local p16 首次运行在 shuffle/assemble 尾部因
   `ObjectFreedError` 失败；相同参数唯一一次复跑成功。故成功矩阵合同通过，但不能写
   “所有尝试零失败”。该事件属于 object lifetime/recovery，不是 poison UDF exception。
8. 严格排除普通异常路径后，在已检查的 Ray Data 2.50.0 和 Daft 0.7.21 公共
   API 中，没有找到与 `RecordFailure` / `GroupFailure` 等价的、由 UDF **返回**且由
   runtime 解释为 lineage-scoped control value 的 sentinel。不过不能声称竞品没有
   “坏项不中断”能力：Daft 已有 exception-to-null 的逐 invocation 容错。

## 数据与注入合同

- 输入按路径排序后取前 160 个 PDF，共 3,584 页。
- 投毒率矩阵使用 seed `rayorch-mineru-poison-v1`，子集随 count 单调嵌套。
- 每个选中 PDF 只给 page 0 注入确定性 poison；指定坏页本身不发送给真实 OCR 模型。
- parent-scoped 矩阵让 RayOrch UDF 返回 `GroupFailure(cause)`；竞品 runner 返回普通
  `poisoned` 状态列，并在 regroup 后丢整父。
- item-local 矩阵让 RayOrch 返回 benchmark payload `PoisonedPage(cause)`，Ray Data / Daft
  返回普通 `poisoned` 状态列；三边 reducer 都只去掉坏页并保留该 PDF。`PoisonedPage`
  是应用数据，不是新增 runtime failure type。
- 小窗口实验固定 `pdf_index=112`：`DeepSeek_V3.pdf`，53 页，投毒 page 0。
- `ocr_grains` 是实际形成 pending RPC 并进入 OCR UDF 的 page 数；
  `group_sibling_pages_not_dispatched = input_pages - ocr_grains`，不是由输出页数反推。
- 已进入 actor 的同父兄弟无法安全追杀；其结果在原子 batch commit 时被 suppression
  barrier 拒绝。

## Batch size 理论边界

160 个 PDF 的页数直方图为：`9×16, 12×16, 14×16, 16×16, 19×16,
20×32, 27×16, 34×16, 53×16`。均值 22.4，median 19.5，P90 34，
P95/P99 53。

设 OCR actor 数为 `R`、单次 dispatch cap 为 `B`。最简单的一轮并发视界是
`H = R × B`；若希望 page 0 的 poison 首批返回后还有同父 READY page，至少需要
父级页数大于 `H`。这里 `R=4`：

| B | 一轮视界 `4B` | 对本数据的理论含义 |
|---:|---:|---|
| 64 / 48 / 32 / 24 | 256 / 192 / 128 / 96 | 大于最大父级 53，不能期待首轮剪枝 |
| 16 | 64 | 仍大于最大父级 53 |
| 12 | 48 | 只给 53 页长父留下结构性机会 |
| 8 | 32 | 给 34/53 页父留下机会 |
| 4 | 16 | 才接近覆盖 median 19.5 的随机父 |

这不是严格上界：work-conserving actor 可以在 poison batch 返回前继续领取后续 batch，
根 `microbatch_size` 和 active arenas 也会改变 admission 顺序。因此理论只用于缩小候选，
最终必须以 `ocr_grains` / `not_dispatched` 实测。尤其不能把 `R×B` 宣称为全局 bounded
inflight guarantee。

## 无毒 sweep 与精确长父配对

下表 baseline 都是 160 PDF / 3,584 pages 单次 feasibility run；相对值以
`B64,m24=263.879s` 为基准。单毒列固定 53 页 `DeepSeek_V3.pdf` 的 page 0，数值为
完全未进入 actor 的 52 个潜在 sibling 数。

| OCR B | Root m | No-poison measured | vs. B64 | OCR RPC / fill | Exact-long-parent not-dispatched |
|---:|---:|---:|---:|---:|---:|
| 64 | 24 | 263.879s | — | 62 / 0.903 | 0 / 52 |
| 48 | 24 | 279.873s | +6.1% | 81 / 0.922 | 0 / 52 |
| 32 | 24 | 307.131s | +16.4% | 116 / 0.966 | 0 / 52 |
| 24 | 24 | 338.259s | +28.2% | 154 / 0.970 | 0 / 52 |
| 48 | 8 | 287.270s | +8.9% | 88 / 0.848 | 0 / 52 |
| 48 | 4 | 327.379s | +24.1% | 118 / 0.633 | 0 / 52 |
| 16 | 4 | 395.364s | +49.8% | 238 / 0.941 | 0 / 52 |
| 12 | 4 | 433.349s | +64.2% | 312 / 0.957 | 22 / 52 |
| 8 | 24 | 513.270s | +94.5% | 449 / 0.998 | 13 / 52 |
| 8 | 4 | 511.463s | +93.8% | 458 / 0.978 | 50 / 52 |

`microbatch_size` 不是 OCR batch 的同义参数。B48 把 root m 从 24 降到 4 只造成
batch fragmentation 和 24.1% 性能损失，仍没有剪枝；B8 的健康性能对 root m 不
敏感，但精确长父剪枝从 13 页升到 50 页。也就是说，OCR cap 决定粒度上限，root
admission 决定 poison 何时相对 siblings 被发现；二者共同构成投机深度。

## 默认吞吐配置：投毒率矩阵

固定参数：`batch_size=64`、`microbatch_size=24`、
`max_active_microbatches=3`、4 OCR actors。每档均为单次 feasibility run。

| Poison PDFs | Poison-parent pages | PRESENT / SUPPRESSED | OCR grains | Model pages | Not dispatched | Computed then discarded | OCR RPCs | Measured wall | vs. baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 160 / 0 | 3,584 | 3,584 | 0 | 0 | 62 | 263.879s | — |
| 4 | 112 | 156 / 4 | 3,584 | 3,580 | 0 | 108 | 62 | 264.619s | +0.280% |
| 16 | 338 | 144 / 16 | 3,584 | 3,568 | 0 | 322 | 62 | 259.454s | −1.677% |
| 64 | 1,308 | 96 / 64 | 3,584 | 3,520 | 0 | 1,244 | 62 | 258.381s | −2.084% |
| 80 | 1,642 | 80 / 80 | 3,584 | 3,504 | 0 | 1,562 | 62 | 261.570s | −0.875% |

五档的 OCR RPC 数、batch histogram 和 `ocr_grains` 完全一致。高投毒率没有出现
可见的超线性状态管理代价、retry 或 actor busy 泄漏；wall-time 差异处于单次运行
波动范围。这里能 claim commit isolation 和健康路径无明显回归，不能 claim 已节省
同父兄弟算力。

## Item-local 挽救：坏一页不丢整篇 PDF

这组控制矩阵直接回答最实际的需求：用户已经能识别偶然坏页时，只跳过该页，健康
siblings 仍然组装成原 PDF 的输出。固定 160 PDF / 3,584 pages、4 GPU、B64、同 seed、
同一嵌套 `4/16/64/80` poison manifest 和 natural `full_value` regroup。p0 与 poison
policy 无关，复用同配置无毒基线；其余 12 个 arm 是独立实跑。

公平墙钟仍从 `ray.init()` 后到输出收集完成，并包含 actor/model startup。所有 poison
arm 的机器断言均为：

`docs=160, output_pages=3,584−N, poison_observed=N, dispatched=3,584,
model_pages=3,584−N, report_contract=true`。

| Poison PDFs | Expected docs / pages | RayOrch wall | Ray Data wall | Daft wall |
|---:|---:|---:|---:|---:|
| 0 | 160 / 3,584 | 307.134s | 463.258s | 435.289s |
| 4 | 160 / 3,580 | 305.017s | 456.246s | 433.436s |
| 16 | 160 / 3,568 | 306.457s | 482.366s | 422.470s |
| 64 | 160 / 3,520 | 302.866s | 458.860s | 422.775s |
| 80 | 160 / 3,504 | 305.556s | 469.664s | 422.430s |

相对各自 p0，RayOrch 为 `−0.69% / −0.22% / −1.39% / −0.51%`，Ray Data 为
`−1.51% / +4.13% / −0.95% / +1.38%`，Daft 为
`−0.43% / −2.95% / −2.88% / −2.95%`。没有系统随 poison 数出现单调恶化；单次
feasibility run 只能支持“未观察到投毒控制面回归”，不能支持统计显著加速。

### 实际参与运算与 batch profile

| Engine | Poison matrix | OCR dispatched | Model pages | Batch/RPC | Fill | 整篇丢失 |
|---|---|---:|---|---:|---:|---:|
| RayOrch B64 | 4/16/64/80 | 每档 3,584 | `3,584−N` | 每档 62 | 0.903 | 0 |
| Ray Data 2.50 | 4/16/64/80 | 每档 3,584 | `3,584−N` | 每档 98 | 0.571 | 0 |
| Daft 0.7.21 | 4/16/64/80 | 每档 3,584 | `3,584−N` | 每档 61 | 0.918 | 0 |

三者都只在含毒 batch 内绕过 N 个坏页的真实模型调用；健康 rows 正常进入模型和
commit。所有 3,584 页仍进入 OCR stage，因此这组实验验证 **item-local salvage**，不
验证 sibling dispatch pruning。批次/RPC 数和 histogram 在各自四档完全恒定，投毒
检查是每 row 的常数时间 predicate，没有制造碎 batch 或毒量相关的超线性状态。

RayOrch 的 `PoisonedPage` 是 benchmark-local 普通 payload，作用等价于竞品的
`poisoned: bool` 列。没有直接用 `RecordFailure`，因为 reducer 需要一个保留
`parent_id/page_id/cause` 的占位值来维护可审计的父内对齐；这不改变 runtime API claim。
真正由 runtime 解释并能建立 READY/commit barrier 的仍只有 `GroupFailure`。

### 错误与压力审计

- 12 个实际 poison arm 的最终 summary 全部合同通过；RayOrch 四档
  `ocr_retries=0`。Ray Data / Daft 不暴露可直接比较的内部 retry counter。
- Ray Data p16 的第一次尝试在 shuffle/assemble 尾部以 `ObjectFreedError` 退出，driver
  同时报出正在恢复 18 个 lost objects，且没有写出 summary。完全相同参数的第二次、
  也是唯一一次复跑成功（表中 482.366s）。因此该行是 `attempts=2,
  failed_attempts=1`；不能把第一次失败隐藏，也不能归因给逻辑 poison。
- Ray Data 成功 arm 的 full-value shuffle 仍约 40.64GB；issue detector 报告 OCR task
  实际内存约 4.7–8.0GB 而声明为 0B，并警告可重启 actor 的构造参数若从 object store
  丢失则恢复会失败。这些资源/恢复警告不计作失败，但与首次 object-loss 同属值得披露
  的稳定性压力。
- Daft 四档均一次成功，未见 fatal error；其 physical plan 的未处理异常策略仍是
  `on_error=raise`。本矩阵的连续运行来自显式 marker，不是异常自动转 null。

在四个 poison 档，RayOrch end-to-end 比 Daft 少 27.5%–29.6%，比 Ray Data 少
33.1%–36.5%。它是同机单次 feasibility 结果，不替代交错多轮统计性能实验。

## 小窗口配对实验

固定参数：`batch_size=8`、`microbatch_size=4`、
`max_active_microbatches=3`、4 OCR actors。baseline 和 poison arm 除注入外完全一致。

| Metric | Baseline | One 53-page poisoned parent | Difference |
|---|---:|---:|---:|
| Successful documents | 160 | 159 | −1（精确目标父级） |
| Final PRESENT / SUPPRESSED | 160 / 0 | 159 / 1 | 精确隔离 |
| OCR grains entering actors | 3,584 | 3,534 | **−50** |
| Physical model pages | 3,584 | 3,533 | −51（含 poison trigger） |
| Sibling pages not dispatched | 0 | **50** | READY 门禁命中 |
| Sibling pages computed then discarded | 0 | 2 | inflight commit 门禁命中 |
| OCR RPCs | 458 | 451 | −7 |
| Mean OCR grains/RPC | 7.825 | 7.836 | 基本一致 |
| Batch fill | 0.9782 | 0.9795 | 基本一致 |
| OCR retries | 0 | 0 | 无异常重试 |
| Measured wall | 511.463s | 502.102s | −1.83%，单次运行不作性能 claim |

poison arm 的 `poison_injected=1`、`poison_observed=1`、
`poison_report_contract_passed=true`。最终输出页数为 3,531，恰好从 3,584 中扣除
目标父级的 53 页；其余 159 个文档全部正常提交。

这组结果确认门禁的位置正确：50 个 page grain 没有进入 actor/UDF；不是先计算再从
最终输出过滤。`batch_size` 控制单 actor dispatch horizon，`microbatch_size` 控制一次
admit 的根 PDF 数量，二者不能混为同一个参数。

## B12 随机阶梯：长父结果不能泛化

为避免用固定 53 页长父挑有利结果，在 `B=12,m=4` 上复用默认 seed 的嵌套随机
poison 集合。`potential siblings = poison-parent pages - poison count`，剪枝率只统计
完全没进入 OCR actor 的 siblings。

| Poison PDFs | Poison-parent pages | Potential siblings | Not dispatched | Computed/discarded | Prune ratio | OCR grains | Measured wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 112 | 108 | 3 | 105 | 2.8% | 3,581 | 428.519s |
| 16 | 338 | 322 | 20 | 302 | 6.2% | 3,564 | 428.371s |
| 64 | 1,308 | 1,244 | 39 | 1,205 | 3.1% | 3,545 | 424.385s |

三档均满足：

`poison-parent pages = poison triggers + not-dispatched + computed/discarded`。

每档 `poison_report_contract_passed=true`，没有 retry、整批失败或 executor busy 泄漏；
健康 PDF 均完整提交。但 64 poison 时 OCR grains 只减少 39/3,584（1.1%），wall 相对
B12 无毒也只减少 2.1%。因此 B12 的主要价值是证明长父/有利 admission 下存在中间
剪枝点，不应被命名为随机 MinerU workload 的推荐甜点。

对这组页数分布，若确实要求 page-0 poison 对 median 父也大概率在首轮后留下 READY
sibling，理论上需要接近 `B=4`，甚至还要限制 actor 续领深度。这会显著增加 OCR RPC，
且当前实验已经显示 B8 慢 93.8%；没有必要为了制造更漂亮的 suppression 数字继续把
生产 batch 调到 4。

## Ray Data / Daft 实测对照

对照固定为 160 PDF、3,584 pages、4 GPU、B64、同 seed 的嵌套
`0/4/16/64/80` poison、natural `full_value`。Ray Data 使用 4 source blocks；Daft
使用此前该 workload 扫描得到的 8 source partitions。两边都使用 runner 的
`drop_parent`：OCR UDF 对指定 poison item 不调用模型，随后在 groupby/regroup 看到
毒标记后丢整父。它与 GroupFailure 的最终文档语义一致，但不是 runtime 自动隔离。

公平墙钟使用 RayOrch `end_to_end_wall_s`，而非排除 startup 的 `measured_wall_s`；三个
runner 都不计 `ray.init()`，都计 actor/model startup 到输出收集完成。每档为单次
feasibility run，不能写成统计显著性能结论。

| Poison PDFs | Healthy docs / output pages | RayOrch wall | Ray Data wall | Daft wall |
|---:|---:|---:|---:|---:|
| 0 | 160 / 3,584 | 307.134s | 463.258s | 435.289s |
| 4 | 156 / 3,472 | 306.560s | 485.077s | 419.086s |
| 16 | 144 / 3,246 | 300.794s | 483.973s | 418.669s |
| 64 | 96 / 2,276 | 301.414s | 480.645s | 419.368s |
| 80 | 80 / 1,942 | 304.961s | 463.734s | 408.379s |

相对各自 p0，RayOrch 为 `−0.19% / −2.06% / −1.86% / −0.71%`，Ray Data 为
`+4.71% / +4.47% / +3.75% / +0.10%`，Daft 为
`−3.72% / −3.82% / −3.66% / −6.18%`。没有一个系统随 poison 数出现单调恶化或
超线性控制面开销；单次波动不能解读为严格加速。

### 实际运算页与 batch profile

| Engine | Poison matrix | OCR dispatched | Model pages | Sibling not-dispatched | OCR RPC/batches | Fill |
|---|---|---:|---|---:|---:|---:|
| RayOrch B64 | 0/4/16/64/80 | 每档 3,584 | `3,584−N` | 每档 0 | 每档 62 | 每档 0.903 |
| Ray Data 2.50 natural | 0/4/16/64/80 | 每档 3,584 | `3,584−N` | 每档 0 | 每档 98 | 每档 0.571 |
| Daft 0.7.21 natural | 0/4/16/64/80 | 每档 3,584 | `3,584−N` | 每档 0 | 每档 61 | 每档 0.918 |

`N` 是投毒 PDF 数，每个 PDF 只毒 page 0。三个系统都只在模型调用上省掉 `N` 个
trigger；所有同父 siblings 都进入 OCR actor。也就是说，参与昂贵 OCR 运算的比例为
`(3,584−N)/3,584`：p4 99.89%、p16 99.55%、p64 98.21%、p80 97.77%。最终输出页
变少不能被误报成 OCR 算力节省。

Ray Data 的 98 batch / 0.571 fill 和 Daft 的 61 RPC / 0.918 fill 在五档中完全恒定，
说明逻辑 poison 没有破坏它们的物理 batching。RayOrch 同样恒定为 62 / 0.903。
Ray Data 五档 natural full-value hash shuffle 的 map output 均约 40.644GB；运行中 issue
detector 报告 OCR task 约 5.6–8.5GB 的内存估计，但没有 OOM。GPU peak 三套系统均约
77.7GiB；driver RSS peak 约为 RayOrch 722–723MiB、Ray Data 791–793MiB、Daft
662–719MiB。单节点 shuffle bytes 不是 NIC bytes。

### 错误、压力与隔离结论

- 15 个主矩阵 arm 全部 `poison_observed=N`、`poison_report_contract_passed=true`；
  runner-observed fatal pipeline error 和 contract failure 都为 0。
- RayOrch 每档明确记录 `ocr_retries=0`，并把 N 个 poison 作为逻辑 failure outcome，
  不是异常。
- Ray Data / Daft summary 没有暴露可直接比较的内部 retry counter；它们的 stats/plan
  没有 failed/retried task 标记。Daft physical plan 是 `on_error=raise`，本轮没有未处理
  exception，否则查询不会成功。不能把“未暴露 counter”写成“内部 retry 严格为 0”。
- 两个竞品都能承受半数 poison、准确保留健康父并安心结束；但这是 benchmark UDF 手工
  返回 `poisoned` 状态列再 group/drop 的结果，不是公共 runtime 根据 lineage 自动隔离。
- RayOrch 的 typed parent barrier 是独有差异；但在 B64 下它同样来不及挡住 siblings，
  因而本矩阵支持语义/API claim，不支持 B64 下的 sibling compute-saving claim。

64-poison 单档的公平 wall 中，RayOrch 比 Daft 少 28.1%、比 Ray Data 少 37.3%。这只
是同机单次 feasibility，不替代交错多轮主性能表。

## Final report 校验

每个 arm 的 `summary.json` 和 `poison_manifest.json` 都记录：

- 精确 PDF index、绝对路径、PDF 名称、page id、父级页数和 cause；
- `poison_injected`、`poison_observed` 与最终 `ItemOutcome` 分布；
- expected/actual healthy document 和 page 数；
- actor admission、真实模型调用、READY suppression、inflight discard、retry 与 batch
  指标。

本文采用的所有**成功** GPU arm 的 `poison_report_contract_passed` 均为 `true`。小窗口
单毒报告中的 cause 为 `poisoned PDF DeepSeek_V3.pdf at page 0`，最终 outcome 为
`PRESENT=159, SUPPRESSED=1, FAILED=0, DROPPED=0`。item-local 矩阵额外逐行校验
`docs=160` 和 `pages=3,584−N`。Ray Data item-local p16 的首次失败没有 summary，作为
独立 failed attempt 记录，未被成功复跑覆盖或计入“成功 arm”。

原始临时产物：

- `/tmp/rayorch_poison_matrix_160/results.jsonl`：5 行，SHA-256
  `e09c99c9d3c7a3d03e21321762d52f0a758b9e453748ca64fe9648e6d8e42d39`
- `/tmp/rayorch_poison_small_window_160/results.jsonl`：2 行，SHA-256
  `c00a0f3ed5ef8163f3ac45e2fe35e2030a19118a554d43fb243b5c38ac44f2d9`
- `/tmp/rayorch_batch_sweep_160/results.jsonl`：16 行，SHA-256
  `b981a1a447a8f09d75c0a7df95b178a774501d957b9d521dab76ea804fa41b5d`
- `/tmp/rayorch_poison_staircase_b12_160/results.jsonl`：3 行，SHA-256
  `47991a48b8cdb2fa3ae202f9bf99102b7cc14788c3495869481dd78b09aeada9`
- `/tmp/rayorch_poison_compare_160/results.jsonl`：Ray Data / Daft 各 5 行，SHA-256
  `40796d4a00bde5da6e44f71c956f4ce2aa3bac04773328d27f611193cbd62bfd`
- `/tmp/rayorch_poison_skip_matrix_160/results.jsonl`：三引擎各 4 个 item-local poison
  成功 arm，共 12 行，SHA-256
  `cdff090f8c0d726782f8d6f516be4692532fb3f98ca21e3870a3e002468703b9`
- 持久 compact 主矩阵：
  [`controlled_b64_matrix.json`](artifacts/2026-08-27_group_failure_poison/controlled_b64_matrix.json)，
  15 rows，SHA-256
  `32a3690c478590097ef7ffec85e34e5baf30e8cc0a43f6791280003228cc269f`
- 持久 item-local 矩阵：
  [`item_local_skip_matrix.json`](artifacts/2026-08-27_group_failure_poison/item_local_skip_matrix.json)，
  15 logical rows（3 个 p0 复用 + 12 个 poison 实跑），并单列 Ray Data p16 的首次
  `ObjectFreedError` failed attempt；SHA-256
  `516e68edaac96c3e81481abe757427b744f2e38462d9a9d83f355cf0ec4d2dea`。

`/tmp` 不是长期归档；parent-scoped 与 item-local 两个 15-row 逻辑矩阵的可审计标量
都已进入持久 artifact。完整 GPU samples、输出和 framework plan/stats 仍在 `/tmp`，
正式论文复现前应在冻结 commit 上迁移到外部大文件归档并记录 checksum。

## Ray Data / Daft 公共 API 审核

### Ray Data 2.50.0

`Dataset.map` / `map_batches` 的公开签名没有 failure sentinel 或 row-level
`on_error` 参数。最接近的内置机制是 `DataContext.max_errored_blocks`：它允许少量
block-processing exception 后继续执行，但官方合同明确说明**失败 block 中的数据会
整体丢弃**。`actor_task_retry_on_errors` 则是异常 retry 分类，不是逻辑坏项结果。

因此 Ray Data 可以：

- 让用户在 UDF 中手工 `try/except`、返回状态列再 filter；或
- 允许少量物理 block 失败并整块丢弃。

但这两者都不会让 runtime 根据一个返回值建立 `(call, parent)` barrier，也不会在
后续 admission 时剪掉同 lineage 的 READY siblings。把 block 人为切成单 row 可以
逼近 item granularity，却把正确性粒度绑定到物理 partition，并牺牲 batching。

来源：[Ray Data 2.50 `DataContext`](https://docs.ray.io/en/releases-2.50.0/data/api/data_context.html)、
[Ray Data `map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html)。

### Daft 0.7.21

Daft 的新 `@daft.func` API 比普通 raise 更完整：`max_retries=N` 可重试失败 invocation；
重试耗尽后，`on_error="log"` 或 `"ignore"` 会为失败 invocation 产出 `None` 并让查询
继续。它已覆盖“偶发坏行不终止全局”这一层，因此不能写“Daft 没有坏项隔离”。

它与 V3.6 当前 API 的差异是：

- 入口仍是 exception，不是 UDF 返回的 typed logical failure；
- `None` 与合法 nullable output 共用数据表示，异常 cause 不成为最终逻辑 outcome；
- 公共合同没有 record-vs-parent 两种 scope，也没有 immediate-parent lineage barrier；
- 没有把同一个 parent predicate 同时用于 READY admission 和 late commit。

用户当然可以手工返回 `{ok, error, parent}` struct 并在下游 group/filter，但 Daft runtime
只把它视为普通数据；这种写法不能在昂贵 UDF 前剪掉尚未执行的 siblings。

来源：[Daft Functions：`max_retries` 与 `on_error`](https://docs.daft.ai/en/stable/custom-code/func/)、
[Daft legacy UDF migration guide](https://docs.daft.ai/en/stable/custom-code/migration/)。

## 可写与不可写的论文 claim

不应写：

> Ray Data and Daft cannot continue past poisoned records.

也不应在未做更广泛系统综述前写 “first ever”。

可以针对本文 baseline 明确写：

> Unlike block-dropping tolerance in Ray Data and exception-to-null handling in
> Daft, RayOrch exposes typed UDF results whose failure scope is interpreted
> against runtime lineage. `RecordFailure` terminates one grain, whereas
> `GroupFailure` atomically establishes an immediate-parent barrier shared by
> READY admission and late-result commit.

更短的贡献名可以使用 **lineage-scoped failure effects**。它的先进性不在“也能跳过
坏数据”，而在于把坏项从应用状态列提升为 runtime 可解释的、带 scope 和 cause 的
控制值，并使调度剪枝、原子 commit 与最终审计采用同一个 predicate。
