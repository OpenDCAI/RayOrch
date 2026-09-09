# MultiGrain V3.6 真实 workload 发布回归

> **文档生态位：一次可复核的历史实验快照。** 本文证明下述 commit/configuration 的结果，
> 不是对之后任意工作树的滚动担保。学习和规范文档入口见
> [`V3.6 文档地图`](../../multigrain_v3_6_documentation_map.md)。

日期：2026-08-08
结论：通过。V3.6 在冻结配置下未发现系统性性能退化，四项 paired gate 的平均相对变化均在
±2% 内，结构与身份合同全部通过。

## 1. 目的与边界

V3.6 是从冻结的 V3.5（`5f59566`）分出的 breaking architecture release。它重命名并
重新划分 LogicalProgram、ProgramAnalysis、RuntimePlan、RuntimeState、DispatchState 和
Worker DTO，但不应改变业务语义或引入新的调度开销。

本轮验证的是提交 `cdd01d0` 所代表的核心实现，以及测试过程中发现的 benchmark adapter
修正。门禁统一要求：

- 两轮独立冷启动 paired trial，并交替执行顺序；
- 报告均值、样本方差和 paired 相对变化；
- 对比文档/视频身份、结构与业务输出；
- 对比 RPC、平均 batch、actor 和释放计数，排除静默结构膨胀；
- 结束后执行 `ray.shutdown()`，并确认本机无 Ray instance 和 GPU compute process。

机器使用 4×H20。Docling 与视频以包含启动、物化和 teardown 的 outer wall 为主指标；
MinerU 沿用历史 runner 的 engine measured wall，并另外记录 startup/end-to-end。

## 2. 结果总表

| Workload | 基线 | 数据与配置 | 基线均值 / 方差 | V3.6 均值 / 方差 | paired 平均变化 | 正确性 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| MinerU | frozen V3.5 | 368 PDF / 7,072 页；microbatch 24，active 3 | 603.889 s / 16.497 s² | 600.556 s / 8.545 s² | **-0.551%** | 368 identity 全部一致；结构与 OCR batch 分布一致 |
| Docling | V3 | 48 PDF / 992 页 / 512 tables；active 2 | 91.606 s / 5.103 s² | 92.742 s / 0.779 s² | **+1.260%** | identity、结构、Markdown 全部逐文档精确一致；零 error/fallback |
| Video Caption | V3 | 256 videos；microbatch 32，active 4 | 86.854 s / 1.064 s² | 88.570 s / 0.717 s² | **+1.990%** | 结构一致；两轮规范化文本漂移 0.391% / 0.293%（模型非确定性门限 <2%） |
| Video Multimodal | V3 | 256 videos；audio/frame sibling domains；32×4 | 39.632 s / 0.054 s² | 39.827 s / 0.102 s² | **+0.492%** | 输出 byte exact；结构、frame digest、transcript 全部一致 |

正值表示 V3.6 慢，负值表示 V3.6 快。两次样本不足以估计长期分布，但交替顺序、结构诊断
和四类不同 workload 没有显示共同方向或随实体数增长的额外开销。

## 3. MinerU full-368

顺序与 measured wall：

| Pair | 顺序 | V3.5 | V3.6 | V3.6 - V3.5 |
| --- | --- | ---: | ---: | ---: |
| 1 | V3.5 → V3.6 | 606.761 s | 602.623 s | -4.138 s |
| 2 | V3.6 → V3.5 | 601.017 s | 598.489 s | -2.528 s |

paired delta 的均值为 -3.333 s，样本方差 1.296 s²。四次执行都产生 368 个文档、
7,072 页、121 次 OCR RPC，OCR 平均 batch 为 58.446，完整直方图也相同。V3.6 总 RPC
为 640/642，V3.5 为 641/640；active high-watermark 均为 3，actor 均为 13，释放值均为
16,352。因此没有发现新 Ref/Record 或传播事件导致的结构性放大。

368 个 Markdown 路径身份完全相同。首轮 Jaccard median 0.995893、mean 0.992008，次轮
median 0.995953、mean 0.992861。最低值来自 `hyper-connection_copy_7`；但 V3.6 两次
运行彼此的 Jaccard 只有 0.741061，低于同轮跨 runtime 的 0.750666/0.919832，说明它是
业务模型非确定性，不是 Entity/Item 映射错位。

原始产物：`/tmp/rayorch_v36_mineru_paired_20260808/results.jsonl` 及同目录四个输出目录。

## 4. Docling 48-PDF

outer wall 样本为：

- V3：93.202991 s、90.008455 s；
- V3.6：93.366572 s、92.118057 s。

两轮均得到 48 文档、992 页和 512 tables；所有 identity、结构和 Markdown 完全一致，
OCR/Table error 与 fallback 均为 0。V3 RPC 为 598/604，V3.6 为 588/628；这反映 ready
时序下的 batch 边界变化，没有引发结果或错误差异。

预检暴露了一个真实迁移问题：paired adapter 仍把旧参数 `max_in_fight` 传给 V3.6，导致
V3 arm 完成后 V3.6 arm 才失败。它已改为 `max_active_microbatches`，并增加 adapter
forwarding 回归，防止这种 API 边界飞线再次出现。

原始产物：`/tmp/rayorch_v36_docling_20260808/paired-48.json`。

## 5. 视频与容量窗口反例

第一次 Caption paired 使用 CLI 原默认值 microbatch 4、active 2，V3 平均约 123.5 s，
V3.6 约 199.5 s。冻结 V3.5 控制实验同配置也约 194.9 s，且 V3.5/V3.6 的 RPC 分别为
580/约 581，而 V3 约为 390。由此可知它不是 V3.6 重构退化，而是 V3.5 起采用的
immediate work-conserving scheduler 在过小 admission window 下无法形成大 batch。

冻结的性能配置原本就是 microbatch 32、active 4。修正后 Caption 的 V3/V3.6 两轮 RPC
都为 390，V3.6 Caption 阶段为 67 RPC / 1,024 grains，平均 batch 15.284。正式 paired
结果只慢 1.990%。本轮同时做了两项可复现性修正：

- video paired CLI 默认值改为冻结性能窗口 32×4；
- JSON artifact 显式记录 `microbatch_size` 与 `max_active_microbatches`。

这个反例应保留：`microbatch_size` 和 `max_active_microbatches` 是容量/吞吐配置，不是纯粹
的内存安全旋钮。小窗口仍受支持，但不能拿它与 V3 的 timed-wait scheduler 做默认性能结论。

Multimodal 同时覆盖 audio/frame 两个 sibling Domain，各自 Expand/Map/Reduce 后在 root
合并。V3.6 两轮总 RPC 为 1,105/1,107，V3 为 1,103/1,109，输出完全一致，证明该复杂
lineage 路径没有因静态/动态状态拆分产生额外实体或遗漏。

原始产物：

- `/tmp/rayorch_v36_video_20260808/caption-256-mb32-i4-paired2.json`
- `/tmp/rayorch_v36_video_20260808/multimodal-256-mb32-i4-paired2.json`
- 反例：`caption-256.json` 与 `caption-v35-control-256.json`

## 6. 发布判断

V3.6 的版本号是必要的：本轮包含用户可见 breaking rename、静态图/运行时状态表拆分、
Worker ABI 与 Executor 参数收口，不适合回写已经完成回归的 V3.5。真实 workload 证明这些
架构变化没有带来系统性性能退化；保留 V3.5 作为 frozen oracle，也使之后的优化能继续做
paired 判断。

本轮不据此声称任意 workload 都自动等价。新增 primitive、调度策略或 admission 默认值时，
仍需分别验证结果身份、结构、RPC/batch 形状和端到端 wall time。
