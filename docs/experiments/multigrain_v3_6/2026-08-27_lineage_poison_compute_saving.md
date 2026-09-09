# V3.6 parent-lineage poison isolation：B48 可扩展墙钟对照

## 结论

在同一份 1,885-PDF / 45,507-page long-tail workload、同 4×H20、同
`batch_size=48`、同 50 ms/page 昂贵 UDF 下，只有 RayOrch v3.6 能在坏父被识别后阻止
READY sibling 进入 actor。三次轮换重复中：

- RayOrch 墙钟平均从 581.39 s 降到 494.57 s，配对下降 86.81 s（14.93%）；
- Ray Data 平均只下降 1.02 s（0.18%），与跳过 99 个 poison trigger 的理论 1.24 s
  一致，没有 sibling pruning；
- Daft 的 paired delta 有正有负，未观察到稳定墙钟收益；
- RayOrch 每轮完全未派发 6,157–6,293 页，18 个正式 arm 的输出合同、exact identity
  和 suppression accounting 全部通过。

这支持的论文 claim 是：**对于包含少量长文档坏父的 long-tail data-science workload，
parent-lineage-aware runtime 能把用户声明的父级 failure 转化为可测的物理工作和墙钟节省；
仅在末端 group/drop parent 的系统不能节省已生成 sibling 的昂贵 UDF。**

它不支持“真实 MinerU OCR 快 14.93%”或“处理了 284 GB PDF”这两个 claim。

## 单一主表

每个 arm 运行 3 次。墙钟为 `framework_wall_s` 的 mean ± sample SD；变化是同 repetition
的 healthy − poison 配对均值。物理页指真正执行非 poison 昂贵 UDF 的页数。

| Framework | Condition | Physical expensive-UDF pages | READY sibling 未派发 | Wall time (s) | Paired change vs. healthy |
|---|---|---:|---:|---:|---:|
| RayOrch v3.6 | healthy | 45,507 | 0 | 581.39 ± 1.33 | — |
| RayOrch v3.6 | 99 poisoned parents | 39,167 mean（39,115–39,251） | 6,241 mean（6,157–6,293） | 494.57 ± 0.93 | −86.81 s（−14.93%） |
| Ray Data 2.50 | healthy | 45,507 | 0 | 571.35 ± 0.08 | — |
| Ray Data 2.50 | 99 poisoned parents | 45,408 | 0 | 570.32 ± 0.13 | −1.02 s（−0.18%） |
| Daft 0.7.21 | healthy | 45,507 | 0 | 665.26 ± 6.82 | — |
| Daft 0.7.21 | 99 poisoned parents | 45,408 | 0 | 662.84 ± 3.75 | −2.42 s（−0.36%，方向不稳定） |

RayOrch 三轮配对节省分别为 86.54、88.82、85.08 s；描述性 95% t interval 为
[82.13, 91.50] s。Daft 为 7.97、−1.04、0.34 s，区间 [−9.63, 14.48] s，不能解释为
稳定 isolation 收益。Ray Data 的约 1 s 小收益不是 sibling suppression：三轮都执行
45,408 页，只比 healthy 少 99 个 poison trigger。

## 为什么这份数据的名义受益约为 10%

完整 frozen manifest 有 3,689 个有效 PDF、174,744 页。B48 和 4 actors 给出用于设计
workload 的名义投机窗口：

~~~text
R × B = 4 × 48 = 192 pages
~~~

从真实 manifest 中保留：

1. 页数最大的 99 个 PDF；
2. 其余所有不超过 23 页的 PDF；
3. 保持 frozen manifest 的原始顺序，不复制 PDF。

得到 1,885 个 PDF、45,507 页。top-99 只占文档数 5.25%，但贡献 23,613 页，构成典型
long-tail stress。若每个长父的 page 0 投毒，名义 READY 机会为：

~~~text
sum(max(parent_pages - 192, 0)) = 4,605 pages
4,605 / 45,507 = 10.12%
~~~

`R×B` 不是严格全局 inflight bound。实际 runtime 还受 admission 顺序、actor 续领和多父
交错影响，因此实测 RayOrch READY saving 为 6,157–6,293 页（13.53%–13.83%）。论文应把
10.12% 称为 workload-design heuristic，把 13.53%–13.83% 称为实测结果。

## 控制变量与公平性

- Hardware：单机 4×NVIDIA H20 97,871 MiB；driver 535.247.01。
- Software：Python 3.12.11、Ray 2.50.0、Daft 0.7.21、Torch 2.7.1。
- 三套系统：4 个昂贵 UDF replicas、B48、50 ms/non-poison-page、4 expand、4 reduce。
- Healthy 与 poison 的 1,885 个路径和页数逐项相同；只改变 top-99 的 page-0 poison
  标记。
- 每个 child 使用 fresh local Ray runtime；六个 arm 每轮循环平移，避免固定顺序总是偏向
  同一个框架。
- 主计时排除 `ray.init`。RayOrch 从 parent records 开始，计时包含 Parent→Page、昂贵 UDF
  和 parent reduce。
- Ray Data/Daft 在计时前由 driver 构造均衡的 128 个 page partitions。这去除了竞品的
  parent-bound block skew，也把它们的 page expansion 移出计时，属于对竞品保守有利的
  accommodation。

无毒首轮 RayOrch 580.51 s、Ray Data 571.42 s，只差 1.59%；两者都接近固定工作的理想
下界 `45,507×50ms/4 = 568.84s`。因此 poison 差异不是由一个失衡的健康基线制造。

## Failure 语义与闭合

RayOrch 的 page-0 `GroupFailure` 做三件事：

1. 终端输出 suppress 该 parent；
2. READY sibling 在 dispatch gate 被拒绝；
3. 已 inflight sibling 可以完成 RPC，但 commit 时丢弃，不进入下游。

例如 repetition 0：

~~~text
99 poison triggers
+ 6,293 READY siblings not dispatched
+ 17,221 inflight siblings computed then discarded
= 23,613 poisoned-parent pages
~~~

三轮 RayOrch 都闭合。Ray Data/Daft 的控制语义是 page UDF 标记 poison，完成 sibling 后在
parent group/reduce 丢弃整个父；因此同样得到正确终端输出，但 READY saving 恒为 0。

## 证据边界

这是一项 cardinality-faithful scheduler experiment：读取 frozen PDF manifest，按真实页数
生成轻量 page records。50 ms synthetic UDF 在持有 one-GPU actor allocation 时 sleep，用于
隔离 admission、suppression 和 wall-time 因果关系。

因此可以声称：

- 45,507 个真实 cardinality page records 上，RayOrch 少 admit 约 6.2k 个昂贵 UDF items；
- 同卡数、同 B48、同输入下，parent-lineage isolation 带来约 14.9% 墙钟下降；
- Ray Data/Daft 的末端 drop-parent 保持结果正确，但不剪掉 READY sibling。

不能声称：

- 实际读取或 rasterize 了所选 PDF 的 284.3 GB bytes；
- 50 ms sleep 等价于 MinerU GPU kernel、GPU energy 或 OCR accuracy；
- 单个规模点证明 O(N) 渐进复杂度。

真实 MinerU OCR 的小规模实验继续作为外部有效性补充；本表负责可控地证明 lineage
机制转化为物理 work 和墙钟收益。

## 可复现材料

- [protocol.json](artifacts/2026-08-27_lineage_poison_compute_saving/scalable_b48/protocol.json)
- [environment.json](artifacts/2026-08-27_lineage_poison_compute_saving/scalable_b48/environment.json)
- [results.csv](artifacts/2026-08-27_lineage_poison_compute_saving/scalable_b48/results.csv)
- [aggregate.json](artifacts/2026-08-27_lineage_poison_compute_saving/scalable_b48/aggregate.json)

Launcher 最初设置了 5-run cap。数据和单点时长扩大后，报告固定使用前三个完整循环轮换
repetitions；第 4 轮在产生任何 summary 前停止，未混入结果。原始 18 条 JSONL 的 SHA256
为 `7400c23ced564a80c25840de7db39fd579defaa6f2cec8c6406e61e31cc93662`。
