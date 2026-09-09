# V3.6：32×H20 大 PDF parent-scoped poison isolation

日期：2026-08-27
状态：正式实验完成；结果与限制已审计；compute 已 deprovision 并删除

## 结论摘要

本轮实验支持一个有边界的论文 claim：

> RayOrch V3.6 能把用户返回的 typed parent failure 同时用于 READY admission、
> inflight late-commit barrier 和最终 parent outcome；健康 parent 继续输出，坏 parent
> 不泄漏晚到结果。在坏 parent 尚有 READY sibling 时，它还能少派发真实模型工作。

它不支持“RayOrch 在所有 workload 上都更快”或“任何 poison 都能节省大量计算”：

- 在真实 MinerU OCR、293-page poison parent 位于输入首项时，batch 4/8/16/64 分别少执行
  162/30/1/1 页模型工作，其中 READY gate 真正避免派发 161/29/0/0 页。
- batch 从 4 增至 16 时健康吞吐更好，但发现 poison 前已 inflight 的 sibling 越多；这正是
  预注册的 throughput—isolation Pareto，而不是实现错误。继续从 batch 16 增至 64 后
  填充率从 92.2% 降至 59.9%，所以 batch 64 反而比 batch 16 慢。
- 全量 3,689 个 parent、174,744 页的 10 ms/page synthetic-control 中，单个最大 parent
  在 batch 8 下也没有 READY saving，但 top-16 大 parent 有 1,786 页 READY saving；
  batch 64 下，单个最大 parent 和 top-16 均为 0 READY saving。
- 廉价 10 ms kernel 上，V3.6 baseline 为 431.84 s E2E，Ray Data 为 106.35 s。
  这说明 V3.6 的细粒度调度/物化开销明显，不应宣称 universal scheduler speedup。
- Daft 0.7.21 在本 compute 上受 Ray head↔GPU 数据面拓扑影响，3-page compatibility gate
  无法完成；本报告不把它写成 Daft 算法性能落后。

所有完成 arm 的 correctness/accounting contract 均通过。真实 OCR arm 为单次配对，
wall-time 只作运行上下文，不作显著性结论。

## 研究问题与 failure 合同

在大规模 PDF UDF 管线中，对一页确定性坏数据：

1. 不终止其他健康文档；
2. 精确 suppress 坏页所属整篇文档；
3. 同父尚处 READY 的页面不再送入昂贵 OCR；
4. 已被其他 actor 领取的 sibling 允许算完，但在 commit 时丢弃；
5. 正常 parent 的结果照常 commit，不因全局“追杀”而误伤。

RayOrch 使用 GroupFailure 表达 parent-scoped failure。Ray Data reference arm 在模型前把
trigger 标为普通数据，regroup 后丢弃整个 parent；它能达到相同最终文档语义，但无法把
regroup 后才知道的 parent 状态反向用于前序 READY admission。

每个 poison parent 必须满足：

~~~text
1 trigger
+ READY siblings not dispatched
+ inflight siblings computed then discarded
= poison parent page count
~~~

并满足：

~~~text
physical model pages
= all healthy-parent pages
+ inflight poisoned-parent siblings
~~~

## 数据与环境

- Compute：贵阳 4 节点 × 8 H20，共 32 GPU；另有 2 个 CPU worker。
- OCR：32 replicas；真实模型为 MinerU2.5-2509-1.2B。
- Render：128 replicas，每个只请求 1 CPU；并行单位仍是 PDF，不改 renderer UDF。
- Ray：2.51.1；Ray Data 使用 pull-based sort shuffle。
- Daft compatibility gate：0.7.21。
- 代码 HEAD：ff3a1b865765133846066e6548a261a73f77e84f。
- 正式运行前 tracked dirty diff SHA-256：
  97714b29ffc996d3ecd36f55940b03949fbc91490d8550dc28b6c7503689b55c。

输入来自：

- <rayorch-run-root>/inputs/sampled-pdfs-1690
- <rayorch-run-root>/inputs/public-pdfs-2000

冻结 corpus manifest 共发现 3,690 篇：3,689 篇可解析、1 篇 PDFium 格式错误；总大小
1,102,977,830,068 bytes，总页数 174,744。页数分位数为：

~~~text
min / p50 / p90 / p95 / p99 / max = 1 / 24 / 125 / 162 / 236 / 427
~~~

manifest SHA-256：
f743373af9ff9a54944796060cdd423271dae0a41de36abca371aef271b641b7。

最大 parent 位于 manifest index 1,341，共 427 页、2,333,599,794 bytes：

~~~text
<rayorch-run-root>/inputs/
sampled-pdfs-1690/b6b96db8896017d59996308247e2aa0b.pdf
~~~

## 运行时重建与数据面门禁

首次 compute runtime 的控制面健康，但最小的 1 integer / 1 actor RPC 仍阻塞在
_RayBlockStore.get → ray.get；这排除了 PDF、PDFium、DPI、batch 和业务 UDF。
128 个 TCP 检查中只有 8 个失败，恰好是四个 GPU worker 到 Ray head 的
object-manager/node-manager 端口。

重启后仍存在这个方向性限制，因此正式 job 使用一个很窄的拓扑适配：

- Ray Job driver 只负责在 head 提交；
- benchmark driver 作为普通 Ray task 固定到 CPU worker；
- root object 由 CPU worker 持有，render/reduce/OCR actor 仍位于 GPU worker；
- 四个 GPU worker 与两个 CPU worker 均完成共享文件系统实读。

这个适配没有修改 RayOrch、failure 状态机或 renderer UDF。1-page 真实 MinerU
end-to-end gate 完成后才放行正式 arm。

## Stage A：真实 MinerU OCR

输入 manifest 为 128 篇、2,758 页、73,966,227 bytes。首项是 293-page poison parent，
其后为 127 个健康 parent（每篇 8–64 页）。manifest SHA-256：
16b9951e2370ce8ca06492da20027bb24a5d74cca77711a7d18e6a9721b2a2d3。

固定项：200 DPI、128 render replicas、32 OCR replicas、32 reduce replicas、
metadata-only assemble。metadata-only 只省去 Markdown/layout 格式化，真实 OCR 不省略。

### RayOrch admission-window 扫描

| Batch | Arm | 成功文档 | 模型页 | READY 未派发 | Inflight 丢弃 | Measured s | E2E s | 合同 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4 | baseline | 128 | 2,758 | 0 | 0 | 40.486 | 154.921 | pass |
| 4 | poison | 127 | 2,596 | 161 | 131 | 37.467 | 124.036 | pass |
| 8 | baseline | 128 | 2,758 | 0 | 0 | 36.160 | 128.136 | pass |
| 8 | poison | 127 | 2,728 | 29 | 263 | 31.310 | 125.452 | pass |
| 16 | baseline | 128 | 2,758 | 0 | 0 | 32.778 | 127.733 | pass |
| 16 | poison | 127 | 2,757 | 0 | 292 | 30.654 | 125.063 | pass |
| 64 | baseline | 128 | 2,758 | 0 | 0 | 38.024 | 150.199 | pass |
| 64 | poison | 127 | 2,757 | 0 | 292 | 32.842 | 126.852 | pass |

四个 poison arm 都严格满足：

- batch 4：1 + 161 + 131 = 293；
- batch 8：1 + 29 + 263 = 293；
- batch 16：1 + 0 + 292 = 293；
- batch 64：1 + 0 + 292 = 293。

输出均为 127 PRESENT、1 SUPPRESSED、0 FAILED、0 DROPPED；poison parent 输出页为 0，
健康 parent 合计 2,465 页。batch 4 相比等语义 Ray Data 少执行 161 页真实 OCR，
且这 161 页正是 READY gate 的避免派发量。

batch 64 的 72 次 OCR RPC 中只有 34 次达到 64 页，平均 batch 为 38.31，填充率
59.85%；batch 16 的填充率为 92.18%。因此在这组 2,758-page 输入上，batch 64 相对
batch 16 的 measured time 慢 16.0%，E2E 慢 17.6%。单次配对不足以作显著性结论，
但足以否定“32 卡上沿用 batch 64 必然更快”的调参假设。

### Ray Data 等语义 reference arm

| Batch | Arm | 成功文档 | 模型页 | 仅省 trigger | Measured/E2E s | 合同 |
|---:|---|---:|---:|---:|---:|---|
| 4 | baseline | 128 | 2,758 | 0 | 178.384 | pass |
| 4 | poison | 127 | 2,757 | 1 | 182.170 | pass |
| 64 | baseline | 128 | 2,758 | 0 | 165.602 | pass |
| 64 | poison | 127 | 2,757 | 1 | 169.523 | pass |

Ray Data poison arm 也正确输出 127 个健康 parent、2,465 页，并丢弃 293-page poison
parent；但前序 OCR 已处理其 292 个 sibling。Ray Data 的 timer 包含 actor/model 初始化，
而 V3.6 measured timer 不包含 Executor startup，因此跨引擎 measured wall 不可直接比较。

Stage A 的持久 summary 对 V3.6 记录 outcome count、总页数和闭合计数，对 Ray Data 还记录
逐 PDF signature；它没有为 V3.6 持久化逐 PDF identity 列表。因此这里可以声称 parent
count/page count/坏 parent absence 和状态闭合，不把“逐健康 PDF hash 跨引擎相等”写成
超出证据的结论。全量 cardinality arm 对 exact identity 有独立检查。

## Stage B2：全量 cardinality-faithful control

此层使用全部 3,689 个有效 parent 和真实页数分布，共 174,744 page records；每页固定
10 ms synthetic work，32 workers；主矩阵使用 batch 8，后续复核使用 batch 64。它检验
全量状态/工作量记账，不代表真实 MinerU 吞吐。

| Engine | Batch | Poison parent | 成功文档 | 输出页 | 物理模型页 | READY 未派发 | Inflight 丢弃 | E2E s | Exact identity/合同 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V3.6 | 8 | 0 | 3,689 | 174,744 | 174,744 | 0 | 0 | 431.842 | pass/pass |
| Ray Data | 8 | 0 | 3,689 | 174,744 | 174,744 | 0 | 0 | 106.349 | pass/pass |
| V3.6 | 8 | 1（427 页） | 3,688 | 174,317 | 174,743 | 0 | 426 | 435.051 | pass/pass |
| Ray Data | 8 | 1（427 页） | 3,688 | 174,317 | 174,743 | 0 | 426 | 103.645 | pass/pass |
| V3.6 | 8 | top-16（5,233 页） | 3,673 | 169,511 | 172,942 | 1,786 | 3,431 | 419.306 | pass/pass |
| Ray Data | 8 | top-16（5,233 页） | 3,673 | 169,511 | 174,728 | 0 | 5,217 | 104.817 | pass/pass |
| V3.6 | 64 | 0 | 3,689 | 174,744 | 174,744 | 0 | 0 | 286.801 | pass/pass |
| V3.6 | 64 | 1（427 页） | 3,688 | 174,317 | 174,743 | 0 | 426 | 286.288 | pass/pass |
| V3.6 | 64 | top-16（5,233 页） | 3,673 | 169,511 | 174,728 | 0 | 5,217 | 290.316 | pass/pass |

全量 top-16 V3.6 闭合：

~~~text
16 triggers + 1,786 READY + 3,431 inflight = 5,233 poisoned-parent pages
~~~

与 Ray Data 相比，V3.6 少执行 1,786 页物理 work；相对 V3.6 baseline 共减少 1,802 页，
其中另 16 页是所有引擎都跳过的 trigger。单最大 parent arm 没有 READY saving，原因是
10 ms work 很短且其出现位置使 sibling 在 failure 可见前已经 inflight；它只验证 commit
isolation。这个负点确认了收益取决于调度位置和 work duration。

batch 64 的 top-16 闭合为：

~~~text
16 triggers + 0 READY + 5,217 inflight = 5,233 poisoned-parent pages
~~~

源码约束是每个 actor 同时最多一个 pending RPC，并非无界 actor mailbox。这里的关键是
每个 parent 完整归属于一个 32-PDF microbatch，而 32 actors × 64 pages 的单轮容量为
2,048 pages，足以在首个 failure commit 前耗尽该 microbatch 的 OCR READY queue。增加坏
parent 数量只是在多个 microbatch 重复同一现象，不会把它们合成一个共享 READY 池。

因此，当前语料（单 PDF 最大 427 页）在 batch 64 × 32 actors 下，实测从 1 个到 top-16
坏大 PDF 均能正确识别和隔离：top-16 的 16/16 parent、5,233/5,233 页都没有进入下游。
但它拦住 **0 个 READY sibling OCR**，仅跳过 16 个 trigger，另外 5,217 页算完后在 commit
门禁丢弃。若保持 parent pages 连续排队，单个 parent 需要超过约 2,048 页才可能跨过首轮
inflight horizon；这是必要而非充分条件，仍受触发页位置和调度顺序影响。无需据此新增
runtime 实体或 admission 机制：现有 batch 参数已经能表达吞吐—隔离取舍，应由 workload
选择较小 batch。

单点全量规模不能证明渐进 O(N)；报告只声称 174,744 records 下合同完成且 work accounting
闭合，不从一个规模点外推复杂度斜率。

## Daft compatibility 结果

重启后再次执行 3-page、no-model、reference-only gate，仍停在
InMemoryScan → ... → HashAggregate。Daft 的 RemoteFlotillaRunner 位于 Ray head，而
HashAggregate task 位于 GPU worker；当前 GPU→head 的数据端口不可达。remote benchmark
driver 无法改变 Daft 内部 runner placement。

因此未启动 Daft 正式 arm。结论是“当前 compute/runtime topology 不兼容”，不是
“Daft 更慢”或“Daft 无法表达用户态 drop-parent”。

## 为什么没有跑 1.10 TB 全量实体 OCR

当前 MinerU 输入 UDF 会先把整本 PDF 以 200 DPI eager rasterize 后再返回 page list。
对 1.10 TB corpus 做全量实体 OCR，首要测量会变成共享文件系统、PDF decode、对象存储和
whole-document materialization，而不是 parent failure barrier。

本轮采用两层正交证据：

1. p99 以上的 293-page parent + 127 健康文档，跑真实 render 和真实 32-GPU MinerU；
2. 全部 3,689 parent/174,744 页，跑 cardinality-faithful 状态与工作量控制。

因此不能写“已完成 1.10 TB 全量真实 MinerU OCR”。若论文未来需要这个吞吐结论，应单独
设计 streaming/page-source benchmark，而不是为本 claim 临时扩大 runtime 抽象。

## 可写与不可写的论文结论

可以写：

- typed parent failure 在 READY、late commit 和最终 outcome 上共享同一 lineage predicate；
- inflight sibling 不追杀，但其晚到结果不会被下游消费；
- 健康 parent 正常 commit，坏 parent 精确 SUPPRESSED；
- 当 failure 可见时仍有 READY sibling，RayOrch 能减少物理模型工作；
- batch/admission window 构成可量化的吞吐—隔离 Pareto。

不可写：

- poison 一定带来大量 compute saving；
- V3.6 普遍快于 Ray Data；
- 单次 wall-time 是统计显著 speedup；
- 已完成 1.10 TB 全量真实 OCR；
- Daft 算法性能落后；
- 单个全量 cardinality 点证明 O(N) 渐进复杂度。

## 回归与可复现材料

正式运行前与结果收口时：

- 4 个直接 benchmark 文件：34 passed；
- V3.6 非 Ray-integration：156 passed；
- 本轮 batch-64 收口的非 Ray 聚焦回归（failure、transition、compiler、RayOrch/Ray
  Data/Daft/cardinality benchmark）：103 passed、1 skipped。
- 同一命令纳入 17 个本机 Ray integration 后，这 17 项均在 fixture 的 Ray 初始化阶段因
  容器 `psutil.NoSuchProcess(pid=2)` 报错，未执行到测试断言；正式集群合同结果不受影响。

正式集群完成的 21 个结果 JSON 全部通过各自合同；真实 MinerU 1-page remote-driver gate
也通过。完整机器可读配置位于 artifacts/2026-08-27_large_pdf_poison/run_matrix.json，
结果位于 artifacts/2026-08-27_large_pdf_poison/results/，摘要哈希位于同目录
SHA256SUMS。

历史 pre-restart 失败 probe 保留在 artifacts/preflight；它们用于证明最初阻断来自 Ray
head→worker root ObjectRef 数据面，而不是 renderer、DPI 或 failure 状态机，不进入正式
性能表。

完成结果持久化与哈希复核后，compute
8a2b635e-b9a0-4d12-972b-b92980b82a6c 先进入 terminated/provisioned=false，
再执行删除；最终 get-compute 返回 404。batch-64 复核 compute
3a49cf00-d7da-4ace-8b4c-2ba92b336701 也以相同顺序 deprovision、删除并得到 404，
确认两轮 32 张 H20 与 compute 记录均已释放，且两者从未同时运行。
