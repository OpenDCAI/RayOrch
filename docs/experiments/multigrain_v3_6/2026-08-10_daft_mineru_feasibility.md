# V3.6 × Daft MinerU 公平性与可行性实验

日期：2026-08-10
性质：论文前置 feasibility；不是可直接投稿的最终统计结果。

## 1. 本轮要回答什么

这轮只回答四个窄问题：

1. Daft 能否用自身自然 API 正确表达同一个 `PDF → Page → OCR → PDF` workload？
2. 在完全相同的 MinerU UDF、输入、4×H20 和 batch cap 下，V3.6 是否至少没有明显退化？
3. 若出现差异，初步证据更接近 batching、生命周期/分区流水线，还是 regroup 数据移动？
4. 保持 workload 语义不变、只改变 regroup representation 时，full-value payload 是否具有
   可重复的因果成本？

它不回答“RayOrch 普遍快于 Daft”。正式论文仍需冻结代码后做交错顺序、至少三次重复、
均值/方差、资源遥测和多 workload 验证。

## 2. 环境与实现

| 项目 | 值 |
| --- | --- |
| RayOrch commit | `813347a5eb96f90ffffd666adbcd1ceb3d1d89c4`，外加本轮未提交 Daft runner |
| 分支 | `codex/multigrain-recovery-tiers` |
| Python | 3.12.11 |
| Ray | 2.50.0 |
| PyArrow | 21.0.0 |
| Daft | 0.7.21 |
| PyTorch / CUDA | 2.7.1 / 12.9 |
| vLLM endpoint | `${VLLM_ENDPOINT}` |
| GPU | 4×NVIDIA H20 |
| Driver | 535.247.01 |
| CPU / object store | 32 CPUs / 100 GiB |
| OCR replicas / batch cap | 4 / 64 |
| Render / assemble replicas | 4 / 4 |
| 数据 | 同一排序后的 Flash-MinerU throughput manifest；368 PDFs / 7,072 pages |
| PDF 文件名清单 SHA-256 | `d0b7ecbd2062cf862adac1bcf736eff7fabe4e27f3bfad813117371d79b23f18` |
| 排序后 `content SHA-256 + relative path` manifest SHA-256 | `5b1a5a486fc2c0b81bea070245ae6ece87b9effa9ddfdaddd012f43096993a04` |
| Flash-MinerU commit | `7246a353554cee35674d7ef82f2c061db2f43003`；工作区非 clean，tracked diff 仅 `.gitignore` |
| Model path | `MinerU2.5-2509-1.2B`；本轮未记录权重 digest，正式 artifact 必须补 |

输入内容审计显示，368 个路径只有 **22 个不同 PDF checksum**：大多复制 16 次，其中一个
checksum 出现 32 次。因此它是可复现的 replicated throughput/skew workload，不是 368 个
独立文档的 diversity corpus。论文可用它做系统吞吐、packing 和长尾实验，但必须另补公开、
有 license、无人工复制的多样文档集来支撑 generality。

Daft runner 位于
`rayorch/experimental/multigrain_v3_6/benchmark/mineru_daft.py`。它没有模仿 RayOrch
内部协议，而采用 Daft 0.7.21 的自然写法：

```text
from_pydict
  → class UDF render
  → explode(page)
  → batched GPU class UDF OCR
  → groupby(parent_id).map_groups(assemble)
```

Daft physical plan 明确包含 `RayShuffle: Hash`。Daft 0.7.21 会把 Python `dict` 推断成
Struct，但当前 class UDF 的 Struct→Python cast 不可用，因此 runner 使用只包一层的 opaque
Python dataclass 携带完全相同的 page/content payload。这不是 RayOrch 引用协议；自然 Daft
基线仍在 regroup 中携带完整值。

为验证这一点，runner 另有 `reference_only` **专家控制臂**。它把每个 OCR batch 的完整
page/content 粗块交给显式 owner actor，只让 `store_slot/block_id/payload_row` manifest 进入
同一个 hash groupby，assemble 再 `acquire/get/release`。它用于因果分析，不是推荐的 Daft
生产实现；其 owner RPC 和生命周期开销必须计入 wall time，不能隐藏。

## 3. 公平比较口径

- 两个系统复用同一 `MinerUPdfToPages`、`MinerUVlmOcrPage`、`MinerUAssembleDoc` 实现；
- 输入 PDF 顺序、DPI、模型、GPU memory utilization、replica 和 batch cap 相同；
- 主比较使用 post-`ray.init()` run wall：包含 actor/model startup 至全部输出收集完成，
  但两个 runner 都不计 Ray cluster init；
- V3.6 run wall 包含 `Executor.close()`，Daft run wall 不含 `ray.shutdown()`；本轮 V3.6
  368 的 close 差值仅 `617.222 - 39.033 - 578.186 = 0.003s`，但正式 schema 仍应把
  execution 与 teardown 独立报告；
- V3.6 的 `measured_wall_s` 排除了 actor/model startup，不能直接和 Daft 总时间比较；
- 每个 parent 必须恰有连续 page ordinal `0..n-1`；
- Markdown 用 token Jaccard 比较，以容纳 GPU 模型的轻微非确定性；
- Daft natural 的 source partitions 默认为 4，并额外扫描 8/16/32，不能故意保留弱配置；
- 当前执行顺序固定为 Daft→V3.6，缓存和温度可能偏向后运行的 V3.6。

## 4. 已完成结果

### 4.1 CPU 表达与结构 smoke

Daft smoke 在 4 PDFs / 48 pages 上得到 4 个输出，document identity 与 page ordinal 完全
一致；计划中存在 `ActorUDF → Explode → ActorUDF → RayShuffle: Hash → Group-By`。

### 4.2 真实 4×H20 gate

| Scale | Arm | Run wall | OCR RPC | Mean pages/RPC | Fill | Correctness |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 4 PDFs / 48 pages | V3.6 elastic | 55.498s | 4 | 12.00 | 0.1875 | reference |
| 4 PDFs / 48 pages | Daft pre-audit, p=4 | 65.641s | 4 | 12.00 | 0.1875 | Jaccard min 0.9941, mean 0.9963 |
| 48 PDFs / 992 pages | V3.6 elastic | 116.138s | 20 | 49.60 | 0.7750 | reference |
| 48 PDFs / 992 pages | Daft pre-audit, p=4 | 161.650s | 18 | 55.11 | 0.8611 | Jaccard min 0.9892, mean 0.9958 |

上述旧 runner 曾为确定输出顺序在 groupby 后增加全局 `sort(parent_id)`。physical plan 显示
该操作需要额外 range repartition，而 MinerU 并不要求不同 PDF 的全局输出顺序。公平性审计
后已删除该操作，改为 collect 后在 driver 上对 4/48/368 个小结果排序做 correctness。
所以以下单次比值只记录探索过程，不能作为竞品数字：

- 4 PDFs：`65.641 / 55.498 = 1.183×`；
- 48 PDFs：`161.650 / 116.138 = 1.392×`。

### 4.3 目前最重要的观察

48-PDF Daft 的 packing 实际上更好：它比 V3.6 少 2 次 OCR RPC，平均 batch 和 fill 都更高，
但 run wall 仍慢 45.512s。因此这轮**不能**写“Daft 不会跨 parent batching”；相反，它证明
Daft 能做高质量 batching。性能差异若能在重复实验中保持，更可能来自：

1. render、GPU actor 初始化和 OCR 的流水重叠方式；
2. source partition 的长尾与 actor feed 方式；
3. groupby 时完整 page/content payload 的 hash shuffle 与 assemble 尾段。

阶段时间线和 8/16/32 partition sweep 用于区分前两项；shuffle bytes、object-store/network
telemetry 与 payload-size sweep 才能验证第三项。仅凭 wall time 不能确定因果。

## 5. 368-PDF 与分区扫描

Daft natural `p=4` 已完整跑通 368 PDFs / 7,072 pages：

| Arm | Run wall | OCR RPC | Mean pages/RPC | Fill | OCR batch histogram |
| --- | ---: | ---: | ---: | ---: | --- |
| Daft pre-audit, p=4 | 894.866s | 113 | 62.58 | 0.9779 | 8×1, 12×1, 28×1, 48×1, 64×109 |

该 368 数字同样包含不必要的最终 sort，仅证明 full-scale correctness，不进入公平主表。
Daft 的 OCR packing 已接近 batch cap，说明后续若 V3.6 仍更快，主张必须落在端到端流水和
语义感知 regroup，而不能落在“只有 RayOrch 能跨 parent batching”。

V3.6 同配置 run wall 为 617.222s，其中 actor/model startup 39.033s、业务执行
578.186s；121 个 OCR RPC，平均 58.45 pages/RPC，fill 0.9132。它相对仓库中的 V3 golden
measured time 比值为 0.9837，处于 5% regression gate 内。

按 48-PDF sweep 预先选出的 Daft tuned p=8 full-scale 结果为：

| Arm | Run wall | Run throughput | OCR RPC | Mean pages/RPC | Fill |
| --- | ---: | ---: | ---: | ---: | ---: |
| V3.6 elastic | 617.222s | 11.458 pages/s | 121 | 58.45 | 0.9132 |
| Daft tuned, p=8 | 833.798s | 8.482 pages/s | 114 | 62.04 | 0.9693 |

单次 paired ratio 为 `833.798 / 617.222 = 1.351×`，即 V3.6 run wall 短 26.0%。Daft
用了更少 RPC、更满的 batch，仍然更慢，再次排除“RayOrch 只是 packing 更好”这一解释。
这仍是固定顺序的一次 feasibility，论文主表必须采用 balanced repeated matrix。

Daft tuned timeline 为：first/last render 6.399/108.688s，first actor init 93.515s，all OCR
actors ready 159.579s，first/last OCR 130.827/719.231s，post-OCR collect 114.567s。约 1 秒
GPU 采样进一步显示：

| Window | Arm | Mean GPU util | GPU-sample idle <10% | All-4 idle <10% | All-4 active ≥10% |
| --- | --- | ---: | ---: | ---: | ---: |
| Whole run | V3.6 | 54.00% | 22.39% | 7.13% | 50.26% |
| Whole run | Daft p=8 | 39.62% | 42.63% | 29.09% | 33.90% |
| All four models loaded | V3.6 | 56.78% | 18.32% | 2.39% | 52.85% |
| All four models loaded | Daft p=8 | 55.12% | 19.97% | 3.32% | 48.15% |

计算口径：对约 1 秒采样的所有 `(sample, GPU)` utilization 求算术平均；单 GPU idle 为
`utilization < 10`；四卡同时 idle/active 分别为每个 sample 的 `max < 10` / `min ≥ 10`；
“四模型均已加载”要求该 sample 的四卡 `memory_used` 都不低于 70,000,000,000 bytes。
V3.6/Daft 原始 sample 数为 575/770，monotonic span 为 615.764/832.391s。

两者在“四模型均已加载”窗口的跨度也几乎相同：V3.6 580.635s、Daft 582.190s。
因此当前最合理的**推断**不是 RayOrch 的 OCR kernel 或 steady-state batching 更快，而是它
减少了全流程边界上的 actor readiness、materialization 和 terminal regroup 空洞。要把推断
升级为因果结论，仍需 operator-level movement、object-store 和网络 telemetry。

严格结构 gate 为 368/368 document identity 和 7,072/7,072 page count。Markdown 的单次
V3.6↔Daft Jaccard 为 min 0.7503、mean 0.9910、median 0.9954，364/368 ≥0.95。
低分样本需要结合模型自漂移解释：

| Pair | Min | Mean | Median | ≥0.95 |
| --- | ---: | ---: | ---: | ---: |
| V3.6 current ↔ Daft tuned p=8 | 0.7488 | 0.9915 | 0.9954 | 361/368 |
| Daft tuned p=8 ↔ Daft pre-audit p=4 | 0.9354 | 0.9922 | 0.9960 | 366/368 |
| Daft tuned p=8 ↔ V3.5 historical | 0.9357 | 0.9924 | 0.9957 | 363/368 |
| V3.6 current ↔ Daft pre-audit | 0.7503 | 0.9910 | 0.9954 | 364/368 |
| V3.6 current ↔ V3.5 historical | 0.7589 | 0.9928 | 0.9961 | 366/368 |
| Daft pre-audit ↔ V3.5 historical | 0.9520 | 0.9918 | 0.9952 | 368/368 |

最差项 `hyper-connection_copy_7` 在 V3.6↔Daft 为 0.7503，在 V3.6↔V3.5 同样只有
0.7589；这些 `_copy_*` 的 PDF checksum 完全相同。这说明当前 outlier 至少可由同一模型的
调度相关非确定性解释，不能归罪于 Daft。Daft p=8↔p=4 自身也有 min 0.9354 的调度相关
漂移。正式 correctness 应把严格 identity/order/layout 结构与非确定 Markdown 质量分开，
并报告同系统自漂移基线。

修正版 48-PDF partition sweep 当前结果：

| Partitions | Run wall | First OCR | All OCR actors ready | Last OCR | Post-OCR collect | RPC / fill |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 163.213s | 60.457s | 71.890s | 136.499s | 26.714s | 18 / 0.8611 |
| 8 | 149.276s | 54.586s | 57.122s | 125.605s | 23.671s | 21 / 0.7381 |
| 16 | 149.361s | 52.078s | 54.884s | 125.898s | 23.463s | 22 / 0.7045 |
| 32 | 152.443s | 51.526s | 51.840s | 133.790s | 18.653s | 32 / 0.4844 |

修正版 physical plan 为 `global_output_sort=false`。它比含额外 sort 的旧 p=4 run 反而慢
1.563s，差值属于单次噪声量级，也说明旧全局 sort 不是此前差异的主要来源。timeline 的
`post_ocr_to_collect_end` 包含可能与 OCR 重叠的 hash regroup/assemble 尾段，不能直接解释为
纯 shuffle 时间；正式因果实验仍需 operator-level telemetry。

p=8 比 p=4 快 13.937s（8.5%），虽然它增加 3 个 OCR RPC 并降低 fill。当前证据表明，
更细 source partitions 带来的 actor feed/负载均衡收益超过 batch fragmentation 代价；不能把
系统优劣压缩成一个平均 batch size。

p=16 与 p=8 仅差 0.085s，属于单次噪声，但 p=16 多一次 RPC、fill 更低。p=32 又慢
3.167s，RPC 增至 32、fill 降到 0.4844，且最后 render 从 p=16 的 24.212s 推迟到
63.225s。调参曲线已经出现拐点，因此 tuned 候选选择 p=8，而不是对近似并列结果做事后
择优。p=8 与 V3.6 的 48/48 文档全部匹配，Jaccard min 0.9892、mean 0.9956、median
0.9961，全部 ≥0.98。

## 6. Regroup representation 控制变量实验

### 6.1 干预与不变量

两臂固定相同的 368 条输入、7,072 个真实渲染页、p=8、32 CPUs、100 GiB object store、
render/OCR/parent identity/page ordinal/hash key/Reduce concurrency 和输出排序。唯一干预点是
OCR 边界之后进入 `RayShuffle: Hash` 的行表示：

| Arm | 进入 hash regroup 的值 | 完整 page/content 在哪里 |
| --- | --- | --- |
| `full_value` | `parent_id + pdf_path + page + content` | Daft 行本身 |
| `reference_only` | `parent_id + pdf_path + manifest` | owner actor 持有的粗粒度 Ray object block |

physical plan 验证 `reference_only` 在 shuffle 前已投影掉 `page`；两臂都保留同一个
`RayShuffle: Hash` 和同一个 parent group。这里的 object bytes 是 owner actor 记录的完整
payload object 体积，**不是** Daft 或网络报告的 shuffle bytes。

### 6.2 纯 payload 路径：因果成立

首先用真实 PDF render 和完整 page payload，但将 GPU OCR 改为 identity、终端 assemble 改为
metadata-only。这样保留约 74.4 GiB page payload 的生产、分区与 regroup，同时去掉模型和
Markdown 业务计算。按 `full/ref → ref/full → full/ref` 平衡顺序完成三组配对：

| Repeat | Full-value | Reference-only | Paired difference | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 517.377s | 406.370s | 111.007s | 1.273× |
| 2 | 514.748s | 399.898s | 114.850s | 1.287× |
| 3 | 495.355s | 406.581s | 88.774s | 1.218× |
| Mean | 509.160s | 404.283s | 104.877s | 1.260× |

full/reference 的样本标准差分别为 12.028s/3.799s；paired difference 的样本标准差为
14.077s，t(2) 95% CI 为 `[69.907s, 139.847s]`。按 arm mean 计算，reference-only 时间
短 20.6%。六次 `outputs.json` 的 SHA-256 均为
`e0cff352e2556d18326dc5d7dd5ebd9d654df72448da1f61fa88b5c511d00c05`，最后 render 时间在
每个 pair 内也近似相同。因此在该单机、约 74.4 GiB payload 的受控 workload 上，
**让完整值参与通用 hash regroup 确有显著且可重复的因果成本**。

该成本不应被错误读成“最后一个 OCR 结束后的纯 shuffle tail”。full-value 表示会在上游
repartition write、object-store backpressure 和 regroup 全阶段施加成本；第一组中最后一个
identity OCR 分别在 370.656s/286.334s 完成，说明影响早在 terminal tail 之前已经发生。

### 6.3 真实 MinerU：简单外挂引用协议不构成端到端优化

随后保留真实 4×H20 OCR 与完整 `MinerUAssembleDoc`，执行一组同配置 paired trial：

| Arm | Run wall | Last OCR | First assemble | Last assemble | Post-assemble |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-value | 831.762s | 710.439s | 745.184s | 831.405s | 0.356s |
| Reference-only | 848.956s | 719.300s | 725.005s | 848.675s | 0.280s |

reference-only 反而慢 17.194s（2.1%）。两臂的 368 document identities、7,072 page
identities/counts 完全相同；Markdown Jaccard min/mean/median 为
0.9276/0.9926/0.9966。reference arm 持有 114 个 block、79,896,099,579 bytes payload，
结束时四个 store 的 live blocks/rows/bytes 均为 0。

负结果原因可直接从时间线看到：reference arm 更早开始 assemble，但显式
`owner.acquire → ray.get(block) → owner.release` 使 assemble span 达 123.670s，而自然
full-value 为 86.221s。48-PDF metadata-only 对照也同向：full 149.075s，reference
162.629s。也就是说，**把引用管理外挂到 Daft 用户代码会以另一种开销抵消 regroup 收益**。

这组真实试验只有一个 pair，因此只作为 negative feasibility，不作统计显著主张。但它足以
否定两个过强解释：不能把原先 Daft/V3.6 的 1.351× 全部归因于 terminal shuffle，也不能
声称“在 Daft 中换成 ObjectRef 就能获得 RayOrch 的收益”。RayOrch 当前更合理的系统解释是
一组协同设计：first-class lineage/reference identity、在线 completion/ordered Reduce、直接
worker materialization、规范 ownership/release 和 actor lifecycle；每项贡献仍需独立 ablation。

### 6.4 End-to-end critical-path 分解

约 1 秒 GPU 样本以“四卡 memory 均 ≥70GB”标记共同驻留，以最后一个任意 GPU
`utilization ≥10%` 样本标记 GPU 工作结束。该口径不是 kernel profiler，但三个区间与 run
wall 闭合，适合定位边界：

| Critical-path phase | V3.6 | Daft full | Daft − V3.6 |
| --- | ---: | ---: | ---: |
| Run start → four models resident | 35.129s | 142.200s | +107.071s |
| Four resident → last GPU activity | 578.522s | 567.002s | −11.520s |
| Last GPU activity → run end | 3.571s | 122.560s | +118.989s |
| End-to-end | 617.222s | 831.762s | +214.540s |

即 `107.071 − 11.520 + 118.989 = 214.540s`。Daft 的 GPU 主工作区间没有更慢，完整差距
集中在 ramp-up/lifecycle 与 terminal boundary。前段还包含 Daft 部分 Render 和局部 OCR，
不能全部命名为模型初始化；正式 baseline 应增加 Daft 官方可行的 prewarm arm。

Daft terminal 可进一步拆成：

| Daft full terminal interval | Time | Meaning |
| --- | ---: | --- |
| Last OCR → first Assemble | 34.745s | 剩余 regroup、group readiness 与排队 |
| First → last Assemble | 86.221s | 真实 `MinerUAssembleDoc` 业务工作 |
| Last Assemble → collect | 0.356s | 最终收集 residual |
| Last OCR → collect | 121.323s | 不是纯 shuffle tail |

V3.6 最后 GPU activity 后仅余 3.571s，而两边复用同一个 Assemble UDF；这强烈支持 V3.6
已把绝大多数 per-parent Assemble 与尚未完成的 OCR 在线重叠。Daft natural 的第一个
Assemble 晚于最后 OCR，说明本 workload 中 `RayShuffle: Hash → GroupBy` 构成了有效阶段
边界。仍需 `online Reduce on/off` 同底座消融才能给 overlap 单项赋予最终因果倍率。

### 6.5 四卡利用、payload 与 RPC/blocking 总表

GPU 主工作窗口证明两边都使用四个 GPU actor，但不应写成“始终 100% 吃满”：

| Active-window metric | V3.6 | Daft full | Daft reference |
| --- | ---: | ---: | ---: |
| Mean GPU utilization | 56.99% | 56.99% | 52.26% |
| Mean active GPUs (`util ≥10%`) | 3.279/4 | 3.321/4 | 3.029/4 |
| All four active | 53.05% | 47.93% | 36.40% |
| All four idle | 2.03% | 0.19% | 1.91% |
| GPU-active span | 578.522s | 567.002s | 619.189s |

因此 V3.6 与 Daft full 的 steady-state OCR 供给等价；V3.6 的优势不来自更快 kernel 或更少
OCR RPC。Reference arm 在完全相同的 114 RPC/batch histogram 下 active span 增加约
52.2s、四卡同时 active 降低，符合同步 payload publish 阻塞 GPU actor 复用的源码关系。

| Payload fact | Measured value | Interpretation |
| --- | ---: | --- |
| Page-only object volume | 74.373 GiB | CPU identity-OCR arm，基本由 PIL page images 构成 |
| Mean page payload | 10.769 MiB/page | 7,072 pages |
| Real page + OCR content | 74.409 GiB | 真实 reference arm 的 114 owner blocks |
| Mean payload per OCR block | 668.376 MiB | `79,896,099,579 / 114` |
| Mean payload per PDF | 207.051 MiB | `79,896,099,579 / 368` |
| OCR content incremental size | 37.185 MiB total | 约 5.384 KiB/page；regroup 几乎由 page image 主导 |

这些是 Ray object volume，不是 NIC 或 Daft shuffle telemetry。Natural full-value plan 让约
74.4 GiB 逻辑 page/content 参与 hash repartition；实际物理 read/write bytes 仍需引擎指标。

| Path | Known application RPC/block count | Large-payload boundary | Critical blocking relation |
| --- | --- | --- | --- |
| V3.6 | 642 execute：Render 368、Metadata 16、OCR 121、Assemble 137 | Worker 本地粗块 `ray.put`，下游按 RowBinding materialize | Engine commit/advance 在 driver 本地；per-parent Assemble 可在其他 OCR 运行时 ready |
| Daft full | OCR 114；内部 render/shuffle/group task RPC 未暴露 | 约 74.4 GiB 完整行进入 `RayShuffle: Hash` | 第一个 Assemble 晚于最后 OCR 34.745s；groupby 是有效 barrier |
| Daft reference | OCR 114 + Store 114 + `2I` acquire/release；另有 `I` 次 block get | 每次 Store publish 平均 668.376 MiB | GPU actor 内同步 `ray.get(store.remote(payload))`；随后按 block-parent incidence 获取 |

其中 `I = Σ distinct_parent_per_ocr_block`，当前 artifact 未记录；由 368 PDFs / 7,072 pages
只能给出 `368 ≤ I ≤ 7,072`，故 reference owner actor RPC 为 `850 ≤ 114+2I ≤ 14,258`。
这张表也解释了为什么固定 RPC latency 不是主因：RPC 更多的 V3.6 仍更快，关键是每个边界
携带/依赖的 bytes、是否重复 materialize，以及是否阻塞下一批或全局 Reduce。

### 6.6 写作用证据—结论映射

| Question | Direct evidence | Supported conclusion | Wording boundary |
| --- | --- | --- | --- |
| 是否因 V3.6 batch 更好？ | Daft 114 RPC/fill 0.969；V3.6 121/0.913 | 否；Daft packing 更好 | 不写“只有 RayOrch 能跨 parent batch” |
| 是否因 OCR kernel 更快？ | 两者 active-window mean GPU util 均 56.99%；Daft active span 还短 11.52s | 否；steady-state OCR 不是主差距 | 不把 whole-run utilization 当 kernel speed |
| 总 wall 慢在哪里？ | +107.071s ramp-up、−11.520s GPU 主窗口、+118.989s terminal | 差距集中在 lifecycle 和 terminal boundary | 前 107s 不全是 model init；后 119s 不全是 shuffle |
| Full-value regroup 是否有因果成本？ | Payload-only 三组 paired mean 1.260×、95% CI 不跨 1、输出 exact | 是，在该 74.4 GiB 单机 workload 上成立 | 不外推为真实 MinerU 的全部 1.351× |
| V3.6 为什么 terminal 小？ | Daft Assemble 全在 last OCR 后；V3.6 last-GPU→end 仅 3.571s；运行时支持 per-parent READY | 强证据支持 online Reduce/Assemble overlap | 最终倍率仍需 V3.6 on/off ablation |
| Reference 为何更慢？ | 同 114 batches，GPU active span +52.2s；源码同步 publish + acquire/get/release | 用户态 reference protocol 阻塞并增加物化 | 不写成 ObjectRef 本身慢 |
| 是否主要为固定 RPC latency？ | V3.6 642 execute RPC 仍更快；单 block 平均依赖 668 MiB | 否；bytes、materialization 与 barrier 更重要 | Daft 内部总 RPC 未观测，不能比较精确总数 |

### 6.7 Takeaway：V3.6 到底好在哪里

本轮 368-PDF run 的时间差可以直接记成一张账：

| Time account | V3.6 | Daft full | V3.6 gain |
| --- | ---: | ---: | ---: |
| Run start → four GPU models resident | 35.129s | 142.200s | +107.071s |
| Four resident → last GPU work | 578.522s | 567.002s | −11.520s |
| Last GPU work → all documents finished | 3.571s | 122.560s | +118.989s |
| Net | 617.222s | 831.762s | **+214.540s** |

技术上就是两胜一负：

1. **启动赢 107.1s。** V3.6 在执行前并行构造 4 个持久 OCR actors，并等待统一
   `ready()`；Daft natural 到 Render 后段才 lazy 创建 GPU actors。这个数字仍需 Daft
   prewarm arm 判断有多少属于可消除的 baseline 启动策略。
2. **GPU 主段反而输 11.5s。** Daft 114 个 OCR RPC、fill 0.969；V3.6 121 个、fill
   0.913。两边 active-window mean GPU utilization 都是 56.99%，所以 V3.6 没有靠 OCR
   kernel、更多 GPU 或更好 batching 获胜。
3. **收尾赢 119.0s。** Daft 最后 OCR 后先等 34.745s 才进入 Assemble，再用 86.221s
   完成全部文档；V3.6 最后 GPU 工作后只剩 3.571s，说明相同 Assemble 工作已经在其他 PDF
   做 OCR 时按 parent 在线执行。Daft 的 `RayShuffle: Hash → GroupBy` 在本 run 中形成了
   OCR/Assemble barrier。

74.4 GiB full-value regroup 的技术位置也很明确：它参与 Daft 的上游 repartition/backpressure，
并位于最后 OCR 到第一个 Assemble 的 34.745s handoff 内；payload-only 三组控制实验已证明
它本身有成本，但不能把全部 119s terminal 或 214.5s 总差距都记到 shuffle 名下。

论文摘要级表述应是：

> **On this run, RayOrch saves 107.1s before all four GPU models are resident and 119.0s after**
> **the last GPU work, while Daft is 11.5s faster in the GPU-active interval. The net 214.5s**
> **advantage comes from eager actor readiness and overlapping per-document assembly with ongoing**
> **OCR; it does not come from faster OCR or better batching.**

边界：这是当前单机 4×H20、replicated 368-PDF feasibility 的最强解释。`1.351×` 仍需重复
matrix；约 107s lifecycle 需 competitor prewarm；online Reduce 需同底座 on/off ablation；
74.4 GiB 是 object volume 而非网络字节。

### 6.8 竞品 API 能力与公平结论

Daft natural runner 没有手写跨 parent batch scheduler。它使用的核心组合均为 Daft 0.7.21
原生 API：

| Logical need | Daft natural expression | Proven behavior | Ownership |
| --- | --- | --- | --- |
| PDF → pages 1:M | class UDF returns list + `.explode("page")` | 7,072 page rows，identity/ordinal exact | Daft native |
| Cross-parent OCR batching | `@daft.method.batch(batch_size=64)` | 114 RPC、mean 62.04、fill 0.9693；107/114 batches 为 64 | Daft native |
| Four GPU workers | `@daft.cls(gpus=1, max_concurrency=4)` | 四卡 active-window mean util 56.99% | Daft native |
| Pages → PDF M:1 | `.groupby("parent_id").map_groups(...)` | 368 groups/documents correct | Daft native |
| Parent identity | explicit `parent_id` column | 用户维护并参与 hash key | Runner/user |
| Page ordinal | `page_id` field + group-local sort/validation | `0..n-1` exact | Runner/user |
| Reference-only owner protocol | Store/Manifest/acquire/release | 只用于控制变量且真实 workload 为负结果 | RayOrch experiment code，非 Daft natural |

因此不能声称“Daft 不能打穿 parent batching”或“Daft API 无法优雅表达 1:M/M:1”。其 natural
graph 对 DataFrame 用户是简洁、合理的：

```text
with_column(render) → explode(page) → with_column(batched OCR)
→ groupby(parent_id).map_groups(assemble)
```

双方真正差异不是逻辑图能否表达，而是展开后的关系在运行时如何存在：

| Runtime question | Daft natural | V3.6 |
| --- | --- | --- |
| 一个 page 属于哪个 PDF？ | `parent_id` 普通列 | `EntityRef` parent/child lineage |
| 页序是什么？ | payload 中的 `page_id`，用户 sort | Expansion occurrence/ordinal |
| 一个 PDF 何时完整？ | hash groupby 形成完整 group 后可见 | Expansion closed 且 member Items terminal 时立即可知 |
| 何时运行 Assemble？ | 当前 physical plan 在全局 OCR 后进入 GroupBy/Assemble | 对每个 parent 独立 READY，可与其他 OCR 重叠 |
| 如何携带大值？ | natural groupby row 携带 page/content | control state 保存 binding；consumer 按需 materialize |

**公平结论：Daft 原生已经能简洁、高效地完成 1:M 展开和跨 parent batching；V3.6 的可测优势**
**来自保留 parent/child/ordinal/completion 语义，使相同逻辑 workload 获得 per-parent online**
**Reduce，而不是来自 Daft 缺少 `explode` 或 batch API。**

## 7. 当前可以与不可以声称什么

可以：

- Daft 是可运行、语义正确且会有效 batching 的直接竞品，不是纸面 baseline；
- V3.6 在同资源 368-PDF 单次 gate 中通过自身 regression gate，并相对 Daft tuned p=8
  暂快 1.351×；
- Daft 的 OCR packing 更好，而全运行 GPU idle 比例和边界时间更高；当前证据支持继续验证
  lifecycle/materialization/regroup 假设；
- Daft natural plan 使用 full-value hash regroup，而 RayOrch 保留 parent/ordinal 语义关系；
- 在 368-PDF、约 74.4 GiB payload-only 控制实验中，full-value representation 相对轻量
  manifest 有可重复的因果成本：三组配对 mean ratio 1.260×，paired 95% CI 不跨 1；
- 简单外挂的 Daft reference protocol 在真实 MinerU 中没有加速，说明一等语义、调度和
  ownership 的协同设计比“换一种行表示”更重要；
- 当前结果足以支持继续投入正式竞争实验。

不可以：

- “RayOrch 普遍比 Daft 快”；
- 把单次 1.351× 写成稳定或统计显著的 speedup；
- “Daft 不能跨 parent batch”；
- “Daft 无法简洁表达 1:M/M:1”；
- 把 V3.6 排除 actor startup 的 `measured_wall_s` 与 Daft run wall 对比；
- 把 payload-only 的 1.260× 外推为真实 MinerU 的收益；
- 把 Daft/V3.6 的 1.351× 或全部 post-OCR tail 归因于 full-value regroup；
- 把 79.9GB owner object size 写成网络或 shuffle telemetry；

## 8. 正式实验升级清单

1. 冻结 runner、数据 manifest、模型 revision 和容器；
2. 每个 arm 至少 3 次，AB/BA 交错顺序并报告均值、样本标准差和 paired ratio；
3. natural 与竞品自身 tuned 配置并列，不能用 tuned 覆盖 natural；
4. 保存阶段时间线、GPU/CPU/RSS/object-store/network/shuffle 原始 JSONL；
5. 加 Ray Data natural/ref-only、native 和 RayOrch parent-bound；
6. 做 1KB→10MB payload sweep，隔离 full-value regroup 的移动代价；
7. 输出结构 digest、每文档 page ordinal、Markdown 自漂移与跨系统漂移；
8. 对失败、fallback、重试和退出后的 Ray/GPU 清理做独立 gate。

## 9. Runner 与资源门禁

- 真实 Daft corrected p=4/8/16/32 48-PDF 和 p=8 368-PDF 全部完成；
- 真实 tuned physical plan 包含 `RayShuffle: Hash`，不包含 global output Sort；
- Daft 隔离环境 optional tests：7 passed；
- V3.6 benchmark Ray-free suite：51 passed，主环境 Daft optional test 1 skipped；
- `py_compile` 通过，Python 源文件无超过 88 列的行；
- 所有 runner 均执行 `ray.shutdown()`；收尾 `ray status` 无实例、GPU compute process 为空；
- 已明确通知共享 GPU 的并行任务恢复使用资源。

## 10. 临时 artifact

当前 feasibility artifact 位于：

```text
/tmp/rayorch_vldb_daft_gpu4/
/tmp/rayorch_vldb_v36_gpu4/
/tmp/rayorch_vldb_daft_gpu48/
/tmp/rayorch_vldb_v36_gpu48/
/tmp/rayorch_vldb_daft_gpu368/
/tmp/rayorch_vldb_v36_gpu368/
/tmp/rayorch_vldb_daft_gpu48_fixed_p4/
/tmp/rayorch_vldb_daft_gpu48_fixed_p8/
/tmp/rayorch_vldb_daft_gpu48_fixed_p16/
/tmp/rayorch_vldb_daft_gpu48_fixed_p32/
/tmp/rayorch_vldb_daft_gpu368_fixed_p8/
/tmp/rayorch_daft_regroup_control_cpu368_full/
/tmp/rayorch_daft_regroup_control_cpu368_ref/
/tmp/rayorch_daft_regroup_control_cpu368_full_r2/
/tmp/rayorch_daft_regroup_control_cpu368_ref_r2/
/tmp/rayorch_daft_regroup_control_cpu368_full_r3/
/tmp/rayorch_daft_regroup_control_cpu368_ref_r3/
/tmp/rayorch_daft_regroup_control_gpu368_full/
/tmp/rayorch_daft_regroup_control_gpu368_ref/
```

`/tmp` 不属于论文可复现实验仓库；正式 matrix 必须复制到持久、带 run ID 的 artifact root。
