# Video：Kinetics-400 50GB V3↔V3.5 回归

日期：2026-08-06～07；状态：CPU OpenCV full gate 与 GPU representative gates 完成。

## 数据合同

数据来自 CVDF 官方 Kinetics-400 S3 snapshot：validation 全 20 shards，加 train
shards `0..12`。33 个 archive 共 `50,607,623,297` bytes；解压后为 32,881 个 MP4、
`50,941,325,414` bytes。

metadata gate 拒绝 91 个损坏容器，冻结 manifest 为：

```text
videos                    32,790
bytes                     50,561,755,359
duration                  313,674.449s（约 87.13 小时）
train / validation        12,913 / 19,877
source identity           split/shard/filename，全部唯一
```

完整内容审计读取每个 byte 并逐帧 decode：

```text
decoded frames            8,613,414
decode errors             0
short decodes             0
byte mismatches           0
content duplicate groups  0
ordered content digest    10f74cef2e7a4d359100d1953a5812d0
```

本地研究只使用原始数据许可范围，不主张第三方 clip 的再分发权。

## Paired 合同

V3 和 V3.5 使用同一有序 manifest、同一 V3 业务 UDF 和以下配置：

```text
stride                    4
max frames/video          32
decode/reduce actors      4 / 4
transform actors          4 CPU
transform batch cap       16
Arena/source microbatch   32 videos
max active Arenas         4
backend                   deterministic OpenCV
```

每个 arm 的 wall 包含 actor startup、业务执行、最终物化和 Executor teardown；外层
`ray.init()/ray.shutdown()` 不计入。OpenCV 输出要求 frame count、source index、digest
和 edge density 全字段 exact。

## Scale 结果

| Videos | Input bytes | V3 | V3.5 | V3.5 相对 V3 | Correctness |
| ---: | ---: | ---: | ---: | ---: | --- |
| 48 | 80,681,298 | 6.879s | 6.957s | -1.11% | exact |
| 256 | 419,393,722 | 26.962s | 27.426s | -1.69% | exact |
| 4,096 | 6,461,514,035 | 409.873s | 410.395s | -0.13% | exact |
| 32,790 | 50,561,755,359 | 3268.429s | 3223.813s | **+1.38%** | exact |

full gate 处理 1,034,374 sampled frames，两臂 output digest 都是
`bace0b2b38a5819a3e1b75fbdc1862fb`。

## Full 物理证据

```text
V3 RPC / grains per RPC       129,244 / 8.511
V3.5 RPC / grains per RPC     129,875 / 8.469
V3.5 decode Call              32,790 RPC，batch=1
V3.5 transform Call           65,006 RPC，avg batch=15.912
V3.5 transform full batches   64,262
V3.5 reduce Call              32,079 RPC，avg batch=1.022
V3.5 max active Arenas        4
V3.5 retries                  0
worker observation errors     0
released value bindings       2,199,908
```

V3.5 总 RPC 比 V3 多约 `0.49%`，但 full wall 快 `1.38%`；这支持“编译器化和显式
状态机没有引入规模性能回退”，不能单凭 CPU OpenCV 推导 GPU 模型普遍加速。

## GPU 模型门禁

模型门禁同样复用 V3 的业务 UDF，并把 actor startup、materialize 和 Executor teardown
计入每臂 wall。每组正式结果包含 `V3-first` 与 `V3.5-first` 两个顺序。

### SmolVLM caption

最初 256-video Kinetics gate 的 normalized caption mismatch 为 `4.0039%`，同时
Transformers 明确警告 decoder-only generation 使用了 right padding。物理 batch 形状不同
因此被错误 padding 放大为业务输出差异。共享 `FrameCaptioner` 改为 left padding 后警告
消失，结果为：

| Dataset / videos | Frames | V3 median | V3.5 median | V3.5 wall | Structure | Normalized mismatch |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| MSR-VTT / 48 | 183 | 26.309s | 27.061s | +2.86% | exact | 0 / 183 |
| Kinetics / 256 | 1,024 | 85.485s | 86.670s | +1.39% | exact | 1/1,024；2/1,024 |

两轮 Kinetics RPC 都是 `390/390`。left-padding 修复后的文本差异为 `0.098%` 和
`0.195%`，低于冻结 `2%` 上限；该上限只适用于归一化模型文本，video/frame 数与
source index 仍必须 exact。

### Whisper + ViT sibling relations

该图包含 audio/frame 两个 sibling child Domain，各自 Expand→Map→Reduce，最后在同一
root Video entity 上 aligned merge。256-video 两轮结果：

```text
V3 median wall             39.071s
V3.5 median wall           39.695s（+1.60%）
V3 RPC                     1,120 / 1,127
V3.5 RPC                   1,139 / 1,139
structure                  exact
frame digest               exact
formal-run transcript      exact
```

一次诊断 run 曾出现 2/256 raw transcript drift，其中 1/256 经归一化后仍不同
（`0.3906%`）；但所有 audio chunk count、frame count 和 frame digest 均 exact，随后两轮
正式 run 的 transcript 也 exact。这说明它是 Whisper 在物理 batch shape 下的数值漂移，
不是 lineage 或 root merge 错配。runner 因此分别 gate 结构、frame digest 与 normalized
transcript，并保留首个字段级差异；不会用文本阈值放宽结构错误。

MSR-VTT 样本没有音轨，最初多模态 smoke 在 V3 只显示笼统 `ArenaAbort: UDF error`。
改用已确认带音轨的 Kinetics 前缀后，2-video smoke exact。这个 badcase 同时说明框架异常
仍需补 stage/UDF/root-cause 上下文，不能把无音轨输入误判为多模态语义失败。

## 产物

```text
/tmp/mgv35-kinetics50/plan.json
/tmp/mgv35-kinetics50/manifest-full-decodable.json
/tmp/mgv35-kinetics50/manifest-full-decodable.report.json
/tmp/mgv35-kinetics50/manifest-full-decodable.audit.json
/tmp/mgv35-kinetics50/paired-48-opencv-observe-parallel.json
/tmp/mgv35-kinetics50/paired-256-opencv-historical-shape.json
/tmp/mgv35-kinetics50/paired-4096-opencv-historical-shape.json
/tmp/mgv35-kinetics50/paired-full-32790-opencv.json
/tmp/mgv35-video-model/caption-48-paired.json
/tmp/mgv35-video-model/caption-kinetics-256-leftpad-paired.json
/tmp/mgv35-video-model/multimodal-kinetics-256-paired.json
```

## 结论边界

已经证明：

- 50.56GB、32,790 个独立 source 可以稳定经过多 Arena Expand→Map→Reduce；
- V3/V3.5 业务输出 exact；
- 完整规模性能等价且 V3.5 本轮略快；
- 真实 SmolVLM 与 Whisper+ViT representative gates 的结构/血缘等价，wall 回退均小于 2%；
- materialize 后中间 value bindings 被释放，Ray 正常 shutdown。

尚未证明：

- 昂贵模型在全部 32,790 clips 上的 wall；代表性 256-video gate 已足以隔离启动噪声，
  但不能冒充 full-model-50GB 统计；
- 另一顺序的第二次 full-50GB trial。256 档已有双向交替，用于排除明显顺序偏差。
