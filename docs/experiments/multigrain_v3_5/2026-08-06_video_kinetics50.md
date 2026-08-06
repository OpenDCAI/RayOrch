# Video：Kinetics-400 50GB V3↔V3.5 回归

日期：2026-08-06；状态：CPU OpenCV full gate 完成，GPU model gate 待资源恢复。

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
```

## 结论边界

已经证明：

- 50.56GB、32,790 个独立 source 可以稳定经过多 Arena Expand→Map→Reduce；
- V3/V3.5 业务输出 exact；
- 完整规模性能等价且 V3.5 本轮略快；
- materialize 后中间 value bindings 被释放，Ray 正常 shutdown。

尚未证明：

- 四卡 ViT/SmolVLM/Whisper 的 V3↔V3.5 性能；当前节点 GPU driver 不可用；
- GPU batch-shape 浮点漂移上限在 V3.5 上的真实模型复验；
- 另一顺序的第二次 full-50GB trial。256 档已有双向交替，用于排除明显顺序偏差。
