# Ray Data reference-only MinerU baseline

## 1. 目的

原始 Ray Data baseline 的自然写法：

```text
flat_map(page image rows)
→ map_batches(OCR)
→ groupby(parent_id) 搬运完整 page/OCR rows
→ map_groups(assemble)
```

在 368-PDF full run 中产生约 79.9GB shuffle。为避免 strawman，增加 reference-only
baseline：

```text
OCR actor
→ coarse payload block
→ Dataset 只保留 parent/ordinal/store/block/row manifest
→ groupby 小 manifest
→ assemble actor dereference blocks
```

Ray Data 仍负责 flat_map、GPU map_batches、groupby 和 map_groups。

## 2. 与 V3 的边界

该 baseline 不是 V3：

- 没有 GrainId/ItemRef/PortId；
- 没有 lineage compiler；
- 没有 suppression/failure attribution；
- 没有 generation fencing/retry；
- 应用手工定义 parent/ordinal；
- 应用手工创建 coarse blocks；
- 应用手工去重 block refs、排序和 row selection；
- 应用手工实现 owner actor、consumer credit 和 release；
- Ray Data 不原生支持 ObjectRef column，直接放入 Dataset 会退化为 pickled Python
  object extension。

它是一个合理的专家级 Ray Data baseline：使用 Ray primitives 避免大 value shuffle，但不
重新实现完整 V3 语义层。

## 3. 生命周期发现

最初由 transient OCR map worker 直接 `ray.put`。4/48 PDF 可运行，但 368 PDF 中 OCR
actor pool 在下游消费前退出，导致：

```text
OwnerDiedError
```

修正后由 Driver 创建固定 payload-store actors：

```text
store(block, parent-consumer-count) → integer block token
acquire(token) → nested ObjectRef
release(token) → consumer credit -1
```

所有 consumer 完成后 store 删除 block ref。该逻辑正说明 reference-only 优化并非只把一列
换成 ObjectRef；应用必须承担 ownership protocol。

## 4. 48-PDF 结果

```text
                          full-value     reference-only
wall                      165.419 s      154.953 s
pages/s                     5.997          6.402
shuffle payload            11.14 GB        ~0.37 MB
correctness                 all 48 >=0.98
```

reference-only shuffle 本身只有毫秒级 CPU 工作。Ray Data stats 中 `Shuffle executed in
143s` 是 streaming plan 的累计生命周期，不等于 shuffle kernel 消耗 143s；suboperators
的真实工作仅约 0.1s。

Store 水位：

```text
4 store actors
peak live blocks/store      5–9
peak live rows/store      164–352
final live blocks            0
final live rows              0
```

reference-only 比 full-value 更公平，但仍慢于已有 V3 48-PDF E2E `116.928s`。当前差异不能
全部归因到 lineage；Ray Data actor startup、block materialization、Dataset scheduling 和手工
store RPC 都在内。

## 5. 368-PDF full

Transient OCR worker 直接拥有 ObjectRef 的第一版在 full scale 失败：

```text
OwnerDiedError
```

加入 Driver-owned store actor 和 consumer credits 后成功：

```text
                          full-value     reference-only       V3 elastic
wall                      982.549 s      889.164 s            626.890 s
pages/s                     7.198          7.954               12.032
shuffle bytes              79.9 GB          1.13 MB             N/A
documents                   368             368                 368
```

Correctness：

```text
V3/reference Jaccard median    0.99548
matched documents             368/368
```

Store 水位：

```text
peak live blocks/store       40–57
peak live rows/store       1,216–2,192
final live blocks              0
final live rows                0
```

reference-only 将 generic value shuffle 几乎完全消除，并比 natural Ray Data 快约 10.5%；
仍比 V3 elastic 慢约 41.8%。这部分差异来自 Ray Data execution/materialization、额外
store RPC，以及用户层 ownership protocol，不应全部归因于 lineage metadata。

## 6. 论文建议

正式表同时展示：

```text
Ray Data natural/value-shuffle
Ray Data expert/reference-only
V3
```

配套代码量/概念对比：

| 能力 | Ray Data reference-only | V3 |
| --- | --- | --- |
| parent/ordinal | 用户字段 | compiler/identity |
| coarse block | 手工 ray.put | Stage transport |
| row selector | 用户字段 | ValueTake/RowTake |
| ownership | 用户 owner actor | runtime |
| release | 用户 consumer credits | runtime |
| failure lineage | 未提供 | Grain/Item receipts |
| retry fencing | 未提供 | generation token |

这样结论不是“Ray Data 做不到”，而是：

> Ray Data 可以通过专家级手工 reference protocol 避免 value shuffle；V3 将该模式变成
> General-DAG、lineage-aware、failure-aware 的清晰 API 和 runtime contract。
