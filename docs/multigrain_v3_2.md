# Multigrain V3.2：Port-first logical IR

> 状态：历史 logical-IR 原型；其 correctness-first lowering 会创建结构 actor，推荐的无结构 actor 架构已演进至 v3.3。

V3.2 不再把 `Map/Expand/Reduce` 作为用户 UDF wrapper。用户只声明 `RayModule`，grain
变化由 functional Port operator 表达：

```python
page_groups, metadata = self.render(pdfs)
pages = F.expand(page_groups)
contents = self.ocr(pages)
content_groups = F.reduce(contents)
result = self.assemble(metadata, content_groups, page_groups)
```

## 三个正交概念

```text
Node
    计算或结构变换：Source / Call / Expand / Reduce

Port
    一个逻辑 value edge；由 LogicalPortId(node, output) 标识

Domain
    Port 上每个 occurrence 的 Entity 对齐域
```

`PortSpec` 独立保存 `domain` 和 `layout`。因此同一个 multi-output Call 的一个输出可以
展开到 child domain，其他输出仍留在 parent domain。`F.expand()` 总是创建新 Port，绝不
修改 producer 或复用输入 PortId。

## Value layout

```text
DirectValue
    Node 直接产生的值

GroupValue
    一个 parent-domain Port；由 leaf Port 和 expansion path 定义 ordered group
```

Group layout 属于 producing Port，而不是某个 consumer 的临时 InputMode。这让 grouped
Port 可以有多个消费者，也可以作为 Pipeline output。第一版 physical lowering 会显式
物化 GroupValue。

## Expansion 与 alignment

每次独立 `F.expand(port)` 创建新的 `ExpansionId` 和 child `DomainId`。两个独立 expand
即使运行时长度相同，也不会隐式 zip。

需要共享 child Entity 时必须显式声明：

```python
left, right = F.expand_aligned(left_groups, right_groups)
```

对应一个 `ExpansionSpec`，运行时继续校验每个 parent occurrence 的所有输出 cardinality
相同。`F.reduce_aligned()` 使用同一 expansion 和 canonical shape 一次恢复多个 GROUP。

## 当前 physical lowering

第一版采取 correctness-first 策略，每个 logical transform 都降低成现有已验证的 V3
physical Stage：

```text
Call        -> MAP
ExpandNode  -> structural EXPAND actor
ReduceNode  -> structural REDUCE actor
```

这会比融合计划多轻量 actor/RPC，但有两个好处：

- API correctness 不依赖 optimizer；
- logical IR、physical IR 和 runtime failure 可以分层测试。

未来 fusion 只能作为等价优化：单输出且原 group Port 无其他物理消费者时，可以把
`Call + ExpandNode` 融合成一个 EXPAND；单 consumer GroupValue 可以把
`ReduceNode + Call` 融合成一个 REDUCE。部分 multi-output expand 不允许 whole-stage
backpatch。

## 与 V3 的关系

V3.2 的 logical 数据结构是新的，只参考 V3 identity/scope 经验。当前执行后端暂时复用
V3 的 `CompiledDAG/Arena/StageExecutor/Worker`，用于持续运行已有 correctness 和 Ray
回归；这不是 logical IR 对旧数据结构的依赖。未来替换 physical backend 时，用户 API
和 logical graph 不需要变化。

## 当前边界

- `reduce()` 默认关闭最近一层 expansion；
- parent value 下沉到 child domain 仍需未来的显式 `broadcast/carry`；
- 同一 RayModule 多 call-site 暂不承诺共享 actor pool；
- lowering 暂未实现 fusion cost model 和 logical-to-physical explain mapping；
- recovery、filter 和 nested Reduce 仍由 V3 backend 提供，但需补 V3.2 API 回归矩阵。
