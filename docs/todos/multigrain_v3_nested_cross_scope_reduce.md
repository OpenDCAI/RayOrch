# Multigrain V3 跨层级 Nested Reduce TODO

> 状态：已实现；保留本文作为设计记录。
>
> 当前普通 `Reduce` 只关闭一层 Expand scope。本文记录未来显式支持
> list-of-list 跨层级 Reduce 的可能方向。

## 1. 目标场景

```text
Document
→ Expand Pages
→ Expand Regions
→ Region processing
→ Reduce(anchor=Document, members=Regions)
```

期望 Reduce UDF 收到：

```python
list[list[Region]]
```

更深层级应按 Expand ancestry 形成对应深度的 nested list。

当前写法直接表示 canonical nested Reduce：

```python
reduce(anchor=document, members=regions)
```

若 anchor/member 之间跨越 Page、Region 两个 Expand，UDF 收到
`list[list[Region]]`。

## 2. 为什么现有 lineage 有实现基础

V3 已保存：

```text
EntityOrigin
    child → parent + expand_stage + ordinal

ExpandInstance
    anchor + cardinality / failed-before-output

ItemRecord
    PRESENT / DROPPED / FAILED / SUPPRESSED
```

因此可以从 leaf 重建：

```text
root anchor
scope path
完整 ordinal path
每层 zero-output
terminal failure/drop
```

不能只依赖已有 leaf ItemRef：中间 Expand 的 `N=0` 没有 leaf，必须读取
`ExpandInstance.cardinality` 才能保留空 list。

## 3. 最终 API

没有增加 `nested(...)` 或 `NESTED_GROUP`。`ReduceSpec.scope_path` 由 compiler 根据
anchor/member scope 唯一推导：

```text
GROUP depth == len(scope_path)
```

## 4. 候选 list-of-list 规约

以下规则尚待拍板：

1. 每一层按对应 Expand ordinal 排序；
2. completion order、RPC packing 和 actor 不影响结构；
3. intermediate Expand 成功且 `N=0` 时保留 `[]`；
4. intermediate occurrence 在进入下一层前正常 drop 时，从 parent list 省略；
5. leaf drop 从最内层 list 省略；
6. required intermediate/leaf failure 默认 Suppress 整个 root Reduce；
7. failure causes 按完整 ordinal path canonicalize；
8. 多个 `GROUP` 只允许完全相同的 `scope_path`。

## 5. 最终数据结构

现有 flat：

```text
InputBinding.items: tuple[ItemRef, ...]
ValueTake: tuple[RowTake, ...]
```

不能表达不同 tree shape。候选结构：

```text
GroupShape
└── offsets_by_level: tuple[tuple[int, ...], ...]

InputBinding
├── items: flat ordered ItemRef leaves
└── group_shape: GroupShape | None

ValueTake
├── rows: flat ordered RowTake leaves
└── group_shape: GroupShape | None
```

第一版优先保证语义直观；是否改成 flat leaves + offsets/CSR，应由 metadata benchmark
决定。

## 6. 四组件边界

若实现，职责应保持：

```text
CompiledDAG
    推导并验证唯一 scope_path。

ArenaEngine
    根据 EntityOrigin、ExpandInstance 和 ItemRecord 重建 nested shape。

RunDriver
    无需理解 hierarchy。

StageExecutor / Worker
    只透传 NestedTake，并在 Worker 侧重建 Python nested list。
```

禁止 Driver `ray.get` leaf payload 后自行构造 nested list。

## 7. 性能与语义风险

- root 必须等待完整 subtree；
- intermediate blocks 可能持有更久；
- 单次 Reduce UDF 输入可能很大；
- nested selector metadata 与 tree node 数量成正比；
- failure 默认影响整个 root；
- 不提供 incremental aggregation。

因此该能力应显式 opt-in，不能替代逐层 Reduce。

## 8. 实现前必须讨论的问题

已冻结：

1. 不增加新 API keyword；
2. tree shape 使用 offsets；
3. intermediate successful N=0 保留 `[]`；
4. intermediate normal drop 省略 node；
5. required failure Suppress root Reduce；
6. 多个 GROUP 必须具有相同 scope path；
7. metadata 受 Arena hard limit；
8. 逐层 Reduce 和 direct hierarchical Reduce 均有回归测试。

## 9. 当前已有边界测试

已覆盖：

```text
五层二叉 Expand + 五层逐级 Reduce：成功且每层顺序正确；
二十层一元 Expand + 二十层 Reduce：成功；
五层 Expand + 一个 root Reduce：成功并产生五层 nested list。
```
