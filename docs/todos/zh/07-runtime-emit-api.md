# TODO：用于记录基数变更的批量 Emit API

状态：未来设计方向。不要阻塞当前保留文档记录的 Runtime MVP。

默认基数和混合粒度端口 API 现已在
[`09-multi-grain-port-cardinality-api.md`](09-multi-grain-port-cardinality-api.md) 中进一步细化。
本文档仍聚焦于高级命令式 `ctx.emit()` 路径
及其共享的缓冲执行协议。

## 目标

在保留微批执行以及每次 actor 调用一个 Ray 结果的同时，支持 filter、flat-map、去重、合并、聚类和通用 `N:M` 父关系。

设计原则是：

```text
batch execution
record-level semantics
local buffered emit
single-result Ray RPC
```

## 执行模型

```text
actor receives MicroBatch
  -> create RuntimeEmitContext and local EmitBuffer
  -> user op processes the microbatch
  -> ctx.emit()/ctx.drop() append local metadata
  -> ctx.finalize() builds one ExecResult
  -> actor returns once
  -> scheduler advances downstream nodes
```

`ctx.emit()` 不得执行 RPC、对象存储写入或即时下游分发。

## 高级用户 API

```python
@orch.emit_op(outputs=("doc",), record_local=False)
class DedupOp:
    def __call__(self, docs):
        ctx = orch.ctx()
        seen = set()

        for src, doc in ctx.inputs(docs):
            key = normalize(doc)
            if key in seen:
                ctx.drop(parent=src, reason="duplicate")
            else:
                seen.add(key)
                ctx.emit(parent=src, values={"doc": doc})
```

`src` 值是一个不透明的、仅在本次调用内有效的输入引用。用户不应直接构造记录 ID 或谱系节点。

## 缓冲辅助方法

单记录 API 保持易读：

```python
ctx.emit(parent=src, values={"doc": doc})
ctx.drop(parent=src, reason="duplicate")
```

批处理辅助方法应避免常见模式中的 Python 级 append 开销：

```python
ctx.emit_batch(values=..., parents=...)
ctx.emit_filter(values=..., parents=..., mask=...)
ctx.emit_flatmap(values=..., parents=..., offsets=...)
```

所有辅助方法都追加至同一个 actor 本地缓冲区，并只 finalize 一次。

## Runtime 结果方向

最终结果应包含：

```text
healthy output MicroBatch
multi-parent lineage delta
normal drop records
quarantine records
```

正常过滤与执行失败必须保持区分：

```text
emit       -> produced record
drop       -> intentionally filtered record
exception  -> quarantined record
```

## 调度边界

Emit 描述一次 actor 调用内的输出记录及其父关系。
它不决定哪些记录会到达该调用。

数据集范围的去重、group-by、聚类或 join 可能还需要：

```text
partitioning
shuffle
stateful actors
barriers or windows
```

这些调度原语应与 `EmitBuffer` 分开设计。

## 当前所需准备

当前 `1:1` Runtime 应避免会阻碍 emit 的假设：

- 谱系节点可以有多个父节点；
- 谱系存储应仅追加且可合并；
- 用户可见路径仍保持为不透明 ID；
- Runtime 结果元数据应能容纳未来的 drop 和 emission 增量；
- 扇入应按身份对齐记录，而不是要求路径完全相同；
- 嵌套的页面/块值不得与独立的 Runtime 行混淆。

不要向用户暴露 `record_id` 或 `alignment_id`。静态基数装饰器、具备运行时感知能力的返回辅助方法以及逐端口批次已在多粒度端口设计中规定。Shuffle API 仍推迟至真实的数据集全局工作负载定义其语义之后。

## 初始验收工作负载

实现后，使用批次本地去重 operator 验证：

- 一个微批进入一次 actor 调用；
- 重复记录被报告为正常 drop；
- 唯一记录保留精确的父谱系；
- 不发生逐记录 Ray RPC；
- actor 返回一个最终结果。
