# Multigrain V3.1：RayModule + functional authoring

> 状态：历史 API 原型，用于记录 functional authoring 到旧 V3 StageSpec 的 lowering；推荐架构已演进至 v3.3。

V3.1 是 V3 的编译前端实验，不是第二套运行时。目标是验证新的 functional 用户 API
能否无损降低到现有 V3 physical DAG。

## 稳定边界

V3.1 只能新增 trace-only authoring state。编译完成后必须继续使用 V3 的：

- `CompiledPipeline` / `CompiledDAG` / `StageSpec`；
- `PortId` / `EntityId` / `ItemRef` / `GrainId` / `GroupShape`；
- Arena、Driver、StageExecutor、Worker 和 RPC DTO；
- 每个 physical Stage 一个 persistent actor pool 的通信模式。

因此 `expand()` 和 `reduce()` 默认不是 physical Stage，也不拥有 actor：

```text
RayModule(render) + F.expand
    -> Primitive.EXPAND(render)

RayModule(assemble) consuming grouped views
    -> Primitive.REDUCE(assemble)
```

## Trace-only 数据结构

V3.1 的 `Port._view` 只有 `AUTO`、`SCALAR`、`GROUP` 三种状态，用于区分同一个
physical child Port 的 scalar view 与 lazy grouped view。`Pipeline.compile()` 返回前会
丢弃该字段，并重新构造 V3 原生 `Port`，所以 view metadata 不进入 runtime。

`F.expand(group)` 会把尚未被消费的默认 MAP producer backpatch 为 EXPAND；原变量成为
grouped view，返回值成为 child scalar view。`F.reduce(port)` 只创建 grouped view，直到
下游 RayModule 消费它时才生成 REDUCE Stage。

## 当前刻意限制

- `expand()` 必须发生在 producer 的其他 consumer 之前；
- functional `expand()` 暂时只支持单输出 RayModule；
- `reduce()` 默认只关闭最近一层 Expand；
- lazy group 不能直接作为 Pipeline output；
- 当前未实现真正物化的 `Port[list[T]]`；未来应通过显式 `materialize()` 插入 physical
  Stage，而不能让 Driver 读取业务 payload；
- 同一 RayModule 多 call-site 是否共享 actor pool 尚未定义，第一版不承诺共享。

## 可维护性规则

如果一个 V3.1 功能要求修改 Arena、Worker、protocol 或 identity model，应先证明它是
通用 V3 runtime 能力，而不是把 authoring sugar 泄漏到运行时。任何 lowering 都必须有
测试断言 physical Stage kinds，防止无意插入 adapter actor。
