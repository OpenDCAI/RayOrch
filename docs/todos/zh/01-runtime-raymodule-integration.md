# TODO：Runtime 与 RayModule 集成

状态：核心适配器已实现；生命周期方向已记录在
[`../../runtime_module_lifecycle.md`](../../runtime_module_lifecycle.md) 中。

## 目标

在不将 RayModule 变为谱系负担沉重的抽象的前提下，使运行时语义可通过 RayModule 生态系统使用。

## 当前方向

将 `RayModule` 保持为轻量级的急切产品：

```text
RayModule
  actor replicas
  dispatch
  collect
  max_inflight
```

让 `RuntimeRayModule` 复用选定的底层机制，而不继承
急切生命周期：

```text
RuntimeRayModule
  wraps user op
  owns runtime spec
  shares low-level actor machinery
  defers physical startup to Runtime Executor
```

## 关键设计

每个 Ray actor 副本应持有：

```text
self.op       # user op instance
self.runtime  # rowwise runtime instance
```

Actor 运行流程：

```text
receive MicroBatch shard
runtime.run_rowwise(op.run, shard)
return RuntimeResult
```

## 验收测试

一个由 RayModule 包装的运行时 op 应当：

- 通过 Ray 接收一个 MicroBatch；
- 在 actor 内隔离一个错误行；
- 返回健康的 MicroBatch 输出；
- 返回隔离记录；
- 为健康行保留行 ID；
- 为健康行推进路径 ID。
