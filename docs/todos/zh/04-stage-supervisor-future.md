# TODO：可选的阶段监督器

请参阅 [`../../runtime_module_lifecycle.md`](../../runtime_module_lifecycle.md)，了解当前由执行器拥有的 Runtime 生命周期决策。

## 目标

为每个阶段的监督器 actor 保留未来路径，而不强制将其纳入 MVP。

## 当前 MVP 形态

```text
RuntimeDagExecutor(driver)
  -> RuntimeRayModule(logical stage)
      -> RunnerActor replicas started by the executor
```

只要 driver 不是瓶颈且重试逻辑简单，这已足够。

## 未来形态

```text
RuntimeDagExecutor(driver)
  -> StageSupervisor actor
      -> RunnerActor replicas
```

## 监督器职责

监督器应处理阶段级别的关注点：

- 本地阶段队列；
- 副本健康状态；
- actor/task 重试；
- 自适应批量大小；
- OOM 回退；
- 指标；
- 背压。

它不应处理记录级的拆分与重试。该职责属于每个 worker Runtime 的内部。

## 触发条件

仅当运行时需要 actor 健康状态跟踪、自适应批处理，或 DagExecutor 无法妥善管理的阶段本地调度时才实现。
