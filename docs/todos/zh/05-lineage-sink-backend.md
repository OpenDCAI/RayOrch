# TODO：谱系 Sink 后端

## 目标

将运行时谱系生成与谱系持久化分离。

## 方向

运行时应产生增量：

```text
paths
row_path updates
quarantined records
future parent_edges
future optional mutation events
```

执行器或监督器应将这些增量提交到一个 sink：

```text
InMemoryLineageSink
RayLineageActorSink
SQLite/DuckDB sink
Parquet/object-store sink
Postgres sink
```

## 规则

不要在用户 op 执行期间从其内部针对每一行同步写入谱系。

优先采用：

```text
local buffer
batch finalize
commit delta
```

## 载荷策略

隔离载荷应变为可配置：

```text
metadata
ref
full
```

MVP 可以为调试保留完整值，但生产环境默认应使用元数据或引用。
