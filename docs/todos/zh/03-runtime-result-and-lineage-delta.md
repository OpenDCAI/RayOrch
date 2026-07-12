# TODO：RuntimeResult 与谱系增量

## 目标

在 DAG 集成开始后，用一个小型结构化结果替代临时的 `(MicroBatch, quarantined)` 返回值。

## 最小形态

```python
RuntimeResult:
    batch: MicroBatch
    quarantined: list[QuarantineRecord]
    lineage_delta: LineageDelta
```

`lineage_delta` 应仅追加且合并成本低。

## 原因

DagExecutor 需要一个可识别的对象来承载：

- 健康输出行；
- 错误行；
- 谱系更新；
- 变更事件；
- 用于 flatmap/emit 的未来父边。

## 非目标

不要将完整载荷历史放入 RuntimeResult。载荷存储应保留在 MicroBatch 或外部对象存储中，而非谱系元数据中。
