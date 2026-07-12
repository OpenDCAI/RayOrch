# TODO：MicroBatch 分发与收集

## 目标

使 RayModule 的分发和收集能够理解 MicroBatch，而无需对用户代码进行特殊处理。

## 分发

对于 `Dispatch.SHARD_CONTIGUOUS` 风格的执行，切分必须保留所有与行对齐的状态：

```text
columns
row_ids
path_ids
```

分发函数应调用：

```python
microbatch.slice(start, end)
```

而非仅切分载荷列。

## 收集

副本输出应按副本顺序合并：

```text
healthy MicroBatch:
  concat columns
  concat row_ids
  concat path_ids

quarantined:
  concat records

lineage delta:
  merge append-only updates
```

## 验收测试

使用 16 行、4 个副本，以及一个分片中的一个错误行：

- 所有健康行均恰好返回一次；
- 错误行恰好被隔离一次；
- 输出顺序具有确定性；
- 行 ID 始终与输出行对齐。
