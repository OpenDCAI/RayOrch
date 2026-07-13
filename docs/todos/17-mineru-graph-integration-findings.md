# MinerU 图集成测试套件：覆盖范围与发现

状态：已实现，并于 2026-07-12 全部测试通过。

这套测试有意放在“单元测试”和“完整 VLM 基准测试”之间：它使用 MinerU
历史输出中的真实 `layout.json` 记录和真实解码后的 `PIL.Image` 对象，但以确定性的
轻量 UDF 和 CPU sleep 代替模型推理。目标是在不承担 vLLM 成本与非确定性的前提下，
充分检验图语义、血缘、容灾和物理重排。

## 复现方法

默认数据集位于 `Flash-mineru/outputs_regression_baseline_verify`，可通过
`RAYORCH_MINERU_ARTIFACT_ROOT` 覆盖。如果未安装 Pillow 或数据集不存在，测试会正常
跳过，不会导致测试收集失败。

```bash
conda run -n torch-base python -m pytest -q \
  test/experimental/multigrain/test_mineru_*_integration.py --runslow
```

当前受限 fixture 最多选择 3 篇文档，每篇最多 3 个含真实图片的页面，共生成 34 条由
真实图片承载的 block/evidence 记录。图片进入 Ray object store 前会缩放到受控尺寸，
但调度器使用的工作量估计仍保留原始像素数。

## 已落地的覆盖范围

图形形态：

- 两级嵌套的 `doc -> page -> block` Expand；
- 零子项页面；
- 共享关系的多输出 Expand `(block, metadata)`；
- 同粒度 diamond 分叉与汇合；
- 两侧都存在重复 key 的真实 M:N `Relate(on=page_key)`；
- Reduce 回 document 锚点；
- 独立 catalog 根与 document 派生根的多根关联；
- 本地 executor 与真实 Ray 执行的结果及血缘一致性。

容灾机制：

- 可归因记录的 inline 重试，成功后正常继续；
- 永久失败记录隔离，以及随后 fail-closed 的文档级输出抑制；
- 通过有预算的自适应拆分定位 opaque shard 故障；
- 使用 `IsolationBudget` 限制密集 opaque 故障的递归隔离成本；
- 跨两级 Expand 的 deferred stage-epoch 重试，Reduce 在 drain 完成前保持阻塞；
- 替换死亡 actor，同时保留其余健康副本。

效率机制：

- 使用真实 artifact 工作量估计，对比 contiguous 与 LPT 的理论负载；
- 两级 Expand 后使用 4 个 Ray 副本实际执行；
- 验证 LPT 重排后的结果与按 key 比较的血缘保持一致；
- 验证 Reduce 能完整恢复多级层次顺序。

## CPU 代理负载的实测证据

服务时间代理与实测 artifact 大小呈非线性关系
（`power_image_work = image_work ** 3`），用于模拟视觉/OCR 任务中常见的、由像素数或
token 数导致的长尾。这里的数字不能被解释为 GPU 或完整 VLM 的性能结论。

在当前受限数据集和 4 个副本下：

- contiguous 理论分片负载：`[13.93, 17.18, 8.76, 8.00]`；
- LPT 理论分片负载：`[11.80, 12.15, 11.78, 12.14]`；
- 实测 stage makespan：`0.5165s -> 0.3654s`，降低 29.3%；
- 实测 idle bubble：`0.3029 -> 0.0148`；
- 完整预热后的端到端时间：`0.5371s -> 0.3894s`，降低 27.5%。

慢测试断言使用的是保守比例，而不是绑定上述精确数值。

本版本的验证结果：

- MinerU 专项集成测试：`13 passed`；
- 完整快速 multigrain 测试：`154 passed, 32 skipped`；
- 带 `--runslow` 的完整 multigrain 测试：`186 passed`。

## 测试发现并修复的语义缺陷

### F1：`Relate(on=)` 原先并不是真正的 M:N

当第一个 role 包含重复 join key 时，原实现只用该 key 的第一条记录作为笛卡尔积起点。
因此 2x2 关系只会输出 2 条，而不是 4 条。现在 `Relate` 会用第一个 role 中所有匹配
记录初始化组合；record ID 仍然由内容寻址生成，并保持唯一。

快速回归测试：
`test_relate_key_join.py::test_key_join_is_truly_many_to_many_when_both_roles_repeat_a_key`。

### F2：diamond Map 汇合会丢失一条分支的血缘

多输入 Map 会按身份对齐值，但原先只使用第一个输入端口构造输出元数据。因此值虽然
正确，血缘中却缺少其他分支。现在 Map 会先合并所有已对齐输入的 lineage path、
ancestors、展示血缘、ordinals、relations 和 errors，再追加当前 Map 自己的步骤。

快速回归测试：
`test_executor.py::test_same_grain_diamond_map_merges_lineage_from_every_branch`。

### F3：嵌套 Reduce 原先只恢复第一层 ordinal

LPT 重排 `doc -> page -> block` 后，原 Reduce 只按文档下的 page ordinal 排序。
同一页面内的 block 仍会保留物理 shard 的完成顺序，最终组装结果可能与串行执行不同。
现在 Reduce 会按完整 ordinal path 排序，并以内容寻址的 record identity 作为确定性的
最终 tie-breaker。

快速回归测试：
`test_reordering_invariance.py::test_nested_expand_reduce_restores_full_ordinal_path`，
覆盖 25 组随机合法分片计划。

## 尚存缺口与风险

1. 可归因恢复目前仍只实现于 Map。本套测试验证的是“嵌套 Expand 之后、下游 Reduce
   之前”的 Map 恢复，而不是 Expand/Reduce/Relate 自身的记录级重试。这些原语专属的
   可归因单元仍属于 M3 工作。
2. 测试使用真实 payload 和真实分布，但 UDF 是确定性的 CPU 代理。完整 VLM/GPU 性能
   及输出等价性仍应由现有 368-PDF benchmark 负责。
3. 完整 M:N 笛卡尔展开目前没有基数保护或 spill 策略；超大 hot key 可能造成关系爆炸。
4. 合并 ancestor 字典时，当前假设各对齐分支对同名祖先 key 的取值一致。在接受独立
   重建的同粒度端口前，应增加 verifier/runtime 冲突检查，替代当前隐含的后写覆盖。
5. batch 内 source ID 会有意在不同 microbatch 间重复。当前 coordinator 通过上下文
   隔离和唯一 deferred token 保证正确性，但持久 checkpoint identity 仍依赖文档 16
   中的 `BatchArena` 方案。
6. stage-global deferred drain 当前驻留在 driver 内存中。持久化 quarantine 和
   checkpoint/resume 仍属于 Phase 3。

## 相关文件

- fixture 与算子：`test/experimental/multigrain/mineru_integration_ops.py`；
- 图形形态矩阵：`test/experimental/multigrain/test_mineru_graph_integration.py`；
- 容灾矩阵：`test/experimental/multigrain/test_mineru_recovery_integration.py`；
- 效率矩阵：`test/experimental/multigrain/test_mineru_efficiency_integration.py`。
