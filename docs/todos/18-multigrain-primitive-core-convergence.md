# Multigrain Primitive Core 收敛记录

## 结论

新 `MultigrainIR` 现在是唯一演进主线。用户 API 仍是平坦的
`Map / Filter / Select / Expand / Reduce / Relate`；旧
`RuntimeRayModule / RuntimeDagExecutor` 不作为兼容目标，也没有 facade。
旧 Runtime 中值得保留的工厂式实例化、每 stage 常驻 actor、inflight 控制等设计，
由新执行后端按需吸收。

本轮只收敛 primitive 的声明、输出构造、执行分派和能力边界，不重构 Ray actor
生命周期，也不把 handler 或 builder 放入被动 IR。

## 重复/差异矩阵

| 流程 | 重构前重复 | 真正共享的不变量 | 必须保留的差异 | 收敛点 |
|---|---:|---|---|---|
| constructor/recipe/lazy op | 6 个 wrapper | 类引用、参数、默认属性、每 executor 懒实例化 | Expand 默认物理提示、各原语 provenance | `primitives._binding.PrimitiveBinding` |
| output arity/shape | Map/Expand/Reduce/Select 各自检查 | 声明输出数必须等于实际输出数，平行列等长 | Expand 还要求逐父 group shape 相同 | `primitives._utils.checked_output_lists` + 原语专属校验 |
| `PortBatch` 输出构造 | Map/Filter/Expand/Reduce/Relate 手写 | values/id/display/ancestor/ordinal/lineage/relations 必须平行 | preserve、parent-expand、anchor-reduce、role-relation 的边语义不同 | `primitives.output.PortBatchBuilder` |
| diamond lineage | Map 内部手写 | 同 identity 输入的祖先必须一致、路径做稳定 union | 不推断跨 identity join | `primitives.output.merge_aligned_inputs` |
| compiled dispatch | executor 中 NodeKind 条件链 | IR recipe 只在 executor/actor 内还原一次 | Reduce 需要 group、Map 有 deferred recovery、Relate 有 evidence | `execution.handlers.PrimitiveHandlerRegistry` |
| Ray shardability | `_SHARDABLE` NodeKind 白名单 | 是否可按行分片由 relation contract 决定 | Reduce completion、Relate cross-row evidence 不可拆 | `ir.capabilities.capabilities_for` |
| IR verification | NodeKind 分支扩张 | refs、grains、relations、parents 与 outputs 必须结构一致 | 各 relation family 有自己的约束 | `VerifyPass` structural + family validation |

抽取后没有保留旧 wrapper 构造分支、executor wrapper 重建条件链或 `_SHARDABLE`
双路径。新增模块均为内部模块，不扩大公共 API。

## 每个抽象的责任边界

### `PrimitiveBinding`

- 解决：六类 wrapper 重复保存 recipe/default/lazy factory。
- 拥有：一个 operator class recipe 在一个 executor replica 内只实例化一次；
  compiled graph 只能接收可导入 class。
- 不负责：调用 UDF、cardinality、lineage、恢复和调度。

### `PortBatchBuilder`

- 解决：primitive 手写多组平行 metadata 列，容易遗漏 relation/error/lineage。
- 拥有：primitive 输出的平行列完整性和统一命名；明确执行
  aligned-preserve、filter、parent-expand、role-relation 构造。
- 不负责：猜测 parent、anchor、role 或 identity；这些证据必须由原语语义提供。

### `PrimitiveHandlerRegistry`

- 解决：compiled executor 以 NodeKind 条件链重建 wrapper，并用类名后缀特判 Select。
- 拥有：IR node 到 prepare/execute adapter 的唯一映射；compiled 路径复用 eager
  wrapper 和同一个 output builder。
- 不负责：actor pool、shard planner、backpressure、stage lifecycle 或恢复预算。

### `PrimitiveCapabilities`

- 解决：Ray executor 和 verifier 各自维护 NodeKind 白名单。
- 拥有：从现有 `RelationSpec` 派生 identity alignment、group completion、
  row partitionability 和 relation evidence family。
- 不负责：预埋 Window/Shuffle 等尚无消费者的能力，也不复制进序列化 IR。

## 本轮封闭的语义

1. `Map / Expand / Reduce / Select` 严格校验声明输出数；`num_outputs` 必须是正整数。
2. `Select` eager 和 `Map -> SelectFilter -> Project` lowered IR 共用同一个
   mask 类型、长度、输出和 lineage 语义。
3. compiled primitive 只接受可导入 operator class；live instance 明确为 eager-only。
4. `Relate` 当前只允许一个输出；compiled `on=` 只允许字段名，live callable 和
   `relation_fn` 不进入 IR。
5. adapter relation id 由父 evidence 和可选 `stable_key` 决定，不再依赖 emission
   序号；adapter 与 key-join 都继承所有父端口的 ancestry、ordinal、lineage 和 error。
6. diamond aligned merge 遇到同一 ancestor grain 对应不同 record id 时直接报错，
   不再以最后一个输入静默覆盖。
7. `Reduce(missing_child=...)` 同步写入 `RelationSpec.missing`，contract 与 recipe
   不再表达不一致。

## 验证覆盖

- semantic negative：live instance compile、callable compiled join key、Relate
  multi-output、Map/Expand/Reduce 非法 output arity。
- eager/compiled：Select 的值、identity 和 lineage 完全一致。
- builder/lineage：diamond metadata 冲突拒绝、M:N 完整笛卡尔积、adapter 祖先继承。
- registry/capability：普通和 internal recipe 分派、Map/Expand/Reduce capability。
- structural verifier：contract grain、relation output/parents、anchor/role/parent_input。
- 原有随机重排、Ray、recovery、MinerU graph 与 LPT suite 继续作为回归门禁。

## 后续边界

下一阶段按顺序推进 `RecoveryKernel + RecoveryAdapter`、`ErrorLedger/BatchArena`、
统一 `RayStageBackend/StagePool`，最后删除旧 Runtime 双栈。Join/Shuffle/Window
只有在有真实 executor 和 verifier 消费者时才进入 capability/IR。
