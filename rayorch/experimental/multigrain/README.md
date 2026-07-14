# RayOrch Multigrain

`rayorch.experimental.multigrain` 是面向对象数据流水线的实验性执行框架。它让用户声明
`document → page → block` 等 1:1、0:1、1:N、N:1 和 M:N 变化，同时由框架维护
identity、lineage、重排、并行调度与故障恢复。

## 阅读路线

1. [Overview：从用户程序到运行时](docs/overview.md)
2. [Data 与 IR：运行时记录和关系感知 ExecutionGraph](docs/data-and-ir.md)
3. [Primitives：大原语与 Select lowering](docs/primitives.md)
4. [Execution：Local、Ray、验证与 Recovery](docs/execution.md)

应用开发者请从
[`docs/multigrain_user_guide.md`](../../../docs/multigrain_user_guide.md) 开始。

## 30 秒示例

```python
from rayorch.experimental import multigrain as mg


class PdfToPages:
    def run(self, pdfs):
        return [render_pages(pdf) for pdf in pdfs]


class OcrPage:
    def __init__(self, model):
        self.model = load_model(model)

    def run(self, pages):
        return self.model.batch_ocr(pages)


class AssembleDocument:
    def run(self, pdfs, page_text_groups):
        return [
            assemble(pdf, texts)
            for pdf, texts in zip(pdfs, page_text_groups)
        ]


class MinerUPipeline(mg.Pipeline):
    def __init__(self):
        super().__init__()
        pool = mg.WorkerPoolSpec(replicas=4, gpus_per_worker=1.0)
        self.to_pages = mg.Expand(PdfToPages, parent=0, child_label="page")
        self.ocr = mg.Map(OcrPage, model="model/path", workers=pool)
        self.assemble = mg.Reduce(
            AssembleDocument,
            missing_child=mg.IncompleteGroupPolicy.FAIL_CLOSED,
        )

    def forward(self, documents):
        pages = self.to_pages(documents)
        texts = self.ocr(pages)
        return self.assemble(mg.group_by(documents, texts))


graph = MinerUPipeline().compile()
inputs = {"documents": mg.source(["a.pdf", "b.pdf"], name="documents")}
result = mg.MultigrainExecutor().execute(graph, inputs)
```

- `Expand` 创建 document 的 page children；
- `Map` 复用 page identity；
- `Reduce` 按 ancestry regroup 并回到 document identity；
- UDF 从不接收 `record_id`、ancestor、ordinal 或 lineage；
- `compile()` 只保存可导入 class 和构造参数，不实例化模型；
- compile、Local 和 Ray 执行前都会调用 `verify_graph()`。

## 当前 IR 主线

编译结果是不可变的 `ExecutionGraph`：

```text
GraphInputRef / NodeOutputRef
        │
        └── PortRef

GraphInputSpec
OutputSpec(ref, grain, relation)
NodeSpec(inputs, outputs, operation, workers, recovery)
ExecutionGraph(inputs, nodes, outputs)
```

每个 output 直接携带一种关系：

- `SameAs(source)`：复用 source identity 和 grain；
- `SubsetOf(source)`：保留 source identity 的 0:1 子集；
- `ChildrenOf(parent, label)`：创建直接 children；`label` 命名 child grain/display path，
  identity 由 parent identity 与 ordinal 派生；
- `AggregateOf(anchor, members, incomplete)`：聚合 descendants 并回到 anchor；
- `RelatedFrom(RoleSource(...))`：从有序 role-parent tuple 派生 M:N identity。

每个 node 直接携带一个 typed operation：`MapOp`、`FilterOp`、`ExpandOp`、
`ReduceOp`、`RelateOp` 或 `FilterByMaskOp`。用户 operator 的重建信息保存在
`OperatorFactorySpec(import_path, args, kwargs)`；Ray 资源保存在
`WorkerPoolSpec(replicas, gpus_per_worker)`；恢复行为保存在 `RecoveryPolicy`。

## Primitive 边界

大原语是 `Map / Filter / Expand / Reduce / Relate`。它们决定一次 UDF invocation 的
作用域和主要关系。`Select` 是 authoring macro，编译时 lower 为：

```text
MapOp → FilterByMaskOp
```

`FilterByMaskOp` 消费 Map 产生的 mask，但不把 mask port 暴露为 output；它只返回过滤后
的原 inputs 与 annotations。

未来只计划为 Expand return 增加两个小 marker：`mg.out.same` 和
`mg.out.children`。新的 relation IR 与 verifier 已能表示、校验 mixed-output identity
forest，但 marker API 和 runtime output materialization **尚未实现**；当前多输出
Expand 仍要求所有 outputs 共享 parent、child 数和 identity。

## 包结构

```text
multigrain/
├── data/          # PortBatch、identity、lineage、errors、grouping
├── ir/            # refs、relations、operations、graph、policy、verify、capabilities
├── tracing.py     # Pipeline、GraphTracer、TracePort
├── primitives/    # authoring wrappers 与唯一 runtime semantics
├── execution/     # typed operation handlers、Local、coordinator、metrics
└── ray/           # opt-in Ray backend、sharding、actor pools、recovery
```

旧的 `ir/model.py`、`ir/passes.py` 及其 contract/pass 模型已移除。当前没有 optimizer
pass、rebatch execution 或 materialize execution。`data.rebatch()` 只是 `PortBatch`
数据辅助函数，不是图 operation。

## 验证与 capability

`verify_graph(graph)` 是普通函数，不是持久化 verifier 对象。它验证拓扑、ref、grain、
operation/relation 配对、Reduce anchor/member、Relate roles，以及 node-local output
forest 无 forward/self source。它还固定 Map/Filter/FilterByMask 的逐输出 source
contract，并要求 `ChildrenOf.label == OutputSpec.grain`。tracing、Local executor 和
Ray executor 都强制调用它。

唯一派生 capability 查询是 `is_row_partitionable(node)`；当前只对 Map、Filter、
FilterByMask 和 Expand 为真。

Ray 在运行自定义 shard planner 后调用 `validate_shard_plan()`，要求 partition indexes
范围合法、无重复，并且对输入 rows **恰好覆盖一次**。

运行时 grain 也被强制校验：每个 output `PortBatch.name` 必须等于对应
`OutputSpec.grain`。Map/Filter 保留 source grain，Expand 使用 child `label`，Reduce
返回 anchor grain，Relate 使用 `output_grain`。

## 当前边界

- `Relate` 当前单输出、whole-batch；
- compiled Relate 只接受 `KeyJoinSpec` 或 dotted `RelationAdapterSpec`；
  `relation_fn` eager-only；
- `Reduce` 当前 whole-batch，不是 distributed two-phase reduce；
- record-level recovery 以 Map 支持最完整；
- mixed-output identity forest 可被 IR/verifier 表示，但 runtime `mg.out` materialization
  deferred；
- UDF 和 relation adapter 必须满足重排不变性所需的纯度/置换等变契约。

## 验证命令

```bash
python -m pytest -q test/experimental/multigrain
```
