# Multigrain V2.3 精简原型设计稿

状态：V2.3 原型主旨与最小架构草案。

本文重新收敛 Multigrain 的目标。V2.3 不追求一次性建立通用dataflow runtime、完整
provenance系统或durable workflow engine，而是优先实现并验证一个明确闭环：

```text
dynamic 1:M fan-out
-> child-level reorder / rebatch
-> lineage-preserving execution
-> M:1 fiber reconstruction / Reduce
-> item-level fault isolation and recovery
```

若本文与V2.2在后续runtime复杂度上冲突，V2.3原型优先采用本文的精简方案。V2.2已经完成
的纯errors、column、graph和identity测试可以选择性复用，但不得因此恢复本文明确删除的
复杂机制。

## 1. 一句话命题

> Multigrain是面向动态fan-out/fan-in applied-ML pipelines的Ray原生执行框架：它把
> Expand产生的children从parent-bound physical batches中解耦，跨parent重排和重新batch
> 以减少长尾气泡，同时保持item lineage，用同一lineage重建Reduce fibers并完成item级
> 错误定位、传播、隔离和恢复。

论文与原型的核心不是LPT、lineage或retry中的任意单点，而是三者围绕动态1:M→M:1
pipeline形成的闭环。

## 2. 目标

V2.3只优先解决以下问题：

1. **Dynamic fan-out**：一个parent可产生零个、一个或多个children。
2. **Elastic rebatching**：children不绑定原parent batch，可跨parent重排并组成更均衡的
   actor batches。
3. **Lineage preservation**：无论children如何分片、重排和重试，仍能恢复parent、child
   ordinal和direct parents。
4. **Correct fan-in**：Reduce只消费一个parent对应的完整、有序fiber，并输出回parent
   identity。
5. **Item fault isolation**：定位失败item，向上trace其source/parents，向下抑制依赖它的
   item或fiber，同时保留无关健康工作。
6. **Friendly API**：沿用RayOrch `RayModule`的UDF class、`pre_init()`、`ray_options()`
   配置风格，并提供五个数据流原语。

## 3. 明确不做

首版V2.3不实现或不主张：

- driver crash recovery、checkpoint/resume；
- external exactly-once、事务sink或side-effect rollback；
- 通用window/state/iteration streaming；
- distributed lineage authority；
- durable、跨run稳定的identity编码；
- 通用provenance查询语言或全量lineage持久化；
- arbitrary custom Relate evidence protocol；
- publication后ObjectRef的局部修复；
- 任意Python UDF的自动语义证明；
- universal optimal scheduler；
- byte-perfect memory admission或byte-sparse transfer；
- 一次性支持所有V2.2 failure state-machine边界。

这些能力可以在核心闭环被实验验证后再增加。

## 4. 用户API

顶层API保持有限：

```text
Pipeline
Map / Filter / Expand / Reduce / Relate
Port / keyed
Executor / RunResult
ItemError / ExecutionError / CompileError
```

Module配置风格参考`RayModule`：

```python
class DocumentPipeline(mg.Pipeline):
    def __init__(self, model: str) -> None:
        self.read_pages = (
            mg.Expand(ReadPages)
            .pre_init(dpi=144)
            .ray_options(
                replicas=4,
                batch_size=8,
                num_cpus=2,
            )
        )
        self.ocr = (
            mg.Map(Ocr)
            .pre_init(model=model)
            .ray_options(
                replicas=8,
                batch_size=16,
                num_gpus=1,
                error_policy="isolate",
                max_retries=1,
                cost_fn=estimate_page_cost,
            )
        )
        self.assemble = (
            mg.Reduce(Assemble)
            .ray_options(replicas=4, batch_size=8)
        )

    def forward(self, documents):
        pages = self.read_pages(documents)
        texts = self.ocr(pages)
        return self.assemble(documents, texts)
```

配置含义：

- `replicas`：该node的persistent actor数量；
- `batch_size`：一个physical RPC目标包含的semantic items/fibers数量；
- `cost_fn`：可选item工作量估计；缺省时每个item权重为1；
- `error_policy="raise" | "isolate"`：普通UDF异常是终止还是允许递归隔离；
- `max_retries`：基础设施错误或可重试执行错误的有限预算；
- `num_cpus/num_gpus/resources/runtime_env`：Ray资源参数的有限白名单。

`pre_init()`只配置UDF constructor；`ray_options()`只配置执行。两者均返回同一module recipe，
不在authoring阶段创建actors。

## 5. 五原语

### 5.1 Map

```text
item -> item
```

- 输出继承primary input的`ItemKey`；
- 多输入按`ItemKey`对齐，不按物理位置zip；
- 多输出共享同一identity layout；
- 每个physical batch可包含来自不同parents的items。

### 5.2 Filter

```text
item -> zero-or-one item
```

- survivor继承target identity；
- predicate为false是正常业务结果，不生成failure；
- predicate执行错误生成`ItemFailure`；
- 正常filtered与missing必须可区分。

### 5.3 Expand

```text
parent -> zero-or-many children
```

UDF按parent batch执行并返回：

```python
list[list[Child]]
```

runtime flatten后为每个child生成：

```text
child key = (expand node, parent key, sibling ordinal)
parent ref = parent item
ordinal = sibling ordinal
```

Expand同时发布每个parent的child count。该count是后续判断Reduce fiber是否complete的
authority。

### 5.4 Reduce

```text
parent + complete child fiber -> parent output
```

Reduce是1:M故事闭环中的必要原语，不是附属操作。

children经过跨parent重排后，runtime必须使用lineage重新构造：

```text
parent A <- child A0, A1, A2
parent B <- child B0, B1
```

fiber内部按child ordinal恢复稳定顺序。Reduce UDF的batch ABI为：

```python
parents: list[Parent]
groups: list[list[ChildResult]]
```

一个Reduce RPC可以包含多个完整fibers，但一个fiber不得跨RPC拆分。Reduce输出复用parent
identity。

首版只有`fail_closed`：

- 全部expected children成功：fiber ready，调用Reduce；
- 任一required child失败：该fiber suppressed，不生成不完整parent结果；
- 其他parents正常Reduce并输出。

### 5.5 Relate

```text
multiple keyed roles -> multi-parent items
```

首版只实现key-based Relate：

```python
pairs = self.join(
    left=mg.keyed(left_rows, by=left_keys),
    right=mg.keyed(right_rows, by=right_keys),
)
```

- role顺序来自UDF signature；
- 相同key形成完整parent tuples；
- output记录带role的direct parents；
- parent失败时只抑制依赖它的relation items或对应key group；
- arbitrary Custom Relate延期。

Relate不是首篇论文性能故事的必要主角，但用于证明通用multi-parent lineage和failure
propagation。

## 6. 最小内部模型

首版只保留以下核心记录：

```python
@dataclass(frozen=True, slots=True)
class ItemKey:
    batch: int
    path: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class ItemRef:
    port: int
    key: ItemKey


@dataclass(frozen=True, slots=True)
class ParentRef:
    role: str
    item: ItemRef


@dataclass(frozen=True, slots=True)
class ItemMeta:
    key: ItemKey
    parents: tuple[ParentRef, ...]
    ordinal: int | None = None


@dataclass(frozen=True, slots=True)
class ItemFailure:
    item: ItemRef
    node: str
    kind: str
    message: str
    causes: tuple[ItemRef, ...]
```

设计原则：

- `ItemKey`只要求同一active run内跨rebatch/retry稳定；
- 不冻结跨run、跨版本或跨语言的canonical bytes；
- direct parents就是执行所需lineage，不再建立五套独立provenance records；
- primitive kind和parent roles共同解释relation semantics；
- value与metadata分离，driver只处理metadata和ObjectRefs，不读取大型业务payload。

## 7. Port状态与lineage索引

每个逻辑port维护：

```python
@dataclass
class PortState:
    items: list[ItemMeta]
    shards: list[ray.ObjectRef]
    locations: list[tuple[int, int]]  # shard index, row index
    failures: dict[ItemKey, ItemFailure]
```

同一microbatch额外维护两个轻量索引：

```text
item -> direct parents
parent item -> direct dependents
```

用途：

- backward trace：从失败item递归找到source parents；
- forward propagation：找到依赖失败item的下游items/fibers；
- Reduce grouping：通过Expand parent ref恢复fiber；
- diamond alignment：通过ItemKey查找，不使用物理position。

索引只在active microbatch内存活，result delivery后释放。

## 8. Fan-out、重排与rebatching

Expand输出flatten后进入node-local pending item queue。scheduler从多个parents的children中
组成physical batches。

默认策略：

1. `cost_fn`缺省时按item count组成batch；
2. 有cost时使用简单LPT/greedy packing；
3. 保持`batch_size`上限；
4. 同一semantic item在一个attempt中只进入一个active RPC；
5. physical batch顺序不影响ItemKey、parents或最终Reduce结果。

示例：

```text
doc A children costs: 9, 8, 1
doc B children costs: 7, 2
doc C children costs: 6, 5, 1

parent-bound:
  actor 0 <- A: 18
  actor 1 <- B: 9
  actor 2 <- C: 12

rebatch:
  actor 0 <- A0, C2: 10
  actor 1 <- A1, B1: 10
  actor 2 <- B0, C0: 13
  actor 3 <- C1, A2: 6
```

首版不声称LPT算法新颖。研究价值来自：动态1:M itemization、跨parent packing、lineage
correctness和fault containment的共同设计。

## 9. Fiber完成与Reduce触发

每个Expand parent建立一个fiber ledger：

```python
@dataclass
class FiberState:
    parent: ItemRef
    expected: int
    succeeded: dict[int, ItemRef]
    failed: dict[int, ItemFailure]
```

状态规则：

```text
len(succeeded) == expected
-> ready for Reduce

failed is non-empty
-> terminal suppressed fiber

otherwise
-> pending
```

零child是complete empty fiber，合法调用Reduce。多个ready fibers可再次batch执行Reduce。

该ledger使Reduce无需等待所有parents完成；任何fiber一旦terminal即可独立进入Reduce或
failure publication，从而支持fan-out/fan-in pipeline overlap。

## 10. 错误与恢复

### 10.1 错误分类

只区分三类：

```text
InfrastructureError
Explicit ItemError
Generic UDF error
```

#### InfrastructureError

actor crash、timeout、Ray transport failure：

- 废弃该physical batch结果；
- 替换actor；
- 在有限预算内重试同一items；
- 不生成业务`ItemFailure`。

#### Explicit ItemError

```python
raise mg.ItemError("corrupt page", index=bad_index)
```

- `index`定位当前RPC中的semantic item/fiber；
- 该item进入permanently failed；
- 同batch健康items重新执行或保留已确认结果；
- failure进入lineage传播。

#### Generic UDF error

默认`error_policy="raise"`直接终止当前microbatch。

`error_policy="isolate"`时：

1. 对physical batch递归二分；
2. 成功子范围正常发布；
3. singleton仍失败则记录该item的`ItemFailure`；
4. 不允许无限重试。

用户选择`isolate`意味着确认该UDF的batch输出对peer items独立。首版不尝试自动证明任意
Python UDF的separability。

### 10.2 Failure传播

```text
failed item
-> backward parent trace for diagnostics
-> reverse dependency lookup
-> primitive-specific downstream suppression
```

规则：

- Map：primary parent失败，目标item suppressed；
- Filter：target或mask item失败，目标item suppressed；
- Expand：parent失败，不产生未知children；
- Reduce：任一required child失败，整个parent fiber suppressed；
- Relate：parent失败，包含该parent的relation items/key group suppressed。

传播只影响lineage可达的items，不把整个physical batch或microbatch静默丢弃。

## 11. 最小Ray执行协议

actor是持久UDF实例，接口只需要：

```python
run(
    attempt: int,
    selectors: BatchSelectors,
    *value_refs: ray.ObjectRef,
) -> tuple[BatchManifest, OutputColumns]
```

driver维护：

- node queues；
- actor availability；
- pending item/fiber metadata；
- value ObjectRefs与selectors；
-有限retry/split状态；
- lineage和failure indexes。

actor负责：

- 从resolved value shards按selector构造list columns；
- 调用UDF；
- 返回output columns、offsets和显式item error；
- 不生成ItemKey或lineage。

每个RPC带递增attempt token。旧attempt晚到时整体丢弃。首版不建立更复杂的多级attempt
registry。

## 12. Pipeline调度

运行单位是bounded microbatch：

```text
admit source rows
-> execute ready nodes
-> Expand child queues
-> rebatch child work
-> publish child metadata and values
-> reconstruct ready fibers
-> batch Reduce
-> deliver outputs/failures
```

不同microbatches可由`max_inflight`重叠。首版不跨microbatch合batch，以避免identity、
cleanup和错误归属复杂化。

## 13. 最小package布局

```text
rayorch/experimental/multigrain_v2_3/
├── __init__.py
├── api.py          # Pipeline、五原语、配置链
├── graph.py        # Port、NodeSpec、CompiledGraph
├── item.py         # ItemKey、ItemMeta、ParentRef
├── lineage.py      # parent/dependent索引、fiber reconstruction
├── scheduler.py    # pending queues、cost-aware rebatching
├── recovery.py     # retry、split、ItemFailure propagation
├── worker.py       # Ray actor与batch invoke
├── executor.py     # microbatch DAG coordinator
└── metrics.py      # throughput、utilization、recompute、failure指标
```

不提前拆分更多层；仅当一个模块出现明确独立合同和测试需求时再拆分。

## 14. 必须保持的不变量

1. physical batch、actor和completion order不进入`ItemKey`。
2. Map/Filter输出复用primary identity。
3. Expand child identity由parent和ordinal唯一确定。
4. 每个child保存parent ref和ordinal。
5. Reduce只消费完整fiber；一个fiber不跨RPC拆分。
6. Reduce output复用parent identity。
7. 多输入按ItemKey或显式key relation对齐，不按物理position zip。
8. 正常Filter false与ItemFailure严格区分。
9. generic error不会在默认策略下静默变成missing。
10. failure只向lineage可达的downstream items传播。
11. 大型业务values不经过driver反序列化。
12. 同一items同一时刻最多属于一个active RPC attempt。

## 15. 研究问题

### RQ1：Dynamic fan-out效率

当parent fanout或child service time呈长尾分布时，跨parent rebatching能否提高：

- throughput；
- GPU/actor utilization；
- p95/p99 parent completion latency；
- pipeline overlap。

### RQ2：Fan-in正确性

在任意合法rebatch、shard、completion order下，lineage能否稳定重建：

- parent membership；
- child ordinal；
- complete-empty fibers；
- diamond alignment；
- Reduce outputs。

### RQ3：Failure containment

稀疏item错误下，相比whole-batch retry/drop，lineage-directed isolation能否：

- 减少weighted recomputation；
- 减少healthy item误丢失；
- 只抑制受影响parent fibers；
- 保持其他parents正常完成。

### RQ4：成本

测量：

- lineage metadata bytes/item；
- scheduler CPU；
- driver RSS；
- selectors导致的transfer amplification；
- split RPC数量；
- 无故障吞吐开销。

## 16. 最小实验矩阵

### Synthetic

同时sweep：

```text
parent fanout distribution: uniform / log-normal / Zipf
child service time: uniform / log-normal / Pareto
batch size
replicas
item failure rate
failure position
```

对比：

```text
parent-bound batching
flat child batching without lineage-aware Reduce
Multigrain rebatching
Multigrain without cost scheduling
whole-batch retry/drop
item isolation
```

### Real workloads

至少两个真实fan-out/fan-in workloads：

1. document -> pages/blocks -> model processing -> document assembly；
2. video/media -> frames/clips -> inference -> video-level aggregation。

所有性能结果必须包含最终Reduce output，不得只测Expand后的child吞吐。

### 核心图

图一：

```text
x = fanout / service-time skew
y = throughput or GPU utilization
```

图二：

```text
x = bad-item rate
y = weighted recomputation and completed healthy parents
```

## 17. 论文贡献表述

正文收敛为三项：

1. **Dynamic itemization and elastic rebatching**：从1:M fan-out提取可独立调度children，
   跨parent重拍以减少长尾气泡。
2. **Rebatch-invariant lineage and fiber reconstruction**：在任意物理重排后恢复parent、
   ordinal和完整Reduce fibers。
3. **Lineage-directed fault containment**：定位失败item，向上trace并向下传播，只抑制受
   影响fibers并保留健康工作。

推荐的一句话论文定位：

> Multigrain provides lineage-preserving elastic execution for dynamic
> fan-out/fan-in machine-learning pipelines.

## 18. V2.3原型成功标准

原型只有同时满足以下条件才算完成核心闭环：

- Document-like 1:M→M:1 pipeline端到端运行；
- children确实跨parents重排并形成更均衡physical batches；
- 任意重排后Reduce结果与serial oracle一致；
- child ordinal稳定；
- 单个bad child只抑制其parent fiber；
- 其他parent outputs正常交付；
- infrastructure retry不会产生重复或错配items；
- 无故障lineage/rebatch开销可测；
- API只暴露五原语和RayModule式配置，不暴露内部调度类型。

在这些条件达成前，不增加checkpoint、exactly-once、distributed metadata或通用custom
relation等非核心能力。
