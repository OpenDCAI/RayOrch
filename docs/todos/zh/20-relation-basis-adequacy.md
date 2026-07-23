# Multigrain Formal Core 语义与编译可靠性

本文定义可执行 Multigrain MVP 的受支持 formal core，明确不声称一般语言完备性，并与
[重排不变性](13-reordering-invariance-theorem.md)分离：

- **formal-core 覆盖**：当前 authoring contract 如何映射到被动图 relation vocabulary；
- **编译可靠性**：tracing 是否保持 authoring program 的语义；
- **重排不变性**：物理分片与重排是否保持一张已编译图的语义。

结论只针对当前可执行的 finite、typed、direct-role、closed-microbatch authoring
片段；它不声称五种 relation 名称组合完备，也不是 Codd 关系代数、一般图查询或图灵
完备性。

## 1. 语义对象

### 1.1 Port 与 Record

`D` 是不可变 identity domain。`D` 上的一条记录为：

```text
r = (id, v, a, o, ℓ, e)
```

逻辑 key 是 `key_D(r)=(D,id)`：

- `v` 是 UDF 可见值；
- `a : Domain ⇀ ID` 是 functional ancestry；
- `o : Domain ⇀ ℕ` 是 ordinal path；
- `ℓ ∈ Name*` 是 operator lineage；
- `e ⊆ Role × Domain × ID` 是**直接** role-parent evidence。

一个 port 是有限 typed sequence：

```text
P = (D, grain, [r_0, …, r_n])
```

key 只要求在一个 live port/admission scope 内唯一，不假设进程全局或跨
microbatch identity。

### 1.2 源语义 DAG

源语义 DAG 是有限无环图，其 contract 精确对应当前 authoring API：

```text
MapPreserve(input_0)                         1:1
FilterSubset(input_i)                        0:1
Children(parent, finite_ordinal)             1:N
Aggregate(anchor, selectors, incomplete)     N:1
Related(ordered_roles, matcher)              M:N
```

这些 contract 独立于 Python class 和 `ExecutionGraph`，描述 identity 与 dependency：

- MapPreserve 对每个 input-0 key 产生一个值并复用 input 0 的 domain/key；其他 Map
  inputs 必须 identity-aligned，只贡献 metadata，不能被选为另一 identity source；
- FilterSubset 为每个 positional input 产生保持 key 与 source 顺序的子集；
- Children 在派生 domain 中为每个 parent 产生有限有序 cohort；
- Aggregate 按 direct ancestry 或 direct role selector 聚合 descendant fiber，并为每个
  admitted anchor 产生 anchor-keyed row；
- Related 的每条输出由一个有限、有序 role-parent tuple 证明。

## 2. 良构片段 `F_direct`

语义 DAG 属于 `F_direct` 当且仅当：

1. 图有限且无环；
2. 每个 port 是 `(Domain,id)` 唯一的有限序列；
3. row-local transform 确定且 value-pure；
4. aligned row-local inputs 共享 identity domain 和 key order；Map identity 固定来自
   input 0，Filter/FilterByMask 分别保持每个声明的 positional source；
5. child cohort 有限且 ordinal 稳定；
6. 每个 Reduce selector 都是到当前 closed anchor microbatch 的函数：

   ```text
   ByAncestor     π(r) = a(r)(anchor.domain)
   ByRole(role)   π(r) = e(r) 中唯一的直接 edge
   ```

   对 `ByAncestor`，每条 runtime descendant 必须携带唯一 anchor-domain ID，且 parent
   必须存在于 anchor batch。`verify_graph` 只检查结构可达性；Related 后 shared parent
   ID 是否一致属于 runtime/data contract。
7. `ByRole` 只看直接证据；Children 不继承父记录的 role edges；
8. Relate 使用 equi-join 或确定、置换等变的 adapter；
9. 单次有限执行内把 relation hash 视为抗碰撞，runtime 拒绝同一输出 batch 的重复 ID；
10. operation/recovery 组合受所选 backend 支持；该条件由 backend 在执行前检查，而非
    `verify_graph` 检查。

只有组合后仍满足这些条件，片段才对 DAG composition 封闭。仅仅串联五种 cardinality
名称并不自动属于该片段。

## 3. 目标被动图

关系基元映射为：

```text
Preserve-total    ↦ SameAs
Preserve-partial  ↦ SubsetOf
Children          ↦ ChildrenOf
Aggregate         ↦ AggregateOf
Related           ↦ RelatedFrom
```

invocation semantics 由 operation 记录：

```text
MapOp
FilterOp
ExpandOp
ReduceOp(selectors)
RelateOp(matcher)
FilterByMaskOp
```

`OutputSpec.relation` 是 identity/dependency 事实，`NodeSpec.operation` 是执行 recipe；
任何一边单独看都不是完整 node contract。

## 4. 局部表示引理

### 引理 A：保持 identity

formal core 中每个确定性 total row-local Map transform 都由
`MapOp + SameAs(input_0)` 表示；每个确定性 key-preserving Filter predicate 都由
`FilterOp/FilterByMaskOp` 为每个 output source 产生一个 `SubsetOf(input_i)`。

### 引理 B：有限依赖 children

对每个有限 family `C(p)=[c_0,…,c_k]`，`ExpandOp + ChildrenOf(parent)` 表示：

```text
Σ_(p ∈ parent) C(p)
```

runtime 使用派生 child domain、parent ancestry 和 ordinal `i`。

### 引理 C：functional fibers

给定 closed anchor `A` 和 parent function `π`，
`ReduceOp(selectors) + AggregateOf(A)` 表示对每个规范有序 fiber 的确定性 fold：

```text
π⁻¹(a) = [d | π(d)=a.id].
```

### 引理 D：有限 role relation

对有限关系：

```text
J ⊆ P_1 × … × P_k
```

若 evidence 来自 key join 或置换等变 adapter，则
`RelateOp + RelatedFrom(roles)` 对每个
`(value, ordered parent tuple, stable_key)` 产生一行。`ParentRef` 保存不能写入
functional ancestry 的直接多父证据。

## 5. Formal-core 覆盖命题

**命题。** 对每个使用 §1.2 authoring contracts 的良构语义 DAG `S ∈ F_direct`，存在
一张携带对应 operation/relation pair 的 passive `ExecutionGraph G`。当 §2 的
runtime/data obligations 成立时，其 outputs 与 `S` 在 values、record keys、ancestry、
ordinals、lineage 和 direct role evidence 上相同，最多相差派生 domain、node output
及其 lineage occurrence 的确定性命名。

**证明概要。**

对 `S` 拓扑排序：

- inputs lower 为 `GraphInputSpec`；
- 假设所有 predecessor ports 已有 target `PortRef` 和等价 runtime denotation；
- 当前节点按 `F_direct` 定义属于五种 contract 之一，应用引理 A/B/C/D 选择 typed
  operation 和 per-output relation；
- `verify_graph` 检查结构子集：ref 可用、grain 保持、结构 ancestry 可达、direct-role
  availability、matcher shape 与 operation/relation pairing；
- runtime 检查 identity alignment、closed parent batch、output domain/grain 与 backend
  recovery support；确定性、value-purity、shared-parent consistency 和 adapter
  equivariance 是显式 user/data obligations；
- 三层 obligations 全部成立时，新 target ports 保持该节点语义。

有限归纳构造全部 nodes 与 graph outputs。∎

这是当前 authoring core 的**覆盖**结论，不是完备性或最小性结论。强大的 whole-batch
Relate adapter 可以模拟部分其他计算，但不会向 runtime 暴露 locality、cardinality、
canonical order 和 recovery contract。

## 6. Authoring Compilation Soundness

authoring fragment 包含：

```text
Pipeline.forward
Map / Filter / Expand / Reduce / Relate
group_by(anchor, descendants)
via(descendant, role)
Select
```

tracing 只在 `TracePort` 上执行 `forward`。每个 primitive 产生对应 operation/relation。
`Select` 是 macro：

```text
Select_f(inputs)
  ≜ Map_f(inputs) 产生 (mask, annotations)
    然后 FilterByMask(inputs, mask, annotations)
```

**定理（编译可靠性）。** 若 authoring program 成功 tracing、UDF/data 满足
`F_direct` 语义 contract，且所选 backend 支持其 recovery policies，则
`Pipeline.compile()` 返回的 structurally verified `ExecutionGraph` 的执行与 eager
primitive denotation 键控等价。Select 的 eager/traced 路径在相同 mask projection 前
共用同一 Map annotation semantics。

**证明概要。** 对 `forward` 发出的调用做结构归纳。`TracePort.ref` 指向归纳前驱；
primitive lowering 遵循局部表示引理；`GraphTracer.add_node` 记录相同 refs、relations、
operations、worker/recovery policies，`build` 验证结构图；§2 的 backend/runtime
checks 与 semantic obligations 构成定理的其余前提。Select 由 macro 展开得到。∎

该定理不覆盖任意 Python control flow。`forward` 必须返回 traceable ports，output arity
显式确定，依赖 runtime value 的分支不在 tracing 内。

## 7. 反例与非目标

以下不属于 `F_direct`：

- `Relate → Expand → ByRole Reduce`：Expand 有意清空直接 role edges；传递 role path
  需要更丰富的 provenance graph；
- 跨 microbatch join/aggregation、Global、Window、watermark、late data；
- recursion、feedback、fixpoint、动态建图；
- stateful/nondeterministic UDF 与外部副作用；
- mixed-output Expand identity forest（`out.same/out.children`）；
- dataset-scope Partition、union、difference、sort 或当前 contract 未表征的任意
  θ-join；
- monoid contract 形式化前的 distributed/two-phase Reduce。

某些范围外计算可以通过额外业务 key 和 Relate 手工重建，但这不等于框架拥有相应的
optimizer-visible contract。

## 8. 可执行证据

证明义务映射到测试：

- eager/compiled Select parity，包括 multi-input lineage 与 recovery；
- 随机合法 partitions 下每个 port 的完整 keyed metadata；
- 将 intermediate ports 暴露为 graph outputs 后检查 Ray Canon 顺序；
- key-based adapter 在独立 role permutations 下的等价性；
- nested Expand 与 Reduce selectors；
- role path、parent closure、mixed domains、positional adapter 和非法 shard plan 的
  负例。

property tests 是实现证据，不能替代定理前提；任意 Python purity 与 adapter
equivariance 仍是用户 contract。

## 9. 与未来扩展的关系

只有加入新 semantic contract 时才扩展 formal-core 覆盖命题：

- `out.same/out.children`：invocation-local identity forest；
- Partition：固定互斥 routing；
- Global/Window：dataset/time scope 与 completion；
- role paths：传递 provenance selector；
- two-phase Reduce：monoid 与 distributed completion law。

每项扩展必须同时加入 syntax、denotation、well-formedness、lowering、runtime
materialization、verifier checks 与 preservation evidence，不能只复用旧名称便默默扩张
定理。
