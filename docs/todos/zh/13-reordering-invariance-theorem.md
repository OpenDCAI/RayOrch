# Multigrain 执行的重排不变性（M1 形式化）

本文将 `test/experimental/multigrain/test_lineage_under_parallelism.py` 背后的经验事实转化为带证明的形式化陈述：**任何合法分片计划（包括感知工作量的 LPT 再平衡）都保持键控结果与血缘；Ray canonical merge，或 anchor 有序相同的 Reduce，进一步恢复串行可见顺序。** 这是我们能为性能（消除气泡）重排行而不改变逻辑语义的正确性支柱。

下文完全基于实际数据模型，因此该形式化并非空谈：

| 形式对象 | 代码 |
|---|---|
| 静态 port address | `GraphInputRef | NodeOutputRef`（`PortRef`） |
| 静态逐输出关系 | `SameAs`、`SubsetOf`、`ChildrenOf`、`AggregateOf`、`RelatedFrom` |
| 被动执行图 | `ExecutionGraph` / `NodeSpec` / `OutputSpec` |
| 强制结构验证 | `verify_graph` |
| identity namespace | 不可变 `IdentityDomain` |
| record fields | `PortBatch` 平行数组（`values`, `record_ids`, `ancestors`, `ancestor_display`, `ordinals`, `lineage`, `relations`） |
| `take` / `concat` | `PortBatch.take` / `multigrain.data.concat` |
| shard plan 验证 | `validate_shard_plan` |
| Expand lineage | `PortBatchBuilder.expanded` |
| Reduce regroup | `Reduce._groups_for`（`ByAncestor` / `ByRole`） |
| Relate key-join | `Relate._make_key_join_batch` |

`ExecutionGraph` 的静态 relation contract 与运行时证据对应：
`SameAs`/`SubsetOf` 继承 domain；`ChildrenOf`/`RelatedFrom` 创建新 domain；
`AggregateOf` 回到 anchor domain，并按声明的 parent function
（`ByAncestor` 或 `ByRole`）分组。`verify_graph` 检查 ref、grain 和 typed
operation 配对及结构 ancestry 可达性；RelatedFrom 后 shared ancestor ID 是否一致由
runtime/data contract 保证。证明建立在完整记录 `(id,v,a,o,ℓ,e)` 上。

## 0. Fragment claim 与非目标

本文形式化的是 **multi-grain identity algebra fragment**，不是经典 Codd 关系代数。

**支持片段。** 有限 DAG、closed microbatch，基数变化仅来自：

```text
1:1   SameAs          (Map)
0:1   SubsetOf        (Filter / FilterByMask)
1:N   ChildrenOf      (Expand)
N:1   AggregateOf     (Reduce via ByAncestor 或 ByRole)
M:N   RelatedFrom     (Relate)
```

**Motif 覆盖，不是完备性定理。** 五种 relation 命名了当前可执行的局部 motif，
但组合必须满足良构规则：`ByRole` 只消费直接 `RelatedFrom` evidence，
`ChildrenOf` 创建的 child cohort 不继承 role edges。独立的
[Formal Core 文档](20-relation-basis-adequacy.md)定义受支持片段与 lowering
结论；本文主定理只讨论一张已验证图的执行重排。

**明确非目标：**

- 经典 RA 完备（union、集合差、任意 θ-join）；
- 一般图查询（path、传递闭包、递归 CTE）；
- 跨 microbatch 的 Global/Window / watermark；
- `Relate → Expand → ByRole` 等传递 role-path 导航；
- Expand mixed-output forest（`mg.out.same/children`，deferred 扩展）。

## 1. 数据模型

**Domain。** `Domain` 是不可变 identity namespace（`IdentityDomain`）。batch 落在一个
当前 domain `D` 上。grain 与 current port 只是标签，不是 ancestry key。

**记录。** 一个记录是元组 `r = (id, v, a, o, ℓ, e)`，位于 domain `D`；其逻辑
record key 是 `key_D(r)=(D,id)`：

- `id ∈ ID_D` ——在 `D` 的 live port/admission scope 内唯一的 `record_id`，不是
  进程全局 ID；
- `v` ——payload；
- `a : Domain ⇀ ID` ——祖先映射（`ancestors[i]`），**偏函数**，以
  `IdentityDomain` 为 key；
- `o : Domain ⇀ ℕ` ——ordinal 映射；
- `ℓ ∈ Name*` ——算子路径；
- `e ⊆ Role × Domain × ID` ——直接 role-parent 证据（`relations[i]`）。
  Relate 只把无歧义 shared ancestors 写入 `a`；冲突多父完整保留在 `e`。

**不变量 I0（functional ancestry）。** `a` 对每个 domain 至多一个祖先，因此始终描述
forest。会破坏功能性的多父结构只存入 `e`，绝不覆盖进 `a`。

**批次。** `B = [r_0,…,r_{n-1}]` 是同一 domain 上的有限序列。executor 强制
`PortBatch.name == OutputSpec.grain`。

**端口不变量 I1（身份唯一性）。** 每个 live `PortBatch` 内 `(D,id)` 两两不同；
由于一个 batch 只有一个 `D`，runtime 直接检查 ids 两两不同。Sources 产生
`name:i`，后续 admission 可以复用这些字符串，因此不声称跨 microbatch 唯一。
Expand 产生 `op:parent_id:k`；Relate 对有序
`(role, domain, parent_id)` tuple 与 stable key 做稳定 hash；Map/Filter 保留
ids。`PortBatch.__post_init__` 在 runtime boundary 拒绝重复 ID；relation hash
唯一性依赖 §8 的抗碰撞假设。

**键控视图。** `⟦B⟧ : (Domain × ID) ⇀ Record`。当且仅当
`⟦B⟧ = ⟦B'⟧`（相同 record keys，且每个 key
的 `(v,a,o,ℓ,e)` 相同）时，`B ≈ B'`。`≈` 忽略物理顺序，但严格比较全部
per-record fields，**包括 role edges `e`**。

### 1.1 Domain homomorphism（IR → runtime）

| 输出关系 | runtime domain |
|---|---|
| graph input | `source(...)` admission 时的 named/fresh domain |
| `SameAs(s)` / `SubsetOf(s)` | 继承 `δ(s)` |
| `ChildrenOf(p)` | 由 `(op, δ(p))` 派生的新 domain |
| `AggregateOf(anchor, ·)` | 回到 `δ(anchor)` |
| `RelatedFrom(roles)` | 由 `(op, δ(role ports)…)` 派生的新 domain |

aligned Map/Filter 进一步要求全部 inputs 共享同一 domain；仅 grain 相同不够。

### 1.2 Reduce 的 parent function

```text
π : DescendantRecord → AnchorId

ByAncestor     π(r) = a(r)(δ(anchor))
ByRole(ρ)      π(r) = e(r) 中唯一的 (ρ, δ(anchor), id) 之 id
```

二者都要求 `π` 的像落在当前 closed anchor microbatch 内，否则抛 closure
violation。无 `ByRole` 的多父歧义直接拒绝。`ByRole` 只读取当前记录的直接
role evidence；Expand 不复制父记录的 role edges，因此 role path 与一般图遍历不在
当前片段内。

## 2. 物理执行模型

**take / concat。** `take` 完整复制 `value/ancestors/ordinals/lineage/relations`；
`concat` 串接平行数组，并对 inherited `ErrorTrace` 去重。

**分片计划。** `σ` 合法当且仅当它是 `{0..n-1}` 的集合分割。Ray 强制调用
`validate_shard_plan`。

**分片节点执行：**

```
RawExec_σ(N)(P) = concat_j ( N( take(P^0,σ_j), …, take(P^t,σ_j) ) )
Exec_σ(N)(P)    = Canon_N,P(RawExec_σ(N)(P))
```

`Canon` 将 SameAs/SubsetOf 按 source logical position，ChildrenOf 按
`(parent logical position, child ordinal, record_id)` 恢复顺序。

**良构性 WF。** sharded multi-input 要求同 domain 且同 id-order；Ray 在 planner 前
检查。

## 3. 重组引理

**引理 0。** 对任意 batch `B` 与合法 `σ`，`concat_j take(B,σ_j)` 是 `B` 的置换。

## 4. 行独立算子（Map、Filter、FilterByMask、Expand）

- **Map**：保留 `id/a/o/e`（合并 aligned inputs 的 ancestry 与 role evidence），追加
  `ℓ`。
- **Filter / FilterByMask**：保留或丢弃整行，kept 行保留 `id/a/o/ℓ/e`。
- **Expand**：在新 domain 上产生 children，
  `a' = a ∪ {D_p ↦ p}`，`o' = o ∪ {D_p ↦ k}`，`e' = ∅`。

**引理 1。** WF 下对行独立 `N` 与任意合法 `σ`：
`RawExec_σ(N)(P) ≈ Serial(N)(P)`；canonical merge 后
`Exec_σ(N)(P) = Serial(N)(P)`。
其中 `≈` 比较完整 `(v,a,o,ℓ,e)`。

## 5. 顺序规范化算子（Reduce）

```
group_π(A[q], D) =
  [ v : r ∈ D, π(r) = A.record_id(q) ]
  sorted by domain-qualified ordinal path, then record_id
```

**引理 2。** 若 `D ≈ D'`、anchor `A` 相同且 selector `π` 相同，则
`Reduce_π(A,D) = Reduce_π(A,D')` 作为有序序列。
membership 由 `a` 或 `e` 决定，sort key 由 `o`（及 `id`）决定，二者均在 `≈`
下保留。

## 6. Relate

输出 identity 是有序 `(role, domain, parent_id)` tuple 的稳定 hash。
Relate 写入完整 role edges `e`，只把无歧义 shared ancestors 压入 `a`。

**引理 3a / 3b。** key-join 与满足置换等变契约的 adapter 都给出
`Relate(P) ≈ Relate(P')`（含 `e`）。Relate **本身不是**顺序规范化器：role inputs
仅键控相等时发射顺序可能变；有序相等需要上游已 ordered（Ray 由 `Canon` 提供）。

## 7. 主定理

```text
RawPhys(p)  = take/concat 分片后、未经 Canon
Phys(p)     = Ray 模型：每个 sharded 行独立节点后做 Canon
Ser(p)      = 串行整批执行
```

**定理（重排不变性）。**

1. **（处处键控/血缘相等）** `RawPhys(p) ≈ Ser(p)`。完整 `(v,a,o,ℓ,e)` 键控相同。
2. **（Canon 后 / Reduce 有序相等）**
   - 行独立 port 在 Ray 模型下：`Phys(p) = Ser(p)`；
   - Reduce：在 **anchor 有序相同**（`A_phys = A_ser`）时，即使 descendants 仅
     `≈` 也有序相等；若 anchor 本身只是键控相等且被置换，则 Reduce 仍键控相等，
     但按置换后的 anchor 顺序发射；
   - Relate：仅当 role inputs 已有序时升级；一般只保证第 1 部分。

*证明概要。* 第 1 部分用引理 1/2/3；第 2 部分依赖 Ray `Canon`，以及 Reduce 在
ordered anchor 前提下的 ordinal 规范化。MVP 测试管线的 Reduce anchor 都是
graph input，满足更强前提。∎

**推论（恢复对重排稳定）。** 在确定性失败分类与错误渲染前提下，
`ErrorTrace` 由 record key、ancestry/display projection、lineage、selected parent、
失败算子、recovery action 与 error text 决定。这些输入在键控执行下稳定，因此
quarantine 定位与 trace 对合法重排稳定。

**推论（LPT 对正确性安全）。** 合法 LPT 不改变键控结果；Ray Canon 下也不改变
Map/Filter/Expand/Reduce 的用户可见顺序。不声称 lineage 选择调度计划。

## 8. 假设、范围和坦诚限制

- sharded multi-input 需要 WF（同 domain + 共序）。
- Reduce/Relate 在 `CLOSED_MICROBATCH` 内 whole-batch；跨批 Global/Window 不在定理范围。
- UDF value-purity；adapter 置换等变。
- relation identity 使用 canonical role evidence 的 SHA-256；证明采用单次有限执行内
  的抗碰撞假设，runtime 仍拒绝同一输出 batch 的重复 ID。
- 本文证明重排不变性，不证明语言完备性。经典 RA、图递归、跨批 window、
  mixed-output Expand 与传递 role path 都在可执行片段外；受限语言结论见独立的
  Relation-Basis Adequacy 文档。

## 9. 机器检查证据

| 断言 | 证据 |
|---|---|
| 第 1 部分处处 `≈` | `test_reordering_invariance.py`（本地 take→concat，**无** Canon） |
| Reduce 有序相等 | 同上，最终 Reduce port |
| Canon 后 Map/Filter 有序 | Ray `test_reordered_shards_canonize_exposed_intermediate_outputs` |
| Canon 后 Expand 有序 | Ray `test_reordered_shards_canonize_expand_output` |
| 完整 deterministic ErrorTrace 相等 | Ray `test_quarantine_localizes_same_page_under_reordered_parallelism` |
| ByRole / closure / domain | Semantic Hardening 与 relate suites |

诚实缺口：本地 property harness 强验证第 1 部分；第 2 部分“Canon 下处处有序”
在 Ray 模型中成立并有抽检，但尚未在本地 harness 的每个中间 port 上做
property-based 全覆盖（该 harness 有意省略 Canon 以隔离 `≈`）。
