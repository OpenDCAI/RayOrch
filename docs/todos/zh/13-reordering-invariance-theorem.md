# Multigrain 执行的重排不变性（M1 形式化）

本文将 `test/experimental/multigrain/test_lineage_under_parallelism.py` 背后的经验事实转化为带证明的形式化陈述：**任何合法分片计划（包括感知工作量的 LPT 再平衡）都产生与串行基线相同的结果和血缘。** 这是我们能为性能（消除气泡）重排行而不改变语义的正确性支柱。

下文完全基于实际数据模型，因此该形式化并非空谈：

| 形式对象 | 代码 |
|---|---|
| record fields | `PortBatch` 平行数组（`values`, `record_ids`, `ancestors`, `ancestor_display`, `ordinals`, `lineage`），位于 `multigrain.data.batch` |
| `take` | `PortBatch.take(indices)` |
| `concat` | `multigrain.data.concat(batches)` |
| shard/merge | `MultigrainRayExecutor._run_node`（`multigrain.ray.executor`，约第 133–165 行） |
| shard plan | `shard_planner(node, inputs, replicas) -> list[list[int]]`；`lpt_shard_planner`、`_contiguous_ranges` |
| Expand lineage | `PortBatchBuilder.expanded`（`multigrain.primitives.output`） |
| Reduce regroup | `Reduce._groups_for`（`multigrain.primitives.expand_reduce`） |
| Relate key-join | `Relate._make_key_join_batch`（`multigrain.primitives.relate`） |

## 1. 数据模型

**记录。** 一个*记录*是元组 `r = (id, v, a, o, ℓ)`：

- `id ∈ ID` ——全局唯一的身份字符串（`record_id`）；
- `v` ——payload value；
- `a : Name ⇀ ID` ——*ancestor map*（`ancestors[i]`），从 producer port name 到该 port 上 ancestor record id 的部分映射；
- `o : Name ⇀ ℕ` ——*ordinal map*（`ordinals[i]`），一个 ancestor 下的 child index；
- `ℓ ∈ Name*` ——*lineage path*（`lineage[i]`），记录经过的 operators 序列。

（`display_keys` / `ancestor_display` 是相同数据的展示投影，且由 `take`/`concat` 原样承载，因此在证明中省略；它们遵循相同论证。）

**批次。** 一个*批次* `B = [r_0, …, r_{n-1}]` 是记录的有限**序列**。`B[i]` 是第 i 个记录；`|B| = n`；`ids(B)` 为 ids 序列。

**端口不变量 I1（身份唯一性）。** 在系统产生的每个 port 上，ids 两两不同。（Sources 产生 `name:i`；`Expand` 产生 `op:parent_id:k`；`Relate` 产生 `op:role_1=pid_1|…|role_n=pid_n[|key=stable_key]`；`Map`/`Filter` 保留 ids。`Relate` 会拒绝没有不同 stable key 的重复 parent evidence。唯一性由构造维持。）

**键控视图。** 定义 `⟦B⟧ : ID ⇀ Record`，其中 `⟦B⟧(id) = r`，`r ∈ B` 是具有该 id 的唯一记录（由 I1 良定义）。当且仅当 `⟦B⟧ = ⟦B'⟧`（作为完整记录集合相等——每个 id 相同，且每个 id 的 `(v,a,o,ℓ)` 相同）时，两个批次**键控相等**，记作 `B ≈ B'`。当 `B ≈ B'` 且 `|B| = |B'|` 时，`B` 是 `B'` 的**置换**（等价地，`B'` 重排 `B`）。

`≈` 忽略物理顺序，但严格比较每个 per-record field，因此在一个 port 证明 `≈` 已证明该 port 的*血缘相等*。

## 2. 物理执行模型

**take。** 对 `B` 中互异 indices 的列表 `σ = [σ_1,…,σ_k]`，
`take(B, σ) = [B[σ_1], …, B[σ_k]]` ——每个记录完整复制的子序列（`PortBatch.take` 逐元素复制 `value`、`ancestors`、`ordinals`、`lineage`）。

**concat。** `concat(B_1,…,B_m) = B_1 ⧺ … ⧺ B_m` ——序列串接，每个记录完整保留（`core.concat` 逐元素扩展平行数组）。

**分片计划。** 一个 n 行 port 的*分片计划*为 indices lists 的元组 `σ = (σ_1,…,σ_m)`。仅当 `{σ_1,…,σ_m}` 是 `{0,…,n-1}` 的**集合分割**时它才**合法**：`σ_j` 两两不交，且 `⋃_j σ_j = {0,…,n-1}`。

> `_contiguous_ranges` 显然产生合法计划。`lpt_shard_planner` 将每个 index `i` 恰好分配给一个 bin（遍历全部 `i` 的循环内的 `bins[target].append(i)`），故也产生合法计划。合法性是证明使用的 planner 的*唯一*性质——优化器可任意重排/再平衡。

**分片节点执行**（镜像 `_run_node`）：对输入为 `(P^0,…,P^t)`（port 0 是 base）的节点 `N`，以及 `|P^0|` 上的合法计划 `σ`：

```
Exec_σ(N)(P^0,…,P^t) = concat_j ( N( take(P^0,σ_j), …, take(P^t,σ_j) ) )
```

按每个 output port 合并。`Serial(N) = N(P^0,…,P^t)` 是整批运行（`m = 1`，`σ_1 = [0..n-1]`）。

**良构性 WF（共序输入）。** 对一个*分片的*多输入节点，所有输入 ports 以相同 id-order 呈现记录，即 `ids(P^0)=…=ids(P^t)`，故位置 `take(P^r, σ_j)` 在每个 port 上选择同一 id-set。（单输入分片节点平凡满足 WF。MVP 中仅 `Map`/`Filter`/`Expand` 被分片；节点内的 `_align_by_identity` 随后在 shard 内按 id 重新配对，若 WF 被违反则以可读错误*拒绝*，而不是悄然错误 join。）

## 3. 重组引理

**引理 0（分割重组）。** 对任何批次 `B` 和合法计划 `σ`，`concat_j take(B, σ_j)` 是 `B` 的置换。

*证明。* 由合法性，每个 index `i ∈ {0..n-1}` 恰好出现于一个 `σ_j`，且在该列表中恰好一次，因此它被恰好一个 `take(B,σ_j)` 选择一次，且 `B[i]` 被完整复制。`concat` 聚集所有选中记录，故结果恰好包含每个 `B[i]` 一次 ⇒ 相同 id-set、相同 per-record fields、相同长度 ⇒ 为 `B` 的置换。∎

## 4. 行独立算子（Map、Filter、Expand）

**定义（行独立）。** 若存在 per-row function `f_N`，使得对于已对齐输入行，输出是 per-row images 的有序串接，且 `f_N` 仅依赖 record content、不依赖该行位置或其他行，则 `N` 是*行独立的*：

```
N(P^0,…,P^t) = concat_i  f_N( aligned_i )
```

其中 `aligned_i` 是第 i 个 base row 与其他 ports 上同 id partners 的 identity-aligned tuple，`f_N(aligned_i)` 的批次长度为 1（`Map`）、0 或 1（`Filter`），或 `k_i ≥ 0`（`Expand`）。

- **Map** `f = ` 对已对齐行应用 UDF，保留 id/ancestors/ordinals，向 `ℓ` 追加 op。长度为 1。（`multigrain.primitives.map_filter`。）
- **Filter** `f = ` 若 mask 为 true 则为 `[row]`，否则为 `[]`；保留行保持 identity。长度 0/1。（`multigrain.primitives.map_filter`。）
- **Expand** `f = ` 对 id 为 `p` 的 parent row，产生 children：
  `id = op:p:k`，`a' = a ∪ {portname ↦ p}`，`o' = o ∪ {portname ↦ k}`，
  `ℓ' = ℓ⧺[op]`，其中 `k = 0..k_p-1`。长度 `k_p`。仅依赖 parent record 自身的 value（UDF 看到 parent value 并返回其 group）。（`multigrain.primitives.output::PortBatchBuilder.expanded`。）

每个 `f_N` 都是仅关于 record content 的纯函数——没有 `i`、没有 cross-row state——这正是重排安全的原因。

**引理 1（分片与行独立节点可交换）。** 对 WF 下的行独立 `N` 和任意合法计划 `σ`：`Exec_σ(N)(P) ≈ Serial(N)(P)`。

*证明。* 由引理 0，`concat_j take(P^0,σ_j)` 是 `P^0` 的置换；在 WF 下，同一 index sets 在每个输入 port 选择相同 ids，所以 shard `j` 恰包含 `{aligned_i : i ∈ σ_j}` 中的已对齐行。由于 `N` 独立地对每个已对齐行应用 `f_N`，

```
N(take(P,σ_j)) = concat_{i∈σ_j} f_N(aligned_i)        (order within σ_j)
Exec_σ(N)(P)  = concat_j concat_{i∈σ_j} f_N(aligned_i)
Serial(N)(P)  = concat_{i=0..n-1} f_N(aligned_i)
```

二者均是**相同 multiset** `{ f_N(aligned_i) : i }` 的串接，且每个 `f_N(aligned_i)` 在两次运行中相同（它只依赖 `aligned_i`，两次的 record content 相同）。它们仅可能在 blocks 顺序上不同 ⇒ 相同 id-set、每 id 的 per-record `(v,a,o,ℓ)` 相同 ⇒ `Exec_σ(N)(P) ≈ Serial(N)(P)`。（Ids 仍不同：Map/Filter 保留 input ids；Expand ids 由 parent id `p` 和 child index `k` 键控，二者均内容派生，因此重排不会造成冲突。）∎

引理 1 是核心：**对 Map/Filter/Expand，分片/重排运行作为键控集合等于串行运行——包括所有血缘 fields。**

## 5. 顺序规范化算子（Reduce、group_by）

`Reduce(anchor A, descendants D_1..D_s)` 按每个 anchor row 和每个 descendant（`_groups_for`）计算：

```
group(A[q], D) = [ v : (id,v,a,o,ℓ) ∈ D, a(A.name) = A.record_id(q) ]
                 sorted ascending by o(A.name)
```

随后它以**anchor order** 为每个 anchor row `q` 返回一个 output row，对 (anchor value, groups) 应用 reduce UDF。关键是它通过 `a(A.name)`（identity）寻址 descendants，并通过 `o(A.name)`（ordinal）排序它们——**从不依赖 physical position**。

**引理 2（Reduce 对 descendant permutation 不变）。** 若 `D ≈ D'`（键控相等，故为置换）且 anchor `A` 相同，则 `Reduce(A, D) = Reduce(A, D')` **作为有序序列**（不只是 `≈`）。

*证明。* `group(A[q],D)` 由对 `a(·)` 的 filter 和对 `o(·)` 的 sort 定义。membership predicate 和 sort key 都是 `≈` 下保留的 fields 的 per-record functions（相同记录集合及相同 `a,o,v`）。集合 filter 后接在稳定 key 上的全排序，产生只由合格 records 集合及其 keys 决定的序列，独立于 input order。因此每个 `q` 都有 `group(A[q],D)=group(A[q],D')`。Ties：一个 anchor 下的 child ordinals 不同（`Expand` 赋值 `k=0,1,…`），故排序完全且无 tie。输出以 anchor order（相同 `A`）为每个 `q` 一行，故完整 output sequences 相等。∎

（若 descendants 来自 `Filter`，某些 children 缺失；`group` 仅忽略它们。missing-child policy 在两次运行中相同地应用，因为它也只是 surviving keyed set 的函数。）

## 6. Relate

**Key-join（`on=`）。** `_make_key_join_batch` 以 physical order 遍历**第一个 role 的**行，按 key dedup，并为每个 key 产生跨 roles 匹配行的 cross-product。输出 identity 是**content-addressed**：relation row 的 id 是 `op:role_1=pid_1|role_2=pid_2|…`，即其 matched parent record ids（`multigrain.primitives.relate`）的函数，*不是* emission order 的函数。因不同 combos 有不同 parent tuples，ids 唯一（I1）且 permutation-invariant。

**引理 3a（key-join 在置换下为 `≈`）。** 若 role ports 是置换键控相等的（`P^r ≈ P'^r`），则 `Relate_on(P) ≈ Relate_on(P')`。

*证明。* matched combos 集合由每 role 的 key index `key ↦ {matched records}` 决定，它只是 keyed content 的函数（join key 从每个 record 的 value 提取），所以无论 role-port order 如何，都会产生相同*集合*的 parent tuples。对每个 combo，value、`ParentRef`s、merged `(a,o,ℓ)` 与**content-addressed id**均由 matched records 的保留 fields 计算，故相同。相同 id-set、相同 per-record fields ⇒ `≈`。（physical row order 和 surrogate `j` 不再参与 identity，所以无法破坏 `≈`。）∎

**引理 3b（满足置换等变证据契约的 adapter path 为 `≈`）。** 对 `relation_fn` / `relation_adapter` escape hatch，每个输出项提供 `(value, {role: local_index})`，并可选提供确定性的 `stable_key`。框架立即把 invocation-local indexes 解析为 parent record ids，并构造 content-addressed identity：`op:role_1=pid_1|…|role_n=pid_n[|key=stable_key]`。没有不同 stable key 的重复 parent evidence 会被拒绝。

若 adapter 在把 invocation-local indexes 重新绑定到对应 records 后，对输入 rows 的任意置换都产生相同的 `(value, parent tuple, stable_key)` 集合，则解析后的 `ParentRef`s、merged `(a,o,ℓ)`、values 与 ids 均相同。因此 `Relate_adapter(P) ≈ Relate_adapter(P')`。

这一置换等变性是显式的 adapter 契约。把 invocation-local position 当作业务证据，或产生非确定性 stable key 的 adapter，与依赖位置的 UDF 一样，不在本定理范围内。

**推论。** `on=` 与满足契约的 adapter 都使 Relate 完全满足 `≈`（定理第 1 部分可直接适用）。order-sensitive consumer 若要获得 byte-identical sequence order，仍必须先 canonicalize：

> **重排纪律。** sharded node 下游产生的 port 只有在先被规范化（经 `Reduce` 或 unsharded anchor）后才可被*位置性*消费。identity/ordinal-addressed consumers 始终安全。

## 7. 主定理

考虑一个按 topological order `N_1,…,N_K` 排列的 DAG `G`，物理执行中每个 sharded node（`Map`/`Filter`/`Expand`）被任意分配合法 shard plan，WF 在每个 sharded multi-input node 成立，且 `Reduce`/`Relate` 整批运行（单任务，如 MVP）。令 `Phys(port)` 和 `Ser(port)` 是每个 port 上的 physical 和 serial batches。

**定理（重排不变性）。**

1. **（处处键控/血缘相等）** 对 `G` 中每个 port `p`，`Phys(p) ≈ Ser(p)`。特别地，每条 record 的 ancestors、ordinals 和 lineage 与 serial run 相同，独立于处理它的 shard。
2. **（规范输出上的有序相等）** 对每个 `Reduce` 的 output port，若其 anchor 是 graph input 或另一 canonical port，则 `Phys(p) = Ser(p)` 作为 ordered sequences（byte-identical）。

*证明。* 对 topological position 归纳。

*基。* Graph inputs 被完全相同地提供 ⇒ `Phys = Ser`（因此 `≈`）。

*步。* 假设对 `N_k` 的所有 inputs，`Phys(inp) ≈ Ser(inp)`。

- `N_k` 是行独立的（Map/Filter/Expand）：由 IH，其 inputs 与 serial `≈`；`≈` 保留 aligned row multiset，节点逐行应用 `f_N`，所以 `N_k(Phys(inp))` 和 `N_k(Ser(inp))` 共享相同 per-row image multiset。sharding 仅重新分块串接（引理 1）。故 `Phys(out) = Exec_σ(N_k)(Phys(inp)) ≈ N_k(Ser(inp)) = Ser(out)`。（WF 使 positional shard 在全部 ports 上选择匹配 ids。）
- `N_k = Reduce`：整批运行。由 IH descendants 与 serial `≈`，anchor 也与 serial `≈`。若 anchor 是 graph input 或 canonical port，其*顺序*也等于 serial（第 2 部分 / 基），所以由引理 2，`Phys(out) = Ser(out)`（有序）。在所有情形下 `Phys(out) ≈ Ser(out)`（引理 2 给出相等，因而 `≈`）。这为 Reduce outputs 建立第 2 部分。
- `N_k = Relate`：整批运行。由 IH role ports 与 serial `≈`。使用 `on=` 时，引理 3a 直接给出 `Phys(out) ≈ Ser(out)`。使用 adapter escape hatch 时，置换等变 adapter 契约与引理 3b 直接给出 `Phys(out) ≈ Ser(out)`。
- `N_k = Project/Rebatch/Materialize`：对 records 的 identity / pure re-blocking（`take`/`concat`），因此由引理 0 保留 `≈`。

全部 node kinds 保留 `≈`；具有 canonical anchor 的 Reduce 升级为 ordered equality。∎

**推论（血缘引导恢复对重排稳定）。** 一个失败 record 的 `ErrorTrace` 是该 record 的 `(a, ancestor_display, ℓ)` 与 failing op（`map_filter.py::_run_with_bad_index`）的纯函数。由定理第 1 部分，无论 record 落在哪个 shard，这些 fields 都与 serial run 相同。因此 quarantine localization 与 healthy-set 在任意合法 shard plan 下不变。这正是 `test_lineage_under_parallelism.py::test_quarantine_localizes_same_page_under_reordered_parallelism` 所观察的；该定理将其推广到*全部*合法 plans。

**推论（LPT 安全）。** `lpt_shard_planner` 返回合法计划（第 2 节），所以根据该定理，感知工作量的再平衡仅改变 makespan，绝不改变结果或 lineage。因此性能（消除气泡）与正确性解耦：scheduler 可在合法计划族内自由优化。

## 8. 假设、范围和坦诚限制

- *sharded multi-input* nodes 需要 **WF（共序输入）**。它在 MVP pipelines 中成立，并由 `_align_by_identity` *检查*（而非假定），后者在违反时 raise 而不是 mis-join。解除 WF（按 by-id partition 而不是 positional `take` 分片）是未来工作，并将使多输入 sharded nodes 的定理无条件成立。
- MVP 中 **Reduce/Relate 是 whole-batch**。当它们被分片（two-phase reduce、针对 skew 的 salting）时，必须在 *monoid*（associative、commutative-up-to-sort）reduce UDF 下重新建立引理 2；ordinal sort 已提供 canonical order。这恰是 M4 residual（#5 reduce skew）。
- **UDF value-purity（id/position independence）。** 每个 UDF 都是仅关于其 input *values* 的纯函数：绝不读取 `record_id`、`ancestors`、`ordinals`、`lineage`、`display_keys` 或 physical row position。这在 call boundary 执行——operators 传入 `_as_columns(batch) = list(batch.values)`（Map/Filter/Expand）、`anchor.values` + framework-grouped value lists（Reduce）或 `{role: values}`（Relate key-join）；UDF-adjacent hook 唯一会看到的 index 是 relation adapter 的 *invocation-local* index，framework 随即将其转为 `record_id`/`ParentRef`。这正是使 §4 的 per-row image `f_N(aligned_i)` 只依赖 record content、并让引理 1 依赖它的原因：若 UDF 可观察 id 或 position，重排即可改变 output，整个定理将失败。（这也是 [`11-multigrain-primitive-api-ir-review.md`](11-multigrain-primitive-api-ir-review.md) 和 [`12-relation-model-three-tiers.md`](12-relation-model-three-tiers.md) 中的设计红线：“forbidden evidence: global record IDs”。）
- **确定性。** 定理假定 operator UDFs 是其 inputs 的确定函数（`OperatorProperties.deterministic`）。Non-deterministic ops 需要 materialization boundary（已由 `MaterializePolicy` 建模），以使 replay/recovery 有良定义。
- **浮点。** 浮点上的 Reduce 仅在 rounding 意义下 associative；byte equality（第 2 部分）假定 reduce UDF 以 canonical ordinal order 看到 children，而引理 2 保证这一点——因此 sharding 不引入新的 nondeterminism。

## 9. 机器检查证据

假设（row-independence、legal-plan partitioning、WF）和两个结论均由 `test/experimental/multigrain/test_reordering_invariance.py` 运行验证：它在**随机生成的合法 shard plans**（包括完整 shuffles 和 adversarial singleton/one-big splits）下运行每个 motif，并断言：

- 每个 intermediate port 上的 keyed-equality（`≈`，包括全部 lineage fields），以及
- final `Reduce` output 上 byte-identical 的 ordered equality，

相对于 serial baseline——这是一项 property-based check，表明在实际 implementation 中定理保证成立，无需 GPU 也无需 Ray cluster。
