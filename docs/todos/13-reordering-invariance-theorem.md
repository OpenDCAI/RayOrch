# Reordering-Invariance of Multigrain Execution (M1 formalization)

This note turns the empirical fact behind
`test/experimental/multigrain/test_lineage_under_parallelism.py` into a formal
statement with proofs: **any legal shard plan (including work-aware LPT
rebalancing) preserves keyed results and lineage; Ray canonical merge, or a
Reduce with an ordered-identical anchor, additionally restores serial visible
order.** This is the correctness backbone that lets us reorder rows for
performance (bubble elimination) without changing logical semantics.

Everything below is grounded in the actual data model so the formalism is not
hand-wavy:

| formal object | code |
|---|---|
| static port address | `GraphInputRef | NodeOutputRef` (`PortRef`) |
| static output relation | `SameAs`, `SubsetOf`, `ChildrenOf`, `AggregateOf`, `RelatedFrom` |
| graph node | `NodeSpec(operation=..., outputs=(OutputSpec(...), ...))` |
| mandatory graph check | `verify_graph` |
| identity namespace | immutable `IdentityDomain` |
| record fields | `PortBatch` parallel arrays (`values`, `record_ids`, `ancestors`, `ancestor_display`, `ordinals`, `lineage`, `relations`) |
| `take` / `concat` | `PortBatch.take(indices)` / `multigrain.data.concat(batches)` |
| shard-plan check | `validate_shard_plan(partitions, row_count)` |
| Expand lineage | `PortBatchBuilder.expanded` |
| Reduce regroup | `Reduce._groups_for` |
| Relate key-join | `Relate._make_key_join_batch` |

The passive `ExecutionGraph` supplies a static typed relation vocabulary, while
the proof operates on runtime `PortBatch` records. The correspondence is:

- `SameAs(source)` preserves the source keyed records and identity domain;
- `SubsetOf(source)` keeps a keyed subset of the same domain;
- `ChildrenOf(parent)` creates a fresh child domain whose ids are parent
  identity plus ordinal; `OutputSpec.grain` is only the type label;
- `AggregateOf(anchor, incomplete)` returns to the anchor domain after grouping
  descendants by a declared parent function (`ByAncestor` or `ByRole`);
- `RelatedFrom(roles)` creates a fresh relation domain whose identity is the
  ordered role-parent evidence tuple.

`verify_graph` checks that these relations use valid refs and grains and match the
typed operation. It does not replace the runtime evidence: Reduce and the theorem
continue to rely on `record_ids`, `ancestors`, `ordinals`, and role edges `e`
carried by `PortBatch`.

## 0. Fragment claim and non-goals

This note formalizes a **multi-grain identity algebra fragment**, not classical
Codd relational algebra.

**Supported fragment.** Finite DAGs over closed microbatches whose cardinality
changes are drawn from:

```text
1:1   SameAs          (Map)
0:1   SubsetOf        (Filter / FilterByMask)
1:N   ChildrenOf      (Expand)
N:1   AggregateOf     (Reduce via ByAncestor or ByRole)
M:N   RelatedFrom     (Relate)
```

**Motif coverage, not a completeness theorem.** The five relations name the
currently executable local motifs. Their composition is restricted by the
well-formedness rules: in particular `ByRole` consumes direct `RelatedFrom`
evidence, while `ChildrenOf` starts a child cohort with no inherited role edges.
The independent [Formal Core note](20-relation-basis-adequacy.md)
defines the supported authoring fragment and its lowering claim. The theorem below is
only about execution reordering of an already verified graph.

**Explicit non-goals.** The fragment does **not** claim:

- classical RA completeness (union, set difference, arbitrary θ-join);
- general graph queries (path, transitive closure, recursive CTE);
- cross-microbatch Global/Window state or watermark completeness;
- transitive role-path navigation such as `Relate → Expand → ByRole`;
- invocation-local Expand mixed-output forests (`mg.out.same/children`), which
  remain a deferred composition extension of the same vocabulary.

## 1. Data model

**Domains.** Let `Domain` be the set of immutable identity namespaces
(`IdentityDomain`). A batch lives in one current domain `D ∈ Domain`. Grain and
current port are *labels*; they are not keys of ancestry.

**Records.** A *record* is a tuple `r = (id, v, a, o, ℓ, e)` on batch identity
domain `D`. Its logical key is `key_D(r) = (D,id)`:

- `id ∈ ID_D` — an identity string unique within the live port/admission scope
  of `D` (`record_id`), not a process-global identifier;
- `v` — the payload value;
- `a : Domain ⇀ ID` — the *ancestor map* (`ancestors[i]`), a **partial function**
  keyed by `IdentityDomain`, never by grain or current port name;
- `o : Domain ⇀ ℕ` — the *ordinal map* (`ordinals[i]`), child index under an
  ancestor;
- `ℓ ∈ Name*` — the *lineage path* (`lineage[i]`), the sequence of operators the
  record passed through;
- `e ⊆ Role × Domain × ID` — direct role-parent evidence (`relations[i]`).
  Relate copies only shared, unambiguous ancestors into `a`; conflicting
  same-domain parents remain losslessly represented in `e`.

**Invariant I0 (functional ancestry).** For every record, `a` is a partial
function: each domain maps to at most one ancestor id. Therefore `a` always
describes a forest. Multi-parent structure that would violate functionality is
stored only in `e`, never overwritten into `a`.

(`display_keys` / `ancestor_display` are display projections of the same data and
carried verbatim by `take`/`concat`, so we omit them from the proofs; they follow
the same argument.)

**Batches.** A *batch* `B = [r_0, …, r_{n-1}]` is a finite **sequence** of
records on one domain `D`. `B[i]` is the i-th record; `|B| = n`; `ids(B)` the
sequence of ids. Each runtime batch also has a grain name. Execution enforces
`PortBatch.name == OutputSpec.grain`; therefore physical and serial batches at
the same graph port have the same name in addition to the per-record equality
proved below.

**Port invariant I1 (identity uniqueness).** Within every live `PortBatch`,
record keys `(D,id)` are pairwise distinct. Since one batch has one `D`, runtime
checks pairwise-distinct ids. Sources emit `name:i`; a later admission may reuse
the same strings, so no cross-microbatch uniqueness is claimed. `Expand` emits
`op:parent_id:child_index`; `Map` preserves ids and `Filter`/`FilterByMask` keep
a subset. `Relate` emits
the stable hash of the canonical ordered tuple
`((role_1, domain_1, pid_1), …, stable_key)` and rejects duplicate parent
evidence unless a distinct stable key is provided. `PortBatch.__post_init__`
rejects duplicate IDs at every runtime boundary. Relation-hash uniqueness relies
on the explicit collision-resistance assumption in §8.)

**Keyed view.** Define `⟦B⟧ : (Domain × ID) ⇀ Record` by
`⟦B⟧(D,id) = r` for the unique `r ∈ B` with that key (well-defined by I1).
Two batches are **keyed-equal**,
written `B ≈ B'`, iff `⟦B⟧ = ⟦B'⟧` as sets of full records: same ids and
identical `(v, a, o, ℓ, e)` per id. `B` is a **permutation** of `B'` iff
`B ≈ B'` and `|B| = |B'|`.

`≈` ignores physical order but is strict on every per-record field — including
role edges `e` — so proving `≈` at a port already proves *lineage and relation
equality* at that port.

### 1.1 Domain homomorphism (IR → runtime)

Static ports are addressed by `PortRef`. Runtime batches carry `IdentityDomain`.
The homomorphism `δ` is induced by output relations:

| output relation | runtime domain |
|---|---|
| graph input | fresh or named domain chosen at `source(...)` admission |
| `SameAs(s)` / `SubsetOf(s)` | inherit `δ(s)` |
| `ChildrenOf(p)` | fresh domain derived from `(op, δ(p))` |
| `AggregateOf(anchor, ·)` | return to `δ(anchor)` |
| `RelatedFrom(roles)` | fresh domain derived from `(op, δ(role ports)…)` |

Aligned multi-input Map/Filter further require all inputs to share one domain
(`δ(p_i)` equal) before identity alignment. Grain equality alone is insufficient.

### 1.2 Parent functions for Reduce

A Reduce descendant selector declares a deterministic parent function

```text
π : DescendantRecord → AnchorId
```

Currently supported:

```text
ByAncestor     π(r) = a(r)(δ(anchor))
ByRole(ρ)      π(r) = the unique id s.t. (ρ, δ(anchor), id) ∈ e(r)
```

Both require the image of `π` to land inside the current closed anchor
microbatch; otherwise execution raises a closure violation. Ambiguous
multi-parent ancestry without `ByRole` is rejected (no silent overwrite).
`ByRole` is deliberately direct: `ChildrenOf` does not copy its parent's role
edges, so role paths through Expand and general graph traversal are outside this
fragment.

## 2. Physical execution model

**take.** For an index list `σ = [σ_1,…,σ_k]` of distinct indices into `B`,
`take(B, σ) = [B[σ_1], …, B[σ_k]]` — a subsequence with each record copied
intact (`PortBatch.take` copies `value`, `ancestors`, `ordinals`, `lineage`,
and `relations`/`e` element-wise).

**concat.** `concat(B_1,…,B_m) = B_1 ⧺ … ⧺ B_m` — sequence concatenation, each
record intact (`multigrain.data.concat` extends the parallel arrays element-wise
and deduplicates inherited `ErrorTrace`s).

**Shard plan.** A *shard plan* for an `n`-row port is `σ = (σ_1,…,σ_m)`, a tuple
of index lists. It is **legal** iff `{σ_1,…,σ_m}` is a **set partition** of
`{0,…,n-1}`: the `σ_j` are pairwise disjoint and `⋃_j σ_j = {0,…,n-1}`.

> Ray calls `validate_shard_plan` on planner output. It rejects out-of-range or
> duplicate indexes and any plan that does not cover every row exactly once.
> Legality is therefore checked at runtime, not merely assumed from contiguous
> or LPT planner implementations.

**Sharded node execution** (mirrors the Ray node runner): for a node `N` with inputs
`(P^0,…,P^t)` (port 0 is the base) and legal plan `σ` over `|P^0|`:

```
RawExec_σ(N)(P^0,…,P^t) = concat_j ( N( take(P^0,σ_j), …, take(P^t,σ_j) ) )
Exec_σ(N)(P) = Canon_N,P(RawExec_σ(N)(P))
```

merged per output port. `Canon` sorts `SameAs`/`SubsetOf` by source logical
position and `ChildrenOf` by `(parent logical position, child ordinal,
record_id)`. `Serial(N) = N(P^0,…,P^t)` is the whole-batch run
(`m = 1`, `σ_1 = [0..n-1]`).

**Well-formedness WF (co-ordered inputs).** For a *sharded* multi-input node, all
input ports present records in the same id-order, i.e. `ids(P^0)=…=ids(P^t)`, so
positional `take(P^r, σ_j)` selects the same id-set on every port. (Single-input
sharded nodes satisfy WF vacuously. The current row-partitionable operations are
`Map`/`Filter`/`FilterByMask`/`Expand`; Ray checks exact identity order before
planning, and `_align_by_identity` inside the primitive remains a defensive
check rather than the first line of protection.)

## 3. Reassembly lemma

**Lemma 0 (partition reassembly).** For any batch `B` and legal plan `σ`,
`concat_j take(B, σ_j)` is a permutation of `B`.

*Proof.* By legality each index `i ∈ {0..n-1}` occurs in exactly one `σ_j` and
exactly once there, so it is selected by exactly one `take(B,σ_j)`, exactly once,
with `B[i]` copied intact. `concat` gathers all selected records, so the result
contains each `B[i]` exactly once ⇒ same id-set, same per-record fields, same
length ⇒ a permutation of `B`. ∎

## 4. Row-independent operators (Map, Filter, FilterByMask, Expand)

**Definition (row-independent).** `N` is *row-independent* if there is a per-row
function `f_N` s.t. for aligned input rows, the output is the ordered
concatenation of the per-row images and `f_N` depends only on the record
content, not on the row’s position or on other rows:

```
N(P^0,…,P^t) = concat_i  f_N( aligned_i )
```

where `aligned_i` is the identity-aligned tuple of the i-th base row with its
same-id partners on the other ports, and `f_N(aligned_i)` is a batch of length
1 (`Map`), 0 or 1 (`Filter`), or `k_i ≥ 0` (`Expand`).

- **Map** `f = ` apply UDF to the aligned row, keep `id/a/o/e` (merging
  ancestry and role evidence across aligned inputs), append op to `ℓ`.
  Length 1. (`multigrain.primitives.map_filter`.)
- **Filter** `f = ` `[row]` if mask true else `[]`; kept rows keep
  `id/a/o/ℓ/e`. Length 0/1. (`multigrain.primitives.map_filter`.)
- **FilterByMask** is Select's internal canonical filtering operation. The mask
  is a same-identity Map output; it applies the same 0/1 decision to every
  aligned source/annotation and therefore has the same row-local argument as
  Filter. The mask port itself is consumed, not emitted.
- **Expand** `f = ` for parent row with id `p` on domain `D_p`, emit children on
  a fresh domain with
  `id = op:p:k`, `a' = a ∪ {D_p ↦ p}`,
  `o' = o ∪ {D_p ↦ k}`, `e' = ∅`,
  `ℓ' = ℓ⧺[op]` for `k = 0..k_p-1`. Length `k_p`. Depends only on the parent
  record’s own value (the UDF sees the parent value and returns its group).
  (`multigrain.primitives.output::PortBatchBuilder.expanded`.)

This is an explicit **row-local UDF assumption**: each `f_N` is a deterministic
function of the aligned row values only—no `i`, no batch-size dependence, no
cross-row mutable state, and no dependence on execution order. The call boundary
hides framework metadata, but Python cannot prove that a user object has no
external state; programs violating this contract are outside the theorem.

**Lemma 1 (sharding commutes with row-independent nodes).** For a row-independent
`N` under WF and any legal plan `σ`: `RawExec_σ(N)(P) ≈ Serial(N)(P)` and,
after canonical merge, `Exec_σ(N)(P) = Serial(N)(P)` as an ordered sequence.

*Proof.* By Lemma 0, `concat_j take(P^0,σ_j)` is a permutation of `P^0`; under WF
the same index sets select the same ids on every input port, so shard `j`
contains exactly the aligned rows `{aligned_i : i ∈ σ_j}`. Since `N` applies `f_N`
independently per aligned row,

```
N(take(P,σ_j))   = concat_{i∈σ_j} f_N(aligned_i)
RawExec_σ(N)(P)  = concat_j concat_{i∈σ_j} f_N(aligned_i)
Serial(N)(P)     = concat_{i=0..n-1} f_N(aligned_i)
```

The raw and serial forms concatenate the **same multiset**
`{ f_N(aligned_i) : i }`, each
`f_N(aligned_i)` identical in both runs (it depends only on `aligned_i`, which is
the same record content in both). They may differ only in the order of the blocks
⇒ same id-set, same per-record `(v,a,o,ℓ,e)` per id ⇒
`RawExec_σ(N)(P) ≈ Serial(N)(P)`.
(Ids stay distinct: Map/Filter preserve input ids, Expand ids are keyed by the
parent id `p` and child index `k`, both content-derived, so no collision arises
from reordering.) Finally `Canon` orders preserved rows by serial source
position and expanded rows by parent position plus child ordinal, exactly the
order in `Serial`; hence ordered equality. ∎

Lemma 1 is the crux: **for Map/Filter/FilterByMask/Expand, a
sharded/reordered run equals the serial run as a keyed collection — including
all lineage fields and role edges.**

## 5. Order-canonicalizing operators (Reduce, group_by)

`Reduce(anchor A, descendants D_1..D_s)` computes, per anchor row and per
descendant, using the declared parent function `π` (§1.2):

```
group_π(A[q], D) =
  [ v : r=(id,v,a,o,ℓ,e) ∈ D, π(r) = A.record_id(q) ]
  sorted by the domain-qualified ordinal path, then record_id
```

where

```
ByAncestor     π(r) = a(r)(A.domain)
ByRole(ρ)      π(r) = unique id with (ρ, A.domain, id) ∈ e(r)
```

then returns one output row per anchor row `q`, in **anchor order**, applying the
reduce UDF to the (anchor value, groups). Both selectors require the selected
parent to be present in the closed anchor microbatch and order by
domain-qualified ordinals — **never by physical position**.

**Lemma 2 (Reduce is invariant to descendant permutation).** If `D ≈ D'` (keyed
equal; hence a permutation), the anchor `A` is identical, and the same selector
`π` is used, then `Reduce_π(A, D) = Reduce_π(A, D')` **as ordered sequences**
(not merely `≈`).

*Proof.* Membership in `group_π(A[q],·)` is decided by `π(r)`, a per-record
function of either `a` (`ByAncestor`) or `e` (`ByRole`). The sort key is a
per-record function of `o` (and `id` as tie-break). Both are preserved under
`≈`, which now includes `(v,a,o,ℓ,e)`. A set-filter followed by a total sort on
a stable key yields a sequence determined solely by the *set* of qualifying
records and their keys — independent of input order. Hence
`group_π(A[q],D)=group_π(A[q],D')` for every `q`. Ties: child ordinals under one
anchor are distinct for Expand-derived descendants (`k=0,1,…`); Relate-derived
descendants use `record_id` as a total tie-break. The output is one row per `q`
in anchor order (same `A`), so the whole output sequences are equal. ∎

(If descendants come from a `Filter`, some children are absent; `group` simply
omits them. The missing-child policy is applied identically in both runs because
it too is a function of the surviving keyed set.)

## 6. Relate

**Key-join (`on=`).** `_make_key_join_batch` iterates the **first role's** rows in
physical order, dedups by key, and for each key emits the cross-product of the
matched rows across roles. Output identity is **content-addressed**: a relation
row's id is a stable hash of the canonical tuple
`((role_1, domain_1, pid_1), (role_2, domain_2, pid_2), …)`, i.e. a function of
its matched parent evidence (`multigrain.primitives.relate`), *not* of emission order. Since distinct combos
have distinct parent tuples, ids are permutation-invariant; treating their
SHA-256 digests as distinct uses the collision-resistance assumption in §8.

**Lemma 3a (key-join is `≈` under permutation).** If the role ports are permuted
keyed-equal (`P^r ≈ P'^r`), then `Relate_on(P) ≈ Relate_on(P')`.

*Proof.* The set of matched combos is determined by the per-role key index
`key ↦ {matched records}`, a function of the keyed content only (the join key is
extracted from each record’s value), so the same *set* of parent tuples is
produced regardless of role-port order. For each combo the value, `ParentRef`s,
role edges `e`, unambiguous shared `(a,o)`, merged `ℓ`, and the
**content-addressed id** are all computed from the matched records’ preserved
fields, hence identical. Same id-set, same per-record fields ⇒ `≈`. (The physical row order and the surrogate `j` no longer appear in
identity, so they cannot break `≈`.) ∎

**Lemma 3b (adapter path is `≈` under permutation-equivariant evidence).** For
the dotted `relation_adapter` path, each emitted item supplies
`(value, {role: local_index})`, optionally followed by a deterministic
`stable_key`. The role mapping must contain **exactly** the declared roles.
The framework resolves indexes in declared role order—not mapping insertion
order—to typed `(role, domain, parent id)` evidence and constructs the stable
hash identity from that tuple plus optional `stable_key`. Duplicate parent evidence
without distinct stable keys is rejected.

If, after rebasing invocation-local indexes to their records, the adapter emits
the same set of `(value, parent tuple, stable_key)` for every permutation of its
input rows, then the resolved `ParentRef`s / role edges `e`, unambiguous shared
`(a,o)`, merged `ℓ`, values, and ids are identical. Therefore
`Relate_adapter(P) ≈ Relate_adapter(P')`.

This permutation-equivariance condition is an explicit adapter contract. An
adapter that uses invocation-local position as business evidence, or emits a
non-deterministic stable key, is outside the theorem just as a position-sensitive
UDF is.

The eager-only `relation_fn` path uses the same runtime evidence shape, so this
lemma also describes its eager semantics. It is not a compiled graph matcher;
compiled Relate requires `KeyJoinSpec` or dotted `RelationAdapterSpec`.

**Consequence.** Both `on=` and a contract-conforming adapter make Relate fully
`≈` under role-port permutation. Relate is **not** itself an order-canonicalizer:
if its role inputs are only keyed-equal but physically reordered, emission order
may change. Ordered equality at a Relate port requires already-ordered role
inputs (as provided by upstream `Canon` in the Ray model).

## 7. Main theorem

Consider a DAG `G` in topological order `N_1,…,N_K`, with a legal shard plan on
each row-partitionable node (`Map`/`Filter`/`FilterByMask`/`Expand`), WF at each
sharded multi-input node, and `Reduce`/`Relate` run whole-batch. Write

```text
RawPhys(p)  = result after take/concat sharding without Canon
Phys(p)     = Ray model: Canon applied after every sharded row-independent node
Ser(p)      = serial whole-batch execution
```

**Theorem (Reordering Invariance).**

1. **(Keyed / lineage equality everywhere)** For every port `p`,
   `RawPhys(p) ≈ Ser(p)`. Every record’s `(v,a,o,ℓ,e)` matches the serial run
   as a keyed collection, independent of the legal shard plan.
2. **(Ordered equality after Canon / at Reduce)**
   - For every row-independent port under the Ray model,
     `Phys(p) = Ser(p)` as ordered sequences (Lemma 1).
   - For every `Reduce` output with fixed selector `π` **and an
     ordered-identical anchor** (`A_phys = A_ser` as sequences),
     `RawPhys(p) = Phys(p) = Ser(p)` even if descendants are only `≈`
     (Lemma 2). If the anchor itself is merely `≈` and physically permuted,
     Reduce still gives keyed equality, but emits in the permuted anchor order.
   - For a `Relate` port, ordered equality holds when role inputs are already
     ordered-equal; in general Relate only guarantees part 1 (`≈`).

*Proof.* Induction on topological position.

*Base.* Graph inputs are supplied identically ⇒ equality (hence `≈`).

*Step (part 1).* Assume inputs are `≈` serial.

- Row-independent `N_k`: Lemma 1 gives `RawExec_σ ≈ Serial`.
- `Reduce`: Lemma 2 with descendant `≈` gives keyed-equal outputs; with an
  ordered-identical anchor it also gives ordered equality. MVP pipelines use
  graph-input anchors (or Canon'd SameAs ports), which satisfy the stronger
  premise.
- `Relate`: Lemmas 3a/3b give `≈` from keyed-equal role ports.

*Step (part 2).* Under the Ray model every sharded row-independent node applies
`Canon`, so Lemma 1 upgrades those ports to ordered equality. Reduce upgrades by
Lemma 2 when the anchor is ordered-identical (true for graph-input anchors, and
for SameAs anchors that passed Canon). Relate upgrades only when IH gives
ordered role inputs. ∎

**Corollary (lineage-guided recovery is reorder-stable).** Under deterministic
failure classification and error rendering, an `ErrorTrace` is a function of
the record key, ancestry and display projections, lineage, selected parent,
failing operation, recovery action, and error text. These inputs are stable
under the theorem's keyed execution (display metadata is copied verbatim), so
quarantine localization and the resulting trace are reorder-stable. Observed by
`test_lineage_under_parallelism.py::test_quarantine_localizes_same_page_under_reordered_parallelism`.

**Corollary (LPT is safe for correctness).** `lpt_shard_planner` returns a legal
plan, so by part 1 it cannot change keyed results, lineage, or role edges; by
part 2 (Ray Canon) it also cannot change user-visible `list[obj]` order at
Map/Filter/Expand/Reduce ports. This does **not** claim lineage chooses the
plan; LPT may use an external `weight_of(value)`.

## 8. Assumptions, scope, and honest limits

- **WF (co-ordered inputs)** is required for *sharded multi-input* nodes. It holds
  in the MVP pipelines and is checked by identity-domain equality plus
  `_align_by_identity`,
  which raises on violation instead of mis-joining. Lifting WF (shard by a
  by-id partition instead of positional `take`) is future work and would make the
  theorem unconditional for multi-input sharded nodes.
- **Reduce/Relate are whole-batch within `CLOSED_MICROBATCH`** in the MVP.
  Every selected ancestry/role parent must be present in that microbatch;
  cross-microbatch Global/Window state is outside this theorem. When they are sharded (two-phase
  reduce, salting for skew), Lemma 2 must be re-established under a *monoid*
  (associative, commutative-up-to-sort) reduce UDF; the ordinal sort already
  supplies the canonical order. This is exactly the M4 residual (#5 reduce skew).
- **Row-local UDF value-purity.** Map/Filter/Expand UDF output for a row is a
  deterministic function of that aligned row's *values* only. It must not depend
  on batch size, sibling rows, mutable cross-call state, invocation order,
  framework IDs, or physical position. The call boundary hides `record_id`,
  `ancestors`, `ordinals`, `lineage`, and display fields, but semantic purity is
  a user contract. Reduce may inspect its complete, canonically ordered value
  groups; Relate key-join receives values selected by content-derived keys.
- **Adapter permutation-equivariance.** After invocation-local indexes are
  rebound to their records, a relation adapter must emit the same keyed set of
  `(value, ordered role-parent tuple, deterministic stable_key)` for every input
  permutation. Position-sensitive adapters are outside the theorem.
- **Content-addressed identity.** Relation IDs use SHA-256 over canonical role
  evidence. The proof assumes collision resistance over the finite records in
  one execution; the runtime detects any duplicate IDs that nevertheless appear
  in one output batch.
- **Determinism.** All covered UDFs and adapters are deterministic functions of
  their allowed value/evidence inputs. The current minimal graph has no
  materialization operation that could make a nondeterministic operator
  replay-safe.
- **Floating point.** Reduce over floats is only associative up to rounding; byte
  equality assumes the reduce UDF sees children in the canonical ordinal
  order, which Lemma 2 guarantees — so there is *no* new nondeterminism from
  sharding.
- **Fragment boundary.** This theorem proves reordering invariance, not language
  completeness. Classical RA operators, graph recursion, cross-batch
  Global/Window, mixed-output Expand, and transitive role paths are outside its
  executable fragment. The separate Formal Core note states the restricted
  authoring and compilation-soundness claim.

## 9. Machine-checked evidence

What is actually checked today:

| claim | evidence |
|---|---|
| part 1 `≈` everywhere under random legal plans | `test_reordering_invariance.py` (local take→concat **without** Canon; compares `(v,a,o,ℓ,e)`) |
| Reduce ordered equality | same file, final Reduce port |
| part 2 ordered equality after Canon on Map/Filter | Ray `test_reordered_shards_canonize_exposed_intermediate_outputs` |
| part 2 ordered equality after Canon on Expand | Ray `test_reordered_shards_canonize_expand_output` |
| full deterministic ErrorTrace equality | Ray `test_quarantine_localizes_same_page_under_reordered_parallelism` |
| ByRole / closure / domain motifs | Semantic Hardening + relate key-join suites |

Honest gap: the local property harness proves part 1 strongly; part 2’s
“ordered everywhere under Canon” is proved for the Ray `Exec=Canon(RawExec)`
model and spot-checked, but not yet property-tested on every intermediate port
in the local harness (which deliberately omits Canon to isolate `≈`).
