# Reordering-Invariance of Multigrain Execution (M1 formalization)

This note turns the empirical fact behind
`test/experimental/multigrain/test_lineage_under_parallelism.py` into a formal
statement with proofs: **any legal shard plan (including work-aware LPT
rebalancing) produces the same result and the same lineage as the serial
baseline.** This is the correctness backbone that lets us reorder rows for
performance (bubble elimination) without changing semantics.

Everything below is grounded in the actual data model so the formalism is not
hand-wavy:

| formal object | code |
|---|---|
| record fields | `PortBatch` parallel arrays (`values`, `record_ids`, `ancestors`, `ancestor_display`, `ordinals`, `lineage`) in `core.py` |
| `take` | `PortBatch.take(indices)` |
| `concat` | `core.concat(batches)` |
| shard/merge | `MultigrainRayExecutor._run_node` (`ray_executor.py`, lines ~133–165) |
| shard plan | `shard_planner(node, inputs, replicas) -> list[list[int]]`; `lpt_shard_planner`, `_contiguous_ranges` |
| Expand lineage | `Expand._make_outputs` (`expand_reduce.py`) |
| Reduce regroup | `Reduce._groups_for` (`expand_reduce.py`) |
| Relate key-join | `Relate._make_key_join_batch` (`relate.py`) |

## 1. Data model

**Records.** A *record* is a tuple `r = (id, v, a, o, ℓ)`:

- `id ∈ ID` — a globally unique identity string (`record_id`);
- `v` — the payload value;
- `a : Name ⇀ ID` — the *ancestor map* (`ancestors[i]`), partial map from a
  producer port name to the ancestor record id on that port;
- `o : Name ⇀ ℕ` — the *ordinal map* (`ordinals[i]`), child index under an
  ancestor;
- `ℓ ∈ Name*` — the *lineage path* (`lineage[i]`), the sequence of operators the
  record passed through.

(`display_keys` / `ancestor_display` are display projections of the same data and
carried verbatim by `take`/`concat`, so we omit them from the proofs; they follow
the same argument.)

**Batches.** A *batch* `B = [r_0, …, r_{n-1}]` is a finite **sequence** of
records. `B[i]` is the i-th record; `|B| = n`; `ids(B)` the sequence of ids.

**Port invariant I1 (identity uniqueness).** At every port produced by the
system, ids are pairwise distinct. (Sources emit `name:i`; `Expand` emits
`op:parent_id:k`; `Relate` emits `op:j`; `Map`/`Filter` preserve ids. Distinctness
is maintained by construction.)

**Keyed view.** Define `⟦B⟧ : ID ⇀ Record` by `⟦B⟧(id) = r` for the unique
`r ∈ B` with that id (well-defined by I1). Two batches are **keyed-equal**,
written `B ≈ B'`, iff `⟦B⟧ = ⟦B'⟧` (equal as sets of full records — same ids,
and identical `(v,a,o,ℓ)` per id). `B` is a **permutation** of `B'` iff `B ≈ B'`
and `|B| = |B'|` (equivalently, `B'` reorders `B`).

`≈` ignores physical order but is strict on every per-record field, so proving
`≈` at a port already proves *lineage equality* at that port.

## 2. Physical execution model

**take.** For an index list `σ = [σ_1,…,σ_k]` of distinct indices into `B`,
`take(B, σ) = [B[σ_1], …, B[σ_k]]` — a subsequence with each record copied
intact (`PortBatch.take` copies `value`, `ancestors`, `ordinals`, `lineage`
element-wise).

**concat.** `concat(B_1,…,B_m) = B_1 ⧺ … ⧺ B_m` — sequence concatenation, each
record intact (`core.concat` extends the parallel arrays element-wise).

**Shard plan.** A *shard plan* for an `n`-row port is `σ = (σ_1,…,σ_m)`, a tuple
of index lists. It is **legal** iff `{σ_1,…,σ_m}` is a **set partition** of
`{0,…,n-1}`: the `σ_j` are pairwise disjoint and `⋃_j σ_j = {0,…,n-1}`.

> `_contiguous_ranges` yields a legal plan trivially. `lpt_shard_planner` assigns
> every index `i` to exactly one bin (`bins[target].append(i)` inside a loop over
> all `i`), so it too yields a legal plan. Legality is the *only* property of the
> planner the proofs use — an optimizer may reorder/rebalance arbitrarily.

**Sharded node execution** (mirrors `_run_node`): for a node `N` with inputs
`(P^0,…,P^t)` (port 0 is the base) and legal plan `σ` over `|P^0|`:

```
Exec_σ(N)(P^0,…,P^t) = concat_j ( N( take(P^0,σ_j), …, take(P^t,σ_j) ) )
```

merged per output port. `Serial(N) = N(P^0,…,P^t)` is the whole-batch run
(`m = 1`, `σ_1 = [0..n-1]`).

**Well-formedness WF (co-ordered inputs).** For a *sharded* multi-input node, all
input ports present records in the same id-order, i.e. `ids(P^0)=…=ids(P^t)`, so
positional `take(P^r, σ_j)` selects the same id-set on every port. (Single-input
sharded nodes satisfy WF vacuously. In the MVP only `Map`/`Filter`/`Expand` are
sharded; `_align_by_identity` inside the node then re-pairs by id within the
shard and *rejects* a violated WF with a readable error rather than silently
mis-joining.)

## 3. Reassembly lemma

**Lemma 0 (partition reassembly).** For any batch `B` and legal plan `σ`,
`concat_j take(B, σ_j)` is a permutation of `B`.

*Proof.* By legality each index `i ∈ {0..n-1}` occurs in exactly one `σ_j` and
exactly once there, so it is selected by exactly one `take(B,σ_j)`, exactly once,
with `B[i]` copied intact. `concat` gathers all selected records, so the result
contains each `B[i]` exactly once ⇒ same id-set, same per-record fields, same
length ⇒ a permutation of `B`. ∎

## 4. Row-independent operators (Map, Filter, Expand)

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

- **Map** `f = ` apply UDF to the aligned row, keep id/ancestors/ordinals, append
  op to `ℓ`. Length 1. (`map_filter.py`.)
- **Filter** `f = ` `[row]` if mask true else `[]`; kept rows keep identity.
  Length 0/1. (`map_filter.py`.)
- **Expand** `f = ` for parent row with id `p`, emit children
  `id = op:p:k`, `a' = a ∪ {portname ↦ p}`, `o' = o ∪ {portname ↦ k}`,
  `ℓ' = ℓ⧺[op]` for `k = 0..k_p-1`. Length `k_p`. Depends only on the parent
  record’s own value (the UDF sees the parent value and returns its group).
  (`expand_reduce.py::_make_outputs`.)

Each `f_N` is a pure function of the record content only — no `i`, no cross-row
state — which is exactly what makes reordering safe.

**Lemma 1 (sharding commutes with row-independent nodes).** For a row-independent
`N` under WF and any legal plan `σ`:  `Exec_σ(N)(P) ≈ Serial(N)(P)`.

*Proof.* By Lemma 0, `concat_j take(P^0,σ_j)` is a permutation of `P^0`; under WF
the same index sets select the same ids on every input port, so shard `j`
contains exactly the aligned rows `{aligned_i : i ∈ σ_j}`. Since `N` applies `f_N`
independently per aligned row,

```
N(take(P,σ_j)) = concat_{i∈σ_j} f_N(aligned_i)        (order within σ_j)
Exec_σ(N)(P)  = concat_j concat_{i∈σ_j} f_N(aligned_i)
Serial(N)(P)  = concat_{i=0..n-1} f_N(aligned_i)
```

Both are concatenations of the **same multiset** `{ f_N(aligned_i) : i }`, each
`f_N(aligned_i)` identical in both runs (it depends only on `aligned_i`, which is
the same record content in both). They may differ only in the order of the blocks
⇒ same id-set, same per-record `(v,a,o,ℓ)` per id ⇒ `Exec_σ(N)(P) ≈ Serial(N)(P)`.
(Ids stay distinct: Map/Filter preserve input ids, Expand ids are keyed by the
parent id `p` and child index `k`, both content-derived, so no collision arises
from reordering.) ∎

Lemma 1 is the crux: **for Map/Filter/Expand, a sharded/reordered run equals the
serial run as a keyed collection — including all lineage fields.**

## 5. Order-canonicalizing operators (Reduce, group_by)

`Reduce(anchor A, descendants D_1..D_s)` computes, per anchor row and per
descendant (`_groups_for`):

```
group(A[q], D) = [ v : (id,v,a,o,ℓ) ∈ D, a(A.name) = A.record_id(q) ]
                 sorted ascending by o(A.name)
```

then returns one output row per anchor row `q`, in **anchor order**, applying the
reduce UDF to the (anchor value, groups). Crucially it addresses descendants by
`a(A.name)` (identity) and orders them by `o(A.name)` (ordinal) — **never by
physical position**.

**Lemma 2 (Reduce is invariant to descendant permutation).** If `D ≈ D'` (keyed
equal; hence a permutation) and the anchor `A` is identical, then
`Reduce(A, D) = Reduce(A, D')` **as ordered sequences** (not merely `≈`).

*Proof.* `group(A[q],D)` is defined by a filter on `a(·)` plus a sort on `o(·)`.
Both the membership predicate and the sort key are per-record functions of fields
that are preserved under `≈` (same set of records with same `a,o,v`). A set-filter
followed by a total sort on a stable key yields a sequence determined solely by
the *set* of qualifying records and their keys — independent of input order.
Hence `group(A[q],D)=group(A[q],D')` for every `q`. Ties: child ordinals under one
anchor are distinct (`Expand` assigns `k=0,1,…`), so the sort is total and
tie-free. The output is one row per `q` in anchor order (same `A`), so the whole
output sequences are equal. ∎

(If descendants come from a `Filter`, some children are absent; `group` simply
omits them. The missing-child policy is applied identically in both runs because
it too is a function of the surviving keyed set.)

## 6. Relate

**Key-join (`on=`).** `_make_key_join_batch` iterates the **first role's** rows in
physical order, dedups by key, and for each key emits the cross-product of the
matched rows across roles. Output identity is **content-addressed**: a relation
row's id is `op:role_1=pid_1|role_2=pid_2|…`, i.e. a function of its matched
parent record ids (`relate.py`), *not* of emission order. Since distinct combos
have distinct parent tuples, ids are unique (I1) and permutation-invariant.

**Lemma 3a (key-join is `≈` under permutation).** If the role ports are permuted
keyed-equal (`P^r ≈ P'^r`), then `Relate_on(P) ≈ Relate_on(P')`.

*Proof.* The set of matched combos is determined by the per-role key index
`key ↦ {matched records}`, a function of the keyed content only (the join key is
extracted from each record’s value), so the same *set* of parent tuples is
produced regardless of role-port order. For each combo the value, `ParentRef`s,
merged `(a,o,ℓ)`, and the **content-addressed id** are all computed from the
matched records’ preserved fields, hence identical. Same id-set, same per-record
fields ⇒ `≈`. (The physical row order and the surrogate `j` no longer appear in
identity, so they cannot break `≈`.) ∎

**Lemma 3b (adapter path is `≈` up to reindexing).** For the `relation_fn` /
`relation_adapter` escape hatch, output ids are `op:j` (emission order) and an
adapter may emit several rows with the same parent set, so identity is not
content-addressable in general. Under permutation the *relation content* (values,
`ParentRef`s, merged lineage) is still identical as a multiset; only the surrogate
ids/order differ — keyed-equal **up to reindexing**.

**Consequence.** With `on=`, Relate is fully `≈` (Theorem part 1 applies as-is).
With the adapter escape hatch, a Relate output must be consumed by an
order-canonicalizing operator (a `Reduce`, or any identity/ordinal-addressed
consumer) to obtain byte-identical final results — the intended pattern
(`link → group_by(pdf) → Reduce`). We state this as a *typing rule* rather than
hide it:

> **Reordering discipline.** A port produced downstream of a sharded node may be
> consumed *positionally* only if it is first canonicalized (via `Reduce`, or an
> unsharded anchor). Identity/ordinal-addressed consumers are always safe.

## 7. Main theorem

Consider a DAG `G` in topological order `N_1,…,N_K`, executed physically with an
arbitrary assignment of a legal shard plan to each sharded node (`Map`/`Filter`/
`Expand`), WF holding at each sharded multi-input node, and `Reduce`/`Relate` run
whole (single task, as in the MVP). Let `Phys(port)` and `Ser(port)` be the
physical and serial batches at each port.

**Theorem (Reordering Invariance).**

1. **(Keyed/lineage equality everywhere)** For every port `p` in `G`,
   `Phys(p) ≈ Ser(p)`. In particular every record’s ancestors, ordinals, and
   lineage are identical to the serial run, independent of which shard processed
   it.
2. **(Ordered equality at canonical outputs)** For every port that is the output
   of a `Reduce` whose anchor is a graph input or another canonical port,
   `Phys(p) = Ser(p)` as ordered sequences (byte-identical).

*Proof.* Induction on topological position.

*Base.* Graph inputs are supplied identically ⇒ `Phys = Ser` (hence `≈`).

*Step.* Assume `Phys(inp) ≈ Ser(inp)` for all inputs of `N_k`.

- `N_k` row-independent (Map/Filter/Expand): its inputs are `≈` to serial by IH;
  `≈` preserves the aligned row multiset, and the node applies `f_N` per row, so
  `N_k(Phys(inp))` and `N_k(Ser(inp))` share the same per-row image multiset.
  Sharding only re-blocks the concatenation (Lemma 1). Hence
  `Phys(out) = Exec_σ(N_k)(Phys(inp)) ≈ N_k(Ser(inp)) = Ser(out)`. (WF lets the
  positional shard select matching ids on all ports.)
- `N_k = Reduce`: run whole. By IH descendants are `≈` serial and the anchor is
  `≈` serial. If the anchor is a graph input or a canonical port, its *order*
  equals serial too (part 2 / base), so by Lemma 2 `Phys(out) = Ser(out)`
  (ordered). In all cases `Phys(out) ≈ Ser(out)` (Lemma 2 gives equality, a
  fortiori `≈`). Establishes part 2 for Reduce outputs.
- `N_k = Relate`: run whole. By IH role ports are `≈` serial. With `on=`,
  Lemma 3a gives `Phys(out) ≈ Ser(out)` directly. With the adapter escape hatch,
  Lemma 3b gives equality up to relation-row reindexing, which the reordering
  discipline (below) canonicalizes at the next `Reduce`.
- `N_k = Project/Rebatch/Materialize`: identity on records / pure re-blocking
  (`take`/`concat`), so `≈` is preserved by Lemma 0.

All node kinds preserve `≈`; Reduce with a canonical anchor upgrades to ordered
equality. ∎

**Corollary (lineage-guided recovery is reorder-stable).** The `ErrorTrace` for a
failed record is a pure function of that record’s `(a, ancestor_display, ℓ)` and
the failing op (`map_filter.py::_run_with_bad_index`). By Theorem part 1 these
fields are identical to the serial run regardless of the shard the record landed
in. Hence quarantine localization and the healthy-set are invariant under any
legal shard plan. This is precisely what
`test_lineage_under_parallelism.py::test_quarantine_localizes_same_page_under_reordered_parallelism`
observes; the theorem generalizes it to *all* legal plans.

**Corollary (LPT is safe).** `lpt_shard_planner` returns a legal plan (Section 2),
so by the Theorem work-aware rebalancing changes only makespan, never results or
lineage. Performance (bubble elimination) and correctness are thus decoupled: the
scheduler is free to optimize within the legal-plan family.

## 8. Assumptions, scope, and honest limits

- **WF (co-ordered inputs)** is required for *sharded multi-input* nodes. It holds
  in the MVP pipelines and is *checked* (not assumed) by `_align_by_identity`,
  which raises on violation instead of mis-joining. Lifting WF (shard by a
  by-id partition instead of positional `take`) is future work and would make the
  theorem unconditional for multi-input sharded nodes.
- **Reduce/Relate are whole-batch** in the MVP. When they are sharded (two-phase
  reduce, salting for skew), Lemma 2 must be re-established under a *monoid*
  (associative, commutative-up-to-sort) reduce UDF; the ordinal sort already
  supplies the canonical order. This is exactly the M4 residual (#5 reduce skew).
- **UDF value-purity (id/position independence).** Every UDF is a pure function of
  its input *values* only: it never reads `record_id`, `ancestors`, `ordinals`,
  `lineage`, `display_keys`, nor the physical row position. This is enforced at the
  call boundary — operators pass `_as_columns(batch) = list(batch.values)` (Map/
  Filter/Expand), `anchor.values` + framework-grouped value lists (Reduce), or
  `{role: values}` (Relate key-join); the only index a UDF-adjacent hook ever sees
  is the relation adapter's *invocation-local* index, which the framework
  immediately translates to a `record_id`/`ParentRef`. This is precisely what
  makes the per-row image `f_N(aligned_i)` in §4 depend only on record content,
  which Lemma 1 relies on: if a UDF could observe id or position, reordering could
  change its output and the whole theorem would fail. (It is also the design red
  line in [`11-multigrain-primitive-api-ir-review.md`](11-multigrain-primitive-api-ir-review.md)
  and [`12-relation-model-three-tiers.md`](12-relation-model-three-tiers.md):
  "forbidden evidence: global record IDs".)
- **Determinism.** The theorem assumes operator UDFs are deterministic functions
  of their inputs (`OperatorProperties.deterministic`). Non-deterministic ops need
  a materialization boundary (already modeled by `MaterializePolicy`) to keep
  replay/recovery well-defined.
- **Floating point.** Reduce over floats is only associative up to rounding; byte
  equality (part 2) assumes the reduce UDF sees children in the canonical ordinal
  order, which Lemma 2 guarantees — so there is *no* new nondeterminism from
  sharding.

## 9. Machine-checked evidence

The hypotheses (row-independence, legal-plan partitioning, WF) and the two
conclusions are exercised by
`test/experimental/multigrain/test_reordering_invariance.py`: it runs each motif
under **randomly generated legal shard plans** (including full shuffles and
adversarial singleton/one-big splits) and asserts

- keyed-equality (`≈`, incl. all lineage fields) at every intermediate port, and
- byte-identical ordered equality at the final `Reduce` output,

against the serial baseline — a property-based check that the theorem’s
guarantees hold for the actual implementation, with no GPU and no Ray cluster.
