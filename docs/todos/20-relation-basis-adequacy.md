# Multigrain Formal Core Semantics and Compilation Soundness

This note defines the supported formal core of the executable Multigrain MVP.
It deliberately avoids a general language-completeness claim. It is separate from
[Reordering Invariance](13-reordering-invariance-theorem.md):

- **formal-core coverage** maps the current authoring contracts to the passive
  graph relation vocabulary;
- **compilation soundness** asks whether tracing preserves the authoring
  program's denotation;
- **reordering invariance** asks whether legal physical execution preserves an
  already compiled graph.

The result is relative to the exact executable authoring fragment: finite,
typed, direct-role, and closed-microbatch. It is neither a claim that the five
relation names are compositionally complete nor Codd relational, graph-query,
or Turing completeness.

## 1. Semantic objects

### 1.1 Ports and records

An identity domain `D` is an immutable namespace. A record on `D` is

```text
r = (id, v, a, o, ℓ, e)
```

with logical key `key_D(r) = (D,id)`:

- `v` is the UDF-visible value;
- `a : Domain ⇀ ID` is functional ancestry;
- `o : Domain ⇀ ℕ` is the ordinal path;
- `ℓ ∈ Name*` is operator lineage;
- `e ⊆ Role × Domain × ID` is **direct** role-parent evidence.

A port is a finite typed sequence:

```text
P = (D, grain, [r_0, …, r_n])
```

Keys are unique within one live port/admission scope. No process-global or
cross-microbatch identity is assumed.

### 1.2 Source semantic DAG

A source semantic DAG is a finite acyclic graph. Its contracts mirror the
current authoring API rather than an unconstrained relation language:

```text
MapPreserve(input_0)                         1:1
FilterSubset(input_i)                        0:1
Children(parent, finite_ordinal)             1:N
Aggregate(anchor, selectors, incomplete)     N:1
Related(ordered_roles, matcher)              M:N
```

The contracts are independent of Python classes and of `ExecutionGraph`.
They describe identity and dependency behavior:

- **MapPreserve** emits exactly one value per input-0 key and reuses input 0's
  domain/key. Other Map inputs must be identity-aligned and contribute metadata,
  but they cannot be selected as a different identity source.
- **FilterSubset** emits one key-preserving subset for each positional input,
  in that input's source order.
- **Children** emits a finite ordered cohort for each parent key in a derived
  domain.
- **Aggregate** emits one anchor-keyed row per admitted anchor, with descendant
  fibers selected by direct ancestry or direct role evidence.
- **Related** emits rows in a derived domain, each justified by one finite,
  ordered role-parent tuple.

## 2. Well-formed fragment `F_direct`

A semantic DAG is in `F_direct` iff all of the following hold:

1. It is finite and acyclic.
2. Every port is a finite sequence with unique `(Domain,id)` keys.
3. Row-local transforms are deterministic and value-pure.
4. Aligned row-local inputs have one identity domain and one key order. Map
   identity always comes from input 0; Filter/FilterByMask preserve each
   declared positional source.
5. Every child cohort is finite and has stable ordinals.
6. Every Reduce selector is a function to an anchor in the same closed
   microbatch:

   ```text
   ByAncestor     π(r) = a(r)(anchor.domain)
   ByRole(role)   π(r) = unique direct edge in e(r)
   ```

   For `ByAncestor`, every runtime descendant must carry exactly one functional
   anchor-domain ID and that parent must be present in the anchor batch.
   `verify_graph` checks structural reachability only; shared parent-ID
   consistency after `Related` is a runtime/data contract.
7. `ByRole` is direct. `Children` does not inherit the parent's role edges.
8. Relate uses an equi-join or a deterministic permutation-equivariant adapter.
9. Relation hashes are treated as collision resistant for the finite execution;
   duplicate IDs in one output batch are rejected.
10. Every operation/recovery combination is supported by the selected backend;
    this is checked by that backend before execution, not by `verify_graph`.

The fragment is closed under DAG composition only when these conditions remain
true. Merely chaining names from the five-cardinality vocabulary does not imply
membership.

## 3. Target passive graph

The target basis is:

```text
Preserve-total    ↦ SameAs
Preserve-partial  ↦ SubsetOf
Children          ↦ ChildrenOf
Aggregate         ↦ AggregateOf
Related           ↦ RelatedFrom
```

The operation side records invocation semantics:

```text
MapOp
FilterOp
ExpandOp
ReduceOp(selectors)
RelateOp(matcher)
FilterByMaskOp
```

`OutputSpec.relation` is the identity/dependency fact; `NodeSpec.operation` is
the execution recipe. Neither alone is the complete node contract.

## 4. Local representation lemmas

### Lemma A — preserved identity

Every deterministic total row-local Map transform in this core is represented
by `MapOp` with `SameAs(input_0)`. Every deterministic key-preserving Filter
predicate is represented by `FilterOp`/`FilterByMaskOp` with one
`SubsetOf(input_i)` for each declared output source.

The target runtime preserves all metadata columns for retained keys and appends
the operation lineage.

### Lemma B — finite dependent children

For every finite family `C(p) = [c_0,…,c_k]`, `ExpandOp` with
`ChildrenOf(parent)` represents

```text
Σ_(p ∈ parent) C(p)
```

using the derived child domain, parent ancestry, and ordinal `i`.

### Lemma C — functional fibers

For a closed anchor port `A` and descendants with a declared parent function
`π`, `ReduceOp(selectors)` plus `AggregateOf(A)` represents one deterministic
fold over each canonically ordered fiber

```text
π⁻¹(a) = [d | π(d)=a.id].
```

### Lemma D — finite role relation

For a finite relation

```text
J ⊆ P_1 × … × P_k
```

whose evidence is produced by a key join or a permutation-equivariant adapter,
`RelateOp` plus `RelatedFrom(roles)` represents one output row per
`(value, ordered parent tuple, stable_key)` item. `ParentRef` preserves the
direct multi-parent evidence that cannot live in functional ancestry.

## 5. Formal-core coverage proposition

**Proposition.** For every well-formed semantic DAG `S ∈ F_direct` whose nodes
use the authoring contracts in §1.2, there exists a passive `ExecutionGraph G`
with the corresponding operation/relation pairs. When the runtime/data
obligations in §2 hold, its output ports have the same values, record keys,
ancestry, ordinals, lineage, and direct role evidence as `S`, up to deterministic
naming of derived domains, node outputs, and their occurrences in lineage.

**Proof sketch.**

Topologically order `S`.

- Inputs lower to `GraphInputSpec`.
- Assume every predecessor port has a target `PortRef` and equivalent runtime
  denotation.
- The current node belongs to one of the five contracts by definition of
  `F_direct`. Apply Lemma A, B, C, or D to select its typed operation and
  per-output relation.
- `verify_graph` checks the structural subset: available refs, grain
  preservation, structural ancestry reachability, direct-role availability,
  matcher shape, and operation/relation pairing.
- Runtime checks enforce identity alignment, closed parent batches, output
  domain/grain and backend recovery support. Determinism, value-purity,
  shared-parent consistency and adapter equivariance remain explicit user/data
  obligations.
- Under all three layers of obligations, the new target ports preserve the
  node denotation.

Finite induction constructs all nodes and graph outputs. ∎

This is a **coverage** result for the current authoring core, not a completeness
or minimality result. A sufficiently
powerful whole-batch Relate adapter could simulate other computations, but it
would not expose their locality, cardinality, canonical order, or recovery
contract to the runtime.

## 6. Authoring compilation soundness

The authoring fragment contains:

```text
Pipeline.forward
Map / Filter / Expand / Reduce / Relate
group_by(anchor, descendants)
via(descendant, role)
Select
```

Tracing executes `forward` over `TracePort` tokens only. Each primitive emits
the corresponding operation/relation pair. `Select` is a macro:

```text
Select_f(inputs)
  ≜ Map_f(inputs) producing (mask, annotations)
    then FilterByMask(inputs, mask, annotations)
```

**Theorem (Compilation Soundness).** If an authoring program traces
successfully, its UDFs/data satisfy the semantic contracts of `F_direct`, and
the selected backend supports its recovery policies, then execution of the
structurally verified `ExecutionGraph` returned by `Pipeline.compile()` is
keyed-equivalent to the eager primitive denotation. For Select, eager and
traced execution use the same Map annotation semantics before the same mask
projection.

**Proof sketch.** Structural induction over calls performed by `forward`.
`TracePort.ref` names the inductive predecessor. Primitive lowering follows the
local representation lemmas. `GraphTracer.add_node` records exactly those refs,
relations, operations, worker policies, and recovery policies; `build` then
checks the structural graph. Backend/runtime checks and the semantic obligations
listed in §2 complete the theorem premises. Select follows by macro expansion. ∎

The theorem does not prove arbitrary Python control flow. `forward` must
actually return traceable ports, output arity is explicit, and data-dependent
branching over runtime values is outside tracing.

## 7. Counterexamples and non-goals

The following are not in `F_direct`:

- `Relate → Expand → ByRole Reduce`: Expand intentionally erases direct role
  edges; transitive role paths need a richer provenance graph.
- cross-microbatch join, aggregation, Global, Window, watermark, or late data;
- recursion, feedback, fixpoint, and dynamic graph construction;
- stateful/nondeterministic UDFs and external side effects;
- mixed-output Expand identity forests (`out.same/out.children`);
- dataset-scope Partition, union, difference, sort, or arbitrary θ-join
  semantics not represented by the current contracts;
- distributed/two-phase Reduce before its monoid contract is formalized.

Some excluded computations can be manually reconstructed with extra keys and
Relate nodes. That does not make their missing optimizer-visible contracts part
of this fragment.

## 8. Executable evidence

The proof obligations map to tests:

- eager/compiled Select parity, including multi-input lineage and recovery;
- random legal partitions with full keyed metadata comparison at every port;
- Ray graph outputs exposing intermediate ports to check Canon order;
- independent role permutations for a key-based adapter;
- nested Expand and Reduce selector examples;
- negative checks for role paths, parent closure, mixed domains, positional
  adapters, and invalid shard partitions.

Property tests are executable evidence for the implementation, not a substitute
for the theorem's premises. In particular, arbitrary Python purity and adapter
equivariance remain user contracts.

## 9. Relationship to future extensions

An extension changes the formal-core coverage proposition only if it adds a
semantic contract:

- `out.same/out.children` adds an invocation-local identity forest;
- Partition adds fixed disjoint routing;
- Global/Window adds dataset/time scope and completion;
- role paths add transitive provenance selectors;
- two-phase Reduce adds a monoid and distributed completion law.

Each extension must add: syntax, denotation, well-formedness rules, lowering,
runtime materialization, verifier checks, and preservation evidence. It must not
silently broaden the theorem by reusing an existing name.
