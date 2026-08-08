# V3.6 RayModule Symbolic Typing and Signature Ergonomics

> **Document role: deferred design exploration, not an accepted V3.6 contract or an
> implementation bug.** See the
> [`V3.6 documentation map`](../multigrain_v3_6_documentation_map.md) for the main learning path.

Status: **design exploration; deferred; no implementation decision**
Recorded: 2026-08-09

## 1. Question

Can V3.6 reuse the original `RayModule[InitP, RunP, R]` generic pattern so that
`pre_init(...)` and symbolic `module(...)` calls preserve the wrapped UDF's IDE
completion, keyword names and static types?

The desired experience is reasonable, but the original contract cannot be
copied literally because the two APIs execute at different stages.

## 2. The semantic mismatch

The original RayModule receives and returns real Python values:

```text
pre_init : InitP -> configured actor
call     : RunP  -> R
```

V3.6 traces a symbolic Pipeline. The UDF still runs on batch columns inside a
Worker, but its authoring call receives and returns graph handles:

```text
UDF runtime : list[Image], threshold: float -> list[Text]
authoring   : Port,        threshold: Port  -> Port
```

Therefore a literal

```python
def __call__(*args: RunP.args, **kwargs: RunP.kwargs) -> R: ...
```

would make two false claims: a `Port` is not a `list[Image]`, and the symbolic
return is not `list[Text]`. Python `ParamSpec` can forward one parameter list,
but it cannot map every parameter type through a constructor such as
`T -> Port[T]` while preserving positional/keyword names.

## 3. How established Python DSLs handle it

### PyTorch

FX feeds internal `Proxy` values through ordinary Python code, while FakeTensor
is a `torch.Tensor` subclass with no real data. Both approaches work because
most user values share one Tensor protocol; the public function remains
`Tensor -> Tensor`, and tracing is hidden behind `symbolic_trace`,
`torch.compile` or `torch.export`.

This does not generalize directly to RayOrch. A Port may denote strings, PDFs,
images, dicts, audio chunks or arbitrary user objects, so it cannot honestly be
a subtype of every possible business type.

References:

- <https://docs.pytorch.org/docs/stable/fx.html>
- <https://docs.pytorch.org/docs/2.9/torch.compiler_fake_tensor.html>
- <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html>

### Triton

Triton preserves the original callable as an ergonomic typing facade:
`jit(fn: T) -> JITFunction[T]`, and `kernel[grid]` is typed as `T`. Internally,
`JITCallable` stores `inspect.signature(fn)`, builds a binder equivalent to
`Signature.bind() + apply_defaults()`, reads the Python source and compiles its
AST using compiler-owned `tl.tensor`/`constexpr` values.

This deliberately gives the compiler, not Python generics, final authority over
DSL legality. It is the closer precedent for V3.6.

References:

- <https://triton-lang.org/main/python-api/generated/triton.jit.html>
- <https://github.com/triton-lang/triton/blob/main/python/triton/runtime/jit.py>
- <https://github.com/triton-lang/triton/blob/main/python/triton/compiler/code_generator.py>

## 4. Candidate designs

### A. Copy the original three generics

Forward `InitP`, `RunP` and `R` literally.

- Best apparent completion.
- Incorrect symbolic argument and return types.
- Makes downstream Port operations appear to consume real business values.

**Reject unless the V3.6 staging model changes.**

### B. Dual overload as an IDE facade

Expose one `RunP` overload for parameter-name completion and one real symbolic
overload accepting `Port | OptionalInput` and returning symbolic outputs.

- Pragmatic and similar in spirit to Triton.
- The completion overload is not a fully sound semantic contract.
- The symbolic fallback cannot statically reject a misspelled keyword.
- Editors may show two overloads.

Keep as an experiment, not an assumed final design.

### C. Honest symbolic API plus signature binding

- Fully forward constructor `InitP` through `pre_init(...)`.
- Keep `RayModule.__call__` explicitly symbolic.
- Preserve `Self` through `returns()` and `ray_options()`.
- Use the wrapped UDF's existing `inspect.Signature` as the sole authority for
  argument count, names, positional/keyword-only slots and defaults.
- Reject invalid binding during `Pipeline.compile()`, before any Worker starts.
- Keep the symbolic return as `Port | tuple[Port, ...]`; do not expose runtime
  `R` as the authoring result.

This is the cleanest semantic contract, but it cannot promise full static
completion for `module(...)` parameters.

### D. Generated stubs or a type-checker plugin

Generate a per-UDF symbolic signature that maps every business parameter to a
typed Port.

- Potentially complete and sound.
- Requires code generation, `.pyi` lifecycle or type-checker-specific support.
- Too much machinery for the current benefit.

**Defer unless user evidence shows typing is a major adoption blocker.**

## 5. Provisional direction

Prototype C first. Separately test whether a narrowly documented Triton-style
facade from B materially improves Pylance/Pyright completion without confusing
users. Do not introduce the facade based only on a type-level proof; evaluate
the actual editor UI.

All typing machinery must remain in the authoring layer. It must not enter
`LogicalProgram`, `RuntimePlan`, `MicrobatchEngine`, Worker DTOs or Ray actors.
The Python callable remains the single source of truth; no second parameter
schema should be stored and maintained by hand.

## 6. Required acceptance cases

Before implementation, a standalone type fixture should demonstrate:

1. class `__init__` positional, keyword-only and default parameters complete in
   `pre_init(...)`;
2. class `.run`, callable instances and plain functions have one documented
   target-resolution rule;
3. `module(port, keyword=port)` is accepted without casting;
4. wrong arity, missing keyword-only arguments and unknown keywords fail during
   compile-time signature binding;
5. one logical output remains `Port`, multiple outputs remain a tuple of Ports;
6. `.returns()` and `.ray_options()` preserve the configured module type;
7. no Generic/ParamSpec object crosses the Worker/Ray serialization boundary;
8. current Ray-free, real-Ray and core Pyright gates remain unchanged.

## 7. Non-goals

- Making `Port` pretend to be every possible Python business type.
- Inferring arbitrary output arity from `R` annotations.
- Adding a general compiler-plugin or pass framework.
- Moving UDF execution-signature interpretation into Runtime or Worker.
- Weakening the rule that Pipeline authoring manipulates Ports, not payloads.
