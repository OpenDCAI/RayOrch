# Lazy benchmark plugins

RayOrch keeps scheduling primitives independent from benchmark dependencies.
The built-in layout is:

```text
rayorch/
  multigrain/                 # promoted compiler and execution runtime
  benchmark/
    plugin.py                 # dependency-free metadata contract
    registry.py               # lazy name -> plugin module lookup
    runtime.py                # runtime_env loading and deterministic digest
    mineru/
      udfs.py                 # business kernels usable by RayModule
      pipeline.py             # released RayOrch graph and regression runner
      plugin.py               # lightweight registration metadata
      runtime_env.json        # one shared environment for this UDF group
      job.py                  # wheel staging and Ray Jobs submission
```

`rayorch.benchmark.registry` imports only a plugin's lightweight `plugin.py`.
Heavy packages stay inside UDF methods or runner entrypoints, so importing
RayOrch does not require every benchmark environment to be installed.

## Add a UDF group

Create a lightweight module containing a `BenchmarkPlugin`, register that
module path with `register_plugin(name, module)`, and provide a versioned
`runtime_env.json`. Built-in groups may also be listed statically in
`registry.py`. Keep these boundaries:

- `udfs.py`: framework-independent business kernels and data contracts.
- `pipeline.py`: the production graph and its output/performance regression.
- `plugin.py`: names, runner module, environment resource, and dependency hints;
  it must not import workload dependencies.

Comparison adapters and ablation runners belong in experiment branches or
separate benchmark artifacts, not in the release package.

One UDF group has one environment by default. This is the useful isolation unit:
groups can pin conflicting stacks, while all actors in a benchmark reuse the
same Ray runtime environment and cache key.

## Ray Job packaging

The MinerU launcher stages only package sources and build metadata in a temporary
directory, builds RayOrch and Flash-MinerU wheels, and supplies those wheels via
Ray `py_modules`. It never uploads PDFs, models, experiment logs, or the original
work directories. Pip dependencies come from the group's `runtime_env.json`.

Use `--dry-run` to build both wheels and inspect the resolved environment without
submitting a job. Extra one-off dependencies may be appended with repeated
`--extra-pip` flags; stable dependencies should be recorded in the group file so
the digest and Ray cache remain reproducible.
