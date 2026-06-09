# RayModule and RuntimeRayModule Lifecycle

Date: 2026-06-09

## Decision

RayOrch keeps two intentionally different module products:

```text
RayModule
  lightweight Ray actor wrapper
  eager and explicit
  useful for assembling experiments and simple pipelines

RuntimeRayModule
  executor-managed production runtime stage
  resource-free during declaration
  fault isolation, lineage, recovery, and scheduling semantics
```

They may share low-level actor, dispatch, and collect machinery, but they do not
share the same lifecycle contract.

## RayModule

`RayModule` is the lightweight foundation.

```python
module = RayModule(MyOp, replicas=2).pre_init(model_path)
result = module(batch)
```

Its current lifecycle is eager:

```text
RayModule(...)            declare actor configuration
.pre_init(...)            create actor replicas immediately
module(...) / remote(...) submit work
user cleanup              release actors
```

This is appropriate for:

- small experiments;
- simple eager pipelines;
- direct Ray actor composition;
- performance prototyping;
- users who want explicit control without runtime metadata.

`RayModule` does not promise:

- row-level fault isolation;
- lineage or quarantine records;
- executor-owned actor recovery;
- adaptive batching;
- persistent runtime state;
- production lifecycle management.

Keeping this layer small is deliberate. Industrial runtime concerns should not
make the basic Ray wrapper harder to understand or use.

## RuntimeRayModule

`RuntimeRayModule` represents a logical production stage, not an already
running actor group.

```python
module = RuntimeRayModule(
    MyOp,
    replicas=4,
    max_inflight=4,
).pre_init(model_path)
```

Construction and `pre_init()` are resource-free:

```text
RuntimeRayModule(...)      declare stage requirements and semantics
.pre_init(...)             store user-op constructor arguments
Pipeline.forward(...)      declare topology
RuntimeDagExecutor(...)    compile, plan, and decide when actors start
executor.run(...)          execute managed microbatches
executor.close()           release actors owned by the executor
```

Calling a Runtime pipeline through `pipeline(...)` or `pipeline.run(...)`
raises an error. A Runtime pipeline must be bound to a Runtime Executor.

`RuntimeRayModule.start()` exists for isolated tests and low-level debugging,
but is not the normal user path.

## Executor Ownership

Runtime Executors own the physical execution lifecycle.

Their responsibilities include:

- compile and validate the DAG;
- bind inferred input/output port specifications;
- inspect stage resource requirements;
- choose actor placement and startup timing;
- create each unique stage's actor replicas;
- schedule microbatches and enforce inflight limits;
- collect healthy output, quarantine records, and lineage deltas;
- close actors that the executor started;
- eventually recover or replace failed actors.

This gives future executors freedom to choose different startup policies
without changing Pipeline or RuntimeRayModule APIs.

## Future Startup Policies

`RuntimeDagExecutor` currently starts all required stages after compilation.
This is only the first policy.

Future executors may choose:

### Eager Start

Start every stage before accepting input.

Useful when:

- model initialization is expensive;
- predictable latency matters;
- enough cluster resources are reserved.

### Lazy Stage Start

Start a stage when its first input becomes runnable.

Useful when:

- some DAG branches may never execute;
- cluster resources are constrained;
- startup cost can overlap upstream work.

### Rolling Start

Start upstream stages first and initialize downstream GPU stages while the
pipeline is warming up.

Useful for hiding model load latency behind preprocessing.

### Elastic Replicas

Start with a minimum replica count and scale according to queue depth,
throughput, memory pressure, or service-level objectives.

### Recovery Start

Replace a failed replica, possibly with a smaller batch policy or different
placement, without reconstructing the logical Pipeline.

### Shared or Pooled Stages

Reuse compatible long-lived actors across executor sessions when model loading
dominates runtime, while keeping ownership and cleanup explicit.

## Ownership Rules

The lifecycle contract should remain:

1. Declaring a Runtime Pipeline must not initialize Ray or create actors.
2. Compiling a Runtime Pipeline must not require actor processes.
3. The Runtime Executor decides when physical resources are acquired.
4. An executor closes only resources it started or explicitly owns.
5. Repeated `RuntimeRayModule.start()` is idempotent.
6. Constructor arguments cannot change while a module is running.
7. Actor recovery policy belongs to Runtime Executor or a future stage
   supervisor, not to Pipeline construction.

## Feature Boundary

| Capability | RayModule | RuntimeRayModule |
| --- | --- | --- |
| Lightweight actor wrapper | Yes | Uses shared machinery |
| Eager direct call | Yes | No |
| Immediate actor creation in `pre_init()` | Yes | No |
| Executor-managed startup | Optional | Required |
| MicroBatch contract | No | Yes |
| Row-level fault isolation | No | Yes |
| Quarantine records | No | Yes |
| Row/path lineage | No | Yes |
| Pipeline overlap | Via `DagExecutor` | Via Runtime Executor |
| Actor recovery policy | User/Ray defaults | Runtime roadmap |
| Adaptive batch/OOM policy | No | Runtime roadmap |
| Persistent lineage sink | No | Runtime roadmap |

## Architectural Consequence

`RuntimeRayModule` should not gradually add resource-management decisions to
its constructor or `pre_init()`. It describes what a stage needs; the Executor
decides how and when that stage becomes physical.

This separation is required for multi-node placement, elastic scaling,
recovery, warm pools, and future production executors.
