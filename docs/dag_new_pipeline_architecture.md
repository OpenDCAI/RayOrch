# `rayorch/dag_new_pipeline.py` 架构文档

> 对应文件：`rayorch/dag_new_pipeline.py`（796 行）
> 更新日期：2026-04-03

---

## 文件结构总览

```
Module docstring + imports                 (1-36)

# ── Data Types ──                         (42-68)
PipeRef                    # 符号引用：node + index
NodeSpec                   # 不可变节点定义
CompiledGraph              # 不可变 DAG 拓扑

# ── Graph Tracing ──                      (75-193)
_expect_ref()              # 验证 PipeRef
_validate_ref()            # 验证引用合法性（node 存在、slot 范围）
_GraphTracer               # 录制 forward() 调用到 tape
_TraceProxy                # forward 执行时替身 RayModule

# ── Executors ──                          (200-530)
Executor(ABC)              # 策略接口：execute() + run()
├─ SequentialExecutor      # 串行：逐 batch 逐 node 同步调用
└─ DagExecutor             # 并行：ray.wait 重叠调度
   └─ _Scheduler           # 事件循环实现
      _PendingResult       # 已提交的远程调用句柄
      _NodeStatus          # 每 batch 每 node 的状态机
      _InflightCall        # Ray ref → batch/node 映射
_validate_output()         # 校验 node 返回值契约
_validate_columns()        # 校验输入列一致性
_materialize_outputs()     # 从 ctx 提取最终结果

# ── Pipeline (user API) ──               (537-667)
Pipeline                   # 用户继承的基类
DagPipeline = Pipeline     # 向后兼容别名

# ── Inline Tests ──                       (674-796)
if __name__ == "__main__": ...
```
```
┌─────────────────────────────────────────────────────┐
│ Data Types (42-68)                                  │
│  PipeRef(node, index)    — 符号引用                   │
│  NodeSpec                — 不可变节点定义               │
│  CompiledGraph           — 不可变 DAG 拓扑             │
├─────────────────────────────────────────────────────┤
│ Graph Tracing (75-193)                              │
│  _expect_ref()           — 验证 PipeRef               │
│  _validate_ref()         — 验证引用合法性               │
│  _GraphTracer            — 录制 forward() 到 tape      │
│  _TraceProxy             — forward 执行时替身 RayModule │
├─────────────────────────────────────────────────────┤
│ Executors (200-530)                                 │
│  Executor(ABC)           — 策略接口: execute + run     │
│  ├─ SequentialExecutor   — 串行: 逐batch逐node同步调用  │
│  └─ DagExecutor          — 并行: ray.wait 重叠调度     │
│     └─ _Scheduler        — 事件循环实现               │
│        _PendingResult    │                          │
│        _NodeStatus       ├─ 调度器内部状态             │
│        _InflightCall     │                          │
│  _validate_output()      — 校验 node 返回值           │
│  _validate_columns()     — 校验输入列                 │
│  _materialize_outputs()  — 从 ctx 提取最终结果         │
├─────────────────────────────────────────────────────┤
│ Pipeline (537-667)                                  │
│  Pipeline                — 用户继承的基类              │
│    __init__()            — 无参数                     │
│    forward()             — 用户覆写，PipeRef 布线       │
│    compile()             — tracer 追踪 forward        │
│    __call__()            — 委托 run()                 │
│    run()                 — 唯一执行入口                │
│  DagPipeline = Pipeline  — 向后兼容别名               │
└─────────────────────────────────────────────────────┘
```
---

## 核心抽象

### PipeRef

```python
@dataclass(frozen=True)
class PipeRef:
    node: str
    index: int = 0
```

符号引用，双重身份：
- **用户侧**：`forward()` 中的数据流句柄（类比 `torch.Tensor`）
- **内部侧**：`NodeSpec` 中的边引用

所有上下文访问统一为 `ctx[ref.node][ref.index]`，无 isinstance 分支。

### NodeSpec

```python
@dataclass(frozen=True)
class NodeSpec:
    name: str
    module: RayModule
    args: Tuple[PipeRef, ...]
    kw_args: Dict[str, PipeRef]
    max_inflight: int = 1
    num_outputs: int = 1
```

不可变节点定义，`max_inflight` 和 `num_outputs` 从 `RayModule` 读取。

### CompiledGraph

```python
@dataclass
class CompiledGraph:
    nodes: Dict[str, NodeSpec]
    topo_order: Tuple[str, ...]
    deps: Dict[str, Tuple[str, ...]]
    consumers: Dict[str, Tuple[str, ...]]
    graph_outputs: Tuple[PipeRef, ...]
    input_keys: Tuple[str, ...]
```

compile() 的产物。Executor 只读取此结构，不访问 Pipeline 实例。

---

## 交互模式

### 数据流

```
用户定义                    编译时                       运行时
─────────                  ─────                      ─────
Pipeline 子类         ───→  compile()              ───→  run()
 ├ RayModule attrs          ├ 替换为 _TraceProxy         ├ auto-compile
 └ forward(PipeRef)         ├ 执行 forward()             ├ _resolve_inputs
                            ├ _GraphTracer 录制 tape      ├ Executor.execute()
                            ├ build() → CompiledGraph    │  ├ Sequential: module()
                            └ 还原 RayModule              │  └ DAG: _Scheduler
                                                         └ List[结果]
```

### 3 种等价调用方式

```python
pipe(batches)                                     # 串行（默认）
pipe.run(batches, executor=DagExecutor())          # 指定 executor
DagExecutor().run(pipe, batches)                   # executor 主导
```

三者最终都走 `Pipeline.run()` 这一个入口：

```python
def run(self, *inputs, executor=None, **named_inputs) -> List[Any]:
    if self._compiled is None:
        self.compile()
    columns = self._resolve_inputs(inputs, named_inputs)
    if executor is None:
        executor = SequentialExecutor()
    return executor.execute(self._compiled, columns)
```

### Pipeline 与 Executor 的关系

```
Pipeline                         Executor
  │                                │
  ├ 拥有 RayModule 属性             ├ 只看 CompiledGraph
  ├ 拥有 forward() 定义             ├ 不访问 Pipeline 实例
  ├ 拥有 compile() 和 _compiled     ├ execute(graph, columns) → List
  ├ 拥有 _resolve_inputs()         │
  └ run() 调用 executor.execute()  └ run() 委托 pipeline.run()
```

Pipeline 负责"是什么"（图定义），Executor 负责"怎么跑"（调度策略）。

---

## Executor 策略

### SequentialExecutor

- 逐 batch、按 `topo_order`、同步 `module(...)` 调用
- 零并发，适合调试、profiling、正确性验证
- 每个 node 走 `RayModule.__call__()` → `gather(remote())`

### DagExecutor

- `ray.wait` 驱动的事件循环
- `max_batches_inflight` 控制同时执行的 batch 数
- 每个 node 的 `max_inflight` 控制同名 node 的并发提交数
- 调度流程：admit → mark_ready → dispatch → submit → ray.wait → on_complete → release_upstream
- Greedy drain：block for 1 ref，然后 timeout=0 捞取所有已完成 ref
- 上游释放：当某 node 的所有 consumer 都完成时，从 ctx 中删除该 node 的数据

### 上下文存储

所有 node 输出统一存为 tuple：
- 单输出 → `(value,)`
- 多输出 → `(v0, v1, ...)`

访问统一为 `ctx[ref.node][ref.index]`。

---

## 共享 Helper

| 函数 | 用途 | 调用者 |
|---|---|---|
| `_validate_output()` | 校验 node 返回值与 `num_outputs` 匹配 | Sequential, _Scheduler |
| `_validate_columns()` | 校验输入列 key 一致、长度相等 | Sequential, _Scheduler |
| `_materialize_outputs()` | 从 ctx 提取 graph_outputs 对应的值 | Sequential, _Scheduler |

---

## 编译过程（compile）

1. 扫描 Pipeline 实例的所有 `RayModule` 属性
2. 临时替换为 `_TraceProxy`
3. 反射 `forward()` 签名 → 生成输入 `PipeRef`
4. 执行 `forward(*pos_refs, **kw_refs)`
5. `_TraceProxy.__call__` 拦截每次调用 → `_GraphTracer.add_node()` 录入 tape
6. `_GraphTracer.build()` 从 tape 构建 `CompiledGraph`：
   - 拓扑序 = tape 顺序（forward 的声明顺序）
   - deps / consumers 通过遍历每个 node 的 args/kw_args 中的 PipeRef 构建
   - `_validate_ref()` 校验每条边的合法性
7. 还原 `RayModule` 属性

---

## 设计决策记录

| 决策 | 原因 |
|---|---|
| `PipeRef` 替代 `Source = Union[str, Tuple[str, int]]` | 消除 isinstance 分支，统一属性访问 |
| Context 统一存 tuple | `ctx[ref.node][ref.index]` 无分支 |
| Pipeline 和 Executor 拆分 | 定义与调度解耦，策略模式，支持未来扩展 |
| `__call__` 默认串行 | 调试友好，DagExecutor 显式启用并行 |
| `max_batches_inflight` 在 Executor 上 | 调度参数不属于图定义 |
| `Executor.run()` 委托 `pipeline.run()` | 避免 compile/resolve 逻辑重复 |
| 未使用 `__init_subclass__` 签名传播 | 静态类型检查器不认 `__signature__`，投入产出不匹配 |

---

## 未来扩展点

- `PriorityExecutor` — 优先级调度（roadmap 7.1）
- `AdaptiveBatchExecutor` — 动态 batch 大小（roadmap 7.2）
- `ProfileExecutor` — 包装其他 executor，添加 NVTX/timeline 埋点
- `DryRunExecutor` — 验证图结构，不提交 Ray 任务

---

## 流式数据消费模型（迭代器驱动 + 自驱动 source）

> 为 hydp-dataflow 的「有状态流式 reader」承接而加。分两步演进，均**向后兼容**（list 输入行为逐字节不变）。

### 第一步：迭代器驱动（不预知 batch 总数）

`_Scheduler` 不再要求预知 batch 总数。`execute(graph, columns)` 的每个 `columns[key]` 只需是**任意 iterable**
（list / 生成器 / 迭代器）。调度循环从每个 `input_key` 的迭代器**同步取一批**（所有 key 都有下一个才 admit 新
batch；**任一耗尽 = 源枯竭**），按 `max_batches_inflight` 边拉边 admit，终止条件是「源枯竭 **且** in-flight 清空」。
`_ctx`/`_status` 按 batch 懒建、完成即释放（TB 级/未知长度源不 OOM、不预扫）。list 输入走 `iter(list)`，老行为不变。

### 第二步：自驱动 source（空 `input_keys` + StopIteration 哨兵）

**动机**：hydp-dataflow 的多 source fan-in（`a=Read(A); b=Read(B); Merge(a,b)`）要**多路并行读**。若在 driver 侧
顺序 `next(iterA); next(iterB)` 喂 input_key，读取发生在 driver 主线程 → **必然串行**（实测双源 = 1.62× 单源）。
要真并行，reader 必须是**图内各自的 actor 节点**，由调度器同拍 submit 到不同 actor（实测同批 ra/rb 执行区间重叠
≈ 满 sleep）。

**机制**：`input_keys` 允许为空。此时每个 `deps==()` 的 root 是**有状态 reader**（`__init__` 持 `ray.data`/任意
迭代器，`run` 里 `return next(self._it)`）。调度器**乐观 admit** batch（不靠 driver 输入），每批驱动所有 root
各跑一次 `run`。迭代器枯竭时 `run` 抛 `StopIteration`：

```
reader.run():  return next(self._it)          # 算子零耦合:只是 Python 迭代器协议,不 import 任何引擎符号
      │  枯竭
      ▼
RunnerActor.run:  except StopIteration: return SOURCE_EXHAUSTED   # worker 内 catch(StopIteration 不能穿
      │                                                            # 出 remote task,否则变 RayTaskError)
      ▼
_Scheduler._on_complete:  isinstance(value, _SourceExhausted)     # driver 侧认哨兵(返回值,非异常,精确)
      │  → _source_exhausted=True(停 admit) + 作废本 batch(不产出/不下推)
```

**语义**：「**任一 source 枯竭 → 全图停**」，与迭代器驱动的「任一 input_key `StopIteration` 即停」完全对称。
不等长源停在**最短**的；抢先多读的那批落进 `_void`、不产出、不串位。

**1-ahead admit 门控**（`_self_driven` 为真时）：driver 侧 input_key 在 admit 前**同步** `next()`、当场知枯竭，不会
过度 admit；自驱动这边枯竭要等 actor 跑完才知，若乐观 admit 满 `max_inflight`，会在哨兵回来前**级联多 spawn 出界
的 tick**（bug：3 批数据的 reader 曾跑 7 次）。故门控「同一时刻至多 1 批 source 未跑完」。单 actor reader 的 tick
本就在 actor 上串行，提前 admit 多个 source tick 不加速读；下游 pipeline overlap 只需「reader[N] 完成即 admit
N+1」，1-ahead 不损 overlap，代价仅 1 个探测 tick（≈driver next() 撞 StopIteration 那一下）。

**在途竞态收尾**（`_retire_if_drained`）：某 source 先回哨兵作废 batch 时，同批兄弟 source 可能仍在途。不能立即
pop（它们完成时要查得到自己的 ctx/status），按 per-batch 未完成调用计数 `_batch_inflight` 归零才释放。`_dispatch`
对已作废/已释放的 batch 加守卫跳过。

**零耦合边界**：算子只 `return next(self._it)`（Python 原生 `StopIteration`），**不 import 哨兵/不认识调度器**。
`StopIteration→哨兵`的转换、哨兵识别全部收在 RunnerActor / `_Scheduler`（引擎内部执行协议，等价「for 循环认
StopIteration」）。哨兵 `SOURCE_EXHAUSTED` 是 `_SourceExhausted` 单例，跨 Ray 序列化后 `isinstance` 仍成立
（按模块路径引用，非对象 identity）。

**约束**：空 `input_keys` 的图**必须**至少有一个终会枯竭的 source root，否则 admit 循环不停（无外部输入 + 无
枯竭信号 = 无限图）。

**验收**：`test/test_selfdriven_source.py`（正确性 + 边界 + 时间账 bench）。覆盖单源/多源真并行（时间戳重叠）/
不等长（停最短、对齐不串）/空源/一路先枯竭另一路慢在途的竞态/inflight=1 不死锁/向后兼容 list 输入。
