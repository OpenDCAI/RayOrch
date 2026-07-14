# Multigrain User Guide（多粒度数据流开发指南）

> 面向使用者：教你怎么用 `rayorch.experimental.multigrain` 把一个"文档 → 页 → 块"这类
> 跨粒度、带 1:N / N:1 / M:N 关系的流水线，写成**声明式算子图**，交给框架去做分片并行、
> 血缘追踪和故障隔离。你只写"纯值函数"算子，血缘 / 分片 / 恢复全部由框架托管。
>
> 相关设计文档：原语与 IR 评审见 `docs/todos/11-multigrain-primitive-api-ir-review.md`；
> 三层关系模型见 `docs/todos/12-relation-model-three-tiers.md`；重排不变性定理见
> `docs/todos/13-reordering-invariance-theorem.md`；真实 MinerU 图片对象驱动的图形、
> 容灾和 LPT 集成测试与已知缺口见
> `docs/todos/17-mineru-graph-integration-findings.md`；当前 primitive 内核边界见
> `docs/todos/18-multigrain-primitive-core-convergence.md`。本文是**上手手册**，只讲怎么用。
>
> 面向框架开发者的完整代码脉络、字段字典、RelationSpec/Capability 推导、primitive
> 契约和执行/recovery 说明，以
> [`rayorch/experimental/multigrain/README.md`](../rayorch/experimental/multigrain/README.md)
> 为唯一入口。

---

## 0. 30 秒心智模型

- **Grain（粒度）**：一条流水线里，数据可以处在不同粒度——`pdf` 粒度、`page` 粒度、`block` 粒度。
- **Port（端口）/ Record（记录）**：每个粒度是一个 `PortBatch`，里面是**一列同粒度的记录**。
  每条记录有框架分配的 `record_id`、`display_key`、`ancestors`（祖先各粒度的 id）——**这些你都看不到、也不用管**。
- **Op UDF（算子）**：你只写一个类，`__init__` 里加载资源（模型等），`run(...)` 里对**一批值**做纯函数变换。
  **不接收、不依赖 id / 位置 / 血缘**。
- **Primitive（原语）**：`Map / Expand / Reduce / Filter / Select / Relate` 决定粒度怎么变（1:1 / 1:N / N:1 / M:N）。
- 框架负责：跨副本分片并行、按血缘重组、故障隔离与恢复。

一句话：**你描述"值怎么变、粒度怎么变"，框架负责"谁是谁的祖先、谁在哪块 GPU、坏了怎么办"。**

---

## 1. 算子 UDF 契约（最重要，先看这个）

算子是**持有状态的普通类**，遵循 RayModule 工厂模式：

```python
class RealVlmOcrPage:
    """Map 1:1 —— 对一页做 VLM OCR。"""

    def __init__(self, model: str = MODEL, gpu_memory_utilization: float = 0.9) -> None:
        # __init__ 里加载模型。框架的工厂模式保证它只在执行时、每个副本 actor 里跑一次，
        # 而不是在 driver 建图时跑 —— 所以 driver 侧不占显存。
        from vllm import LLM
        from mineru_vl_utils import MinerUClient
        self.llm = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization)
        self.client = MinerUClient(backend="vllm-engine", vllm_llm=self.llm)

    def run(self, pages: list[dict]) -> list:
        # 收到"这一片"的一批值，返回**等长**的一批输出。纯值函数：
        # 输出只依赖输入值本身，不依赖它在全局的位置 / id。
        images = [p["img_pil"] for p in pages]
        return list(self.client.batch_two_step_extract(images=images))
```

**硬性规则（违反会破坏并行正确性）：**

1. **`__init__` 只负责建资源**（加载模型、开连接）。编译图必须把**可导入的类**
   和构造参数交给原语；不要传已经构造好的实例。实例只允许本地 eager 调用，不能进入
   被动 IR。框架会在 executor/actor 中懒加载一次。
2. **`run` 是值纯函数**：输出只依赖输入值，**不得**依赖 `record_id`、全局下标、ordinal。
   （批内顺序框架会保证，你按位置处理没问题；但不能假设"我是全局第 37 条"。）这是重排不变性的前提。
3. **`run` 返回长度必须与输入对齐**（`Map` 1:1）。`Expand` 返回"每条输入 → 一个子列表"（1:N）。
   多输出原语必须显式写 `num_outputs=N`；实际输出数、每个输出的行数以及多输出
   Expand 的逐父组长度都会被严格校验，不再动态猜测。
4. **不要碰血缘 / id / 错误清单**——这些是框架的活。你连线用值，别用"飞线"（out-of-band 传路径 / 全局字典）。

各原语的 `run` 签名：

| 原语 | `run` 签名 | 语义 |
|---|---|---|
| `Map` | `run(rows) -> outs`（等长） | 1:1 逐条变换 |
| `Expand` | `run(rows) -> list[list[child]]`（每条一个子列表） | 1:N 展开 |
| `Reduce` | `run(anchors, grouped_a, grouped_b, ...) -> outs`（每锚一条） | N:1 归并 |
| `Filter` | `run(cols...) -> bool_mask`（等长布尔） | 丢行、保留身份 |
| `Select` | 打分/标注类，`run` 产出注解列 | annotate-then-filter 的糖 |
| `Relate` | 见 §5 三种模式 | M:N 关联 |

---

## 2. 声明流水线：`Pipeline` + `forward`

像 PyTorch 的 `nn.Module`：`__init__` 里声明算子（存类+参数，**不实例化模型**），`forward` 里用符号端口连线。

```python
import rayorch.experimental.multigrain as mg
from rayorch.experimental.multigrain.ir import PhysicalHints

class MinerUReal(mg.Pipeline):
    def __init__(self, output_dir, replicas=4, num_gpus_per_replica=1.0):
        super().__init__()
        self.to_pages = mg.Expand(
            RealPdfToPages, parent=0, child_label="page",
            physical=PhysicalHints(replicas=replicas),          # CPU 渲染，多副本
        )
        self.ocr = mg.Map(
            RealVlmOcrPage, model=MODEL,
            physical=PhysicalHints(replicas=replicas,
                                   num_gpus_per_replica=num_gpus_per_replica),  # GPU
        )
        self.assemble = mg.Reduce(
            RealAssembleDoc, output_dir=output_dir,
            missing_child="fail_closed",   # 见 §6：丢页 → 该文档不落残缺 md
        )

    def forward(self, pdfs):
        pages    = self.to_pages(pdfs)                       # pdf 粒度 -> page 粒度 (1:N)
        contents = self.ocr(pages)                           # page -> page (1:1)
        return self.assemble(mg.group_by(pdfs, contents, pages))  # page -> pdf (N:1)

ir = MinerUReal(output_dir="./out").compile()   # -> MultigrainIR（被动、可序列化、可优化）
```

关键点：
- **`compile()` 只建 IR，不跑模型、不占显存**（工厂模式）。IR 是被动数据结构，可打印、可校验、可过 pass。
- `Expand(parent=0)`：从第 0 个输入端口 fan-out；`child_label` 只是给子粒度起个可读名字。
- `Reduce` 的输入是 `mg.group_by(anchor, *descendants)`：第一个是**锚粒度**（输出粒度），其余是要按血缘归并回锚的后代端口，框架会**按阅读序对齐**后分组喂给 `run`。

---

## 3. 本地执行（无 Ray，用于开发 / 单测）

```python
from rayorch.experimental.multigrain import MultigrainExecutor, source

pdfs = source(pdf_paths, name="pdfs", display_key=lambda p: Path(p).stem)
out  = MultigrainExecutor().execute(ir, {"pdfs": pdfs})   # 单进程串行
print(out.values)      # 每个文档一条结果
print(out.errors)      # 隔离/连锁产生的 ErrorTrace 列表
```

`source(values, name=, display_key=)` 是入口端口；`name` 要和 `execute(ir, {name: ...})` 的键一致。
本地执行器语义与 Ray 执行器**逐行一致**，是写单测的首选（快、无 GPU）。

---

## 4. Ray 执行：副本并行 + 微批重叠 + 分片规划

```python
from rayorch.experimental.multigrain import (
    MultigrainRayExecutor, RunMetrics, lpt_shard_planner,
)
from mg_bridge.ops import page_work

metrics = RunMetrics()
ex = MultigrainRayExecutor(
    default_replicas=4,
    shard_planner=lpt_shard_planner(page_work),  # 见下：长尾均衡
    metrics=metrics,
)
ex.warm_pools(ir)                    # 预热：每副本各加载一次模型（把 load 排除出计时）

# 单批：同样走下面的统一 coordinator（max_inflight=1）
out = ex.execute(ir, {"pdfs": pdfs})

# 流式：输入可以是 generator；结果默认按输入顺序 yield，不会全部攒在内存。
microbatches = ({"pdfs": pdf_sources(chunk)} for chunk in chunks)
for out in ex.execute_stream(ir, microbatches, max_inflight=3):
    consume(out)

ex.shutdown()                        # 释放所有 actor 和 GPU
```

- **统一 DAG coordinator**：`execute`、兼容接口 `execute_microbatches` 和流式
  `execute_stream` 走同一套节点调度。节点在全部输入端口 ready 后运行；不同
  microbatch 与独立分支可并发，支持多输出和 fan-in，不写死 MinerU 拓扑。
- **bounded backpressure**：正在执行以及已完成但尚未被用户消费的结果都计入
  `max_inflight`，因此页图等大对象不会无限堆在 Ray object store。
- **持久 actor 池**：GPU/持模型算子的 actor 长期存活，模型**每副本只加载一次**，跨 chunk / microbatch / 多次 `execute` 复用。**隔离、重试都复用同一 actor，不会重载模型**（只有进程真崩溃才会被 Ray 重建 → 重载，那是基础设施故障，不是数据隔离）。
- **`PhysicalHints(replicas=, num_gpus_per_replica=)`**：写在算子上，声明它要几个副本、每副本几张卡。
- **`shard_planner`**：`(node, inputs, replicas) -> 每片的行下标列表`；返回 `None` 退化为连续切分。
  - **contiguous（连续切分）**：按行号顺序等分——实现简单，但长尾负载会造成 GPU 空泡。
  - **`lpt_shard_planner(work_fn)`（LPT，最长优先）**：按 `work_fn(row)` 估算每行开销，贪心把最重的先分给当前最闲的副本 → 消除长尾空泡。`work_fn` 要对**非本粒度的行**鲁棒（比如上游是字符串就返回 1.0）。

> **object store 上限**：页图很大，`ray.init(object_store_memory=...)` 要设个上限，否则积压会把 plasma 撑爆、被 cgroup OOM 杀掉 raylet。

---

## 5. 关系原语 `Relate`（M:N，三层模型）

按"从简单到通用"分三层，够用就用上面的：

1. **Expand / Reduce**：纯父子 1:N / N:1（最常用，见上）。
2. **`on=` 键连接**：各角色按字段等值关联；重复键会产生完整笛卡尔积。
   ```python
   self.link = mg.Relate(
       MyJoinOp,
       on={"image": "doc_id", "caption": "doc_id"},
   )
   ```
   编译图中的 key extractor 必须是字段名；自定义 callable 只允许 eager。
3. **`relation_adapter=` 可导入适配器**：关系不是等值连接时，用
   `"package.module:function"` 引用一个返回关系 evidence 的函数。
   ```python
   self.link = mg.Relate(
       MyOp,
       roles=("image", "caption"),
       relation_adapter="my_project.relations:link_visual_refs",
   )
   ```
   evidence 项为 `(value, {role: local_index})`；同一父证据要产出多条时，使用
   `(value, {role: local_index}, stable_key)`，保证 relation identity 在重排后稳定。
   `relation_fn=` 是 eager-only；执行已有 compiled IR 时，也可给 executor 注册
   `relation_fns={node_name: fn}`。当前 `Relate` 明确只支持一个输出端口。

`Relate` 满足重排不变性：任何合法的物理分片重排，输出与串行基线**逐条等价**（证明见 doc 13）。

---

## 6. 容错语义（本框架的核心卖点）

### 6.1 恢复阶梯与颗粒度

| | 触发方式 | 行为 | 浪费 |
|---|---|---|---|
| **记录级恢复** | `BadRecordError(msg, index=i, retryable=True)` | inline 单条重试，或进入跨 microbatch 的有界 stage epoch | 最细 |
| **确定性记录隔离** | `BadRecordError(..., retryable=False)` | 只隔离第 `i` 条，其余健康行保持密批执行 | 只赔坏记录 |
| **分片级重试** | 算子抛普通异常 | 立即把 shard 投到健康 replica，最多 `max_shard_retries` 次 | 重算整片 |
| **自适应定位** | shard 重试耗尽且 `degrade` | 有预算地二分失败子集；成功兄弟立即保留 | 稀疏故障时少于约 2× shard row-work |
| **保守终止** | 定位预算耗尽 | 只 quarantine 尚未解析的最小子集，再由 `fail_closed` 连锁 | 成本有硬上限 |

`index` 必须是**这次 `run` 收到的列表里的下标**（0-based），不是全局页号。框架负责把它映射回全局记录身份、写血缘、填 `ancestors`。

恢复分级的被动接口已经进入 IR：

```python
self.ocr = mg.Map(
    Ocr,
    recovery=mg.RecoveryPolicy(
        max_record_retries=2,
        retry_timing="inline",          # inline | deferred
        max_shard_retries=2,
        on_shard_exhausted="degrade",   # abort | degrade
        isolation=mg.IsolationBudget(
            max_work_factor=3.0,
            max_calls=64,
            on_exhausted="quarantine",
        ),
        drain_scope="stage_global",
    ),
)
```

当前实现支持 Map 的 inline/deferred record retry、立即 shard retry、actor
死亡后的单副本重建，以及有预算的自适应 shard 定位。`stage_global`
指**有界 stage epoch**：跨当前活跃 microbatch 聚合，在达到正常 shard
大小、stage 暂时无健康工作或输入关闭时 drain；不是等待整个数据集结束。
尚未支持的算子种类/策略组合会明确抛 `NotImplementedError`。

```python
from rayorch.runtime import BadRecordError

def run(self, pages: list[dict]) -> list:
    try:
        return list(self.client.batch_two_step_extract([p["img_pil"] for p in pages]))
    except PageError as e:
        raise BadRecordError(
            f"ocr failed on page {e.idx}",
            index=e.idx,
            retryable=e.is_transient,
        ) from e
```

> 想要记录级隔离，算子必须能把失败**归因到具体行下标**；否则退化成分片级重试。两者都不会写坏文件，区别只是浪费多少算力。

### 6.2 连锁终止：`missing_child` 策略（Reduce 上声明）

当一条后代记录被**永久隔离**（如某页 OCR 恢复失败），它所属的**锚（文档）**该怎么办？在 `Reduce` 上声明：

- **`missing_child="fail_open"`（默认）**：尽力而为——用**幸存的**后代拼出结果（可能是残缺文档）。
- **`missing_child="fail_closed"`**：**该文档的 `run` 根本不被调用** → **不落残缺 md**；框架给它填一个占位值
  `{"status":"incomplete","lost":[...]}`，并在输出端口的 `errors` 里加一条锚粒度的 `suppressed_incomplete`。
  干净文档不受影响，照常组装。

**`Reduce` 的 UDF 对此零改动**——它永远只收到"等长对齐"的干净文档；毒文档在框架层就被拦下了。

### 6.3 读取血缘 / quarantine

结果端口的 `.errors` 是一列 `ErrorTrace`：

```python
for e in out.errors:
    print(e.action)        # "quarantined" | "suppressed_incomplete" | ...
    print(e.logical_item)  # 如 "paper.pdf/page=1"
    print(e.failed_op)     # 出错算子名
    print(e.grain)         # 出错粒度
    print(e.ancestors)     # {"pdfs": <doc_id>, ...} 各祖先粒度的 id
```

`ancestors` 让下游 `Reduce` 能把"哪一页死了"精确对应到"哪篇文档"，从而做 §6.2 的连锁——**全程由框架维护，算子无感知**。

---

## 7. 端到端最小例子（可直接照抄）

```python
import rayorch.experimental.multigrain as mg
from rayorch.experimental.multigrain import MultigrainExecutor, source
from rayorch.runtime import BadRecordError

class SplitPages:                          # Expand 1:N
    def run(self, docs): return [[{"doc": d["name"], "page": i} for i in range(d["n"])] for d in docs]

class Ocr:                                 # Map 1:1（值纯 + 记录级隔离）
    def run(self, pages):
        outs = []
        for i, p in enumerate(pages):
            if p.get("bad"):
                raise BadRecordError("bad page", index=i)
            outs.append(f"ocr({p['doc']}#p{p['page']})")
        return outs

class Assemble:                            # Reduce N:1（纯值，零容错逻辑）
    def run(self, docs, grouped):
        return [f"{d['name']}=[{'|'.join(g)}]" for d, g in zip(docs, grouped)]

class Doc2MD(mg.Pipeline):
    def __init__(self):
        super().__init__()
        self.split = mg.Expand(SplitPages, parent=0, child_label="page")
        self.ocr = mg.Map(Ocr)
        self.asm = mg.Reduce(Assemble, missing_child="fail_closed")
    def forward(self, docs):
        pages = self.split(docs)
        return self.asm(mg.group_by(docs, self.ocr(pages)))

ir = Doc2MD().compile()
docs = source([{"name": "d1", "n": 2}, {"name": "d2", "n": 3}],
              name="docs", display_key=lambda d: d["name"])
out = MultigrainExecutor().execute(ir, {"docs": docs})
print(dict(zip(out.display_keys, out.values)))
print([e.action for e in out.errors])
```

切到 Ray 只需把执行器换成 `MultigrainRayExecutor(...)` +
`execute_stream` + `shutdown`，**算子和图一行都不用改**。

---

## 8. 常见坑（FAQ）

- **"driver 一 import 就占显存 / OOM"** → 模型加载写进了 driver 侧而非算子 `__init__`。遵守工厂模式：`compile()` 不应加载任何模型。
- **"并行后结果和串行不一样"** → 算子依赖了全局位置 / id，违反值纯度。只用输入值。
- **`Reduce` 里 `contents` 和 `pages` 长度对不上、崩了** → 说明有页被隔离且用了 `fail_open`。改用 `fail_closed`，框架会拦掉毒文档。
- **"某页坏了就整批重试，好慢"** → 抛的是普通异常。改抛 `BadRecordError(index=局部下标)` 走记录级隔离。
- **"隔离后是不是重载了模型？"** → 不会。同一持久 actor 复用，模型每副本只加载一次；日志里两次加载通常是你跑了两个独立实验模式各预热一次。
- **长尾负载 GPU 有空泡** → 用 `lpt_shard_planner(work_fn)` 替代连续切分。

---

## 9. 公开 API 速查

```python
import rayorch.experimental.multigrain as mg
# 原语： mg.Map / mg.Expand / mg.Reduce / mg.Filter / mg.Select / mg.Relate
# 组图： mg.Pipeline（子类化 + forward）
# 数据： mg.source / mg.group_by / mg.concat / mg.rebatch / mg.PortBatch / mg.Grouped / mg.ErrorTrace
# 执行：mg.MultigrainExecutor（本地）/ mg.MultigrainRayExecutor（Ray）
# 调度/观测：mg.lpt_shard_planner / mg.FaultSpec / mg.RunMetrics
# 恢复接口：mg.RecoveryPolicy / mg.IsolationBudget / mg.RetryTiming / mg.DrainScope
from rayorch.experimental.multigrain.ir import PhysicalHints, MissingChildPolicy
from rayorch.runtime import BadRecordError                  # 记录级隔离信号
```

IR / passes 属于研究 & 调试面，从 `graph` / `passes` 子模块取用；用户日常只碰上面这些。
