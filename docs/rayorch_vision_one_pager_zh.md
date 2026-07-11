# RayOrch 愿景：多模态 AI 数据管线的高性能执行与恢复基座

## 定位

RayOrch 是面向 DataFlow-MM、Flash-MinerU、RAG ingestion 和多模态数据治理的
分布式 AI Dataflow Runtime。用户用普通 Python 算子构建 DAG；系统负责 CPU/GPU
异构调度、流水线并行、细粒度 lineage、坏数据隔离与局部恢复。

> 让包含 PDF、图像、页面、文本块、VLM、OCR、LLM 等异构算子的复杂数据管线，
> 在高吞吐执行的同时，能够精确定位失败记录，并只重放真正受影响的计算。

> **上手开发**：见 [`multigrain_user_guide.md`](multigrain_user_guide.md) ——
> 算子 UDF 契约、原语、Ray 执行、以及容错/恢复（`BadRecordError`、`missing_child`
> 连锁终止）的实操指南。

## 编程模型

用户算子始终是可脱离 Ray 单独调试的普通 Python 类。Pipeline 使用接近
Torch module graph 的普通调用描述业务拓扑，逻辑 wrapper 声明 cardinality：

```python
class OCR:
    def run(self, images, layouts):
        return texts

def __init__(self):
    self.pdf2pages = orch.Expand(Pdf2Pages, parent=0)
    self.layout = orch.Map(Layout)
    self.ocr = orch.Map(OCR)
    self.assemble = orch.Reduce(Assemble)

def forward(self, documents):
    pages, page_meta = self.pdf2pages(documents)
    layouts = self.layout(pages)
    texts = self.ocr(pages, layouts)
    return self.assemble(orch.group_by(documents, texts, page_meta))
```

编译器从 `forward()`、函数签名和类型注解中推断 DAG、参数与端口；从
`orch.Map` / `orch.Expand` / `orch.Filter` / `orch.Reduce` / `orch.Relate`
获得逻辑 cardinality contract。这个 DAG 是 RayOrch 的 relation-aware IR：
优化器可以在其上做 grain 校验、`1:N` 后重排、算子融合、reduce group planning
和后续 Ray actor lowering。

## 多粒度数据关系

每个 output port 可以拥有独立 grain，例如 `pages` 是 page records，
`document_meta` 仍是 document records。五种逻辑算子覆盖主要业务：

```text
MAP      1:1     OCR、embedding、layout、VLM inference
FILTER   1:0/1   质量过滤、安全过滤、置信度筛选
EXPAND   1:N     PDF→page、page→block、image→crop、prompt→candidates
REDUCE   N:1     block→page、page→document、candidates→answer
RELATE   M:N     dedup、cluster、merge、多模态样本配对
```

PDF 页面展开示例：

```python
self.pdf2pages = orch.Expand(Pdf2Pages, parent=0)

pages, page_meta = self.pdf2pages(documents)
```

逻辑 wrapper 提供静态端口契约；`expanded()` 或等价的嵌套返回值提供当前批次
的动态父子关系。直接调用用户算子仍返回普通业务值，用户不接触 record ID 或
lineage。

## Runtime 如何执行

```text
dataset
  -> executor 切分 microbatch
  -> 不同 batch 在 DAG stage 间流水线重叠
  -> 每个 stage 由多个 Ray Actor replica 并行执行
  -> output port 独立维护 record grain
  -> 1:N 后按 child grain 重排/重组, 让 GPU stage 吃满批次
  -> 健康记录继续流动，坏记录进入 quarantine
  -> lineage 记录父子、分支和算子关系
```

因此 Flash-MinerU 可以把 PDF 展开为独立 page records，跨 GPU 调度 layout/OCR；
Runtime 在 page grain 上重排以减少长短 PDF 造成的空泡，最终再按 document
anchor 聚合 Markdown。单页失败不拖垮整个 PDF，并且可以直接追溯到
`document/page/op`。

## 研究核心

Lineage 不只用于打印错误路径，而要直接参与恢复：

```text
page OCR 失败或 GPU Actor 崩溃
  -> 根据 lineage 确定受影响的 page、document 和下游子图
  -> 复用已成功的 PDF decode、layout 或 embedding 结果
  -> 只重放必要记录和节点
  -> 其他 microbatch 与 GPU stage 继续执行
```

五个核心贡献：

1. **异构 AI DAG Runtime**：CPU、GPU、OCR、VLM、LLM 算子的 microbatch
   流水线执行和资源隔离。
2. **Relation-aware DAG IR + optimizer**：把用户友好的 Torch-like API 编译成
   带 grain、cardinality、parent relation 的算子 DAG，再通过 pass 做
   cross-grain 校验、rebatching 插入、Map/Filter 融合和 reduce group planning。
3. **低开销 multi-grain lineage**：`1:1` 路径压缩，`1:N/N:1/M:N`
   身份变化时记录 parent relation。
4. **Relation-aware rebatching**：`1:N` 后保留 parent relation 与 child
   ordinal，在 page/block/chunk 粒度重排 GPU-heavy stages，减少空泡后再自动
   `group_by` 回 document/sample。
5. **Lineage-guided partial replay**：根据 lineage、算子代价、故障类型和
   已提交中间结果选择最小恢复范围。

## DataFlow-MM

DataFlow-MM 提供可复用的多模态数据治理算子与 Pipeline；RayOrch 作为高性能
分布式执行后端：

```text
DataFlow-MM operators/pipelines
  -> RayOrch compiler/runtime
  -> heterogeneous CPU/GPU cluster
  -> lineage + quarantine + recovery + observability
```

## 达到投稿标准的验收条件

要达到 VLDB/SIGMOD Research Track 竞争力，必须证明：

- 无故障时，性能接近或超过 plain Ray / Ray Data；
- 编译期 IR 能表达并校验 `Map/Expand/Filter/Reduce/Relate` 以及常见融合计划；
- `1:N` 后重排能显著降低长短文档不均带来的 GPU idle time；
- 有坏数据、GPU OOM、Actor crash 或 node loss 时，GPU goodput 明显更高；
- partial replay 的记录数和子图范围显著小于整 batch/stage 重跑；
- lineage 的时间、内存和持久化开销低且可预测；
- Flash-MinerU 与至少一个 DataFlow-MM/RAG 多模态管线在 8-32 GPU 上验证；
- 与 Ray Data、whole-batch retry、逐条执行和固定二分隔离进行公平比较。

```text
goodput = 成功完成的有效记录数 / 总 GPU 时间
```

当细粒度 lineage 能减少重复 GPU 计算、缩小恢复范围并持续推进异构管线时，
RayOrch 才从工程框架闭环为具有顶会投稿潜力的 AI Dataflow 执行与恢复系统。
