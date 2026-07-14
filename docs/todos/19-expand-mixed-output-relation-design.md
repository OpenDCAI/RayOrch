# Expand 混合输出与未来关系表达空间

状态：**设计已讨论，暂缓实现**。

这不是当前 Primitive Core 收敛、M2/M3 实验或论文主线的阻塞项。现阶段优先继续理解和
稳定现有 `Map / Filter / Select / Expand / Reduce / Relate`、被动 IR、lineage 与执行
边界；只有真实 workload 明确需要不可拆分的异构多输出时，再按本文进行可控迭代。

## 1. 问题来源

当前 primitive 的多输出属于同一个 relation family。以 Expand 为例：

```python
class ParsePages:
    def run(self, documents):
        return page_groups, page_meta_groups

pages, page_meta = mg.Expand(
    ParsePages,
    num_outputs=2,
)(documents)
```

两个 outputs 默认：

- 都是 `document → page` 的 `EXPAND`；
- 每个 document 下 group length 相同；
- 共享 child identity、ordinal 和 ancestry；
- 只是同一批 page records 的不同 value columns。

真实解析 UDF 偶尔会在一次不可拆分的底层调用中同时产生：

```text
pages             document → page
page_metadata      same(page)
document_metadata  same(document)
images             document → image
blocks             page → block
```

若强行把它们都当作一个 Expand child cohort，grain、cardinality 和 lineage 都会错误；
若要求用户无条件拆成多个 UDF，又可能重复 PDF decode、模型调用或数据复制。

## 2. 最终设计判断

不新增万能 `Transform`，不全面迁移现有 primitive，也不引入
`emit(kind=..., parent=..., ...)` 或字典式 output schema。

采用两层语义：

1. **大 primitive** 决定一次 UDF invocation 的主要 relation、输入作用域和执行约束；
2. **小 `mg.out.*` 原语** 只在 Expand corner case 中声明某个 output 的 identity source。

小原语收敛为两种：

```python
mg.out.same(values, as_=source)
mg.out.children(groups, of=source, grain=..., key=None)
```

未标注 output 完全沿用现有 Expand shared-child 行为。

## 3. 为什么只需要 `same + children`

它们对应 node-local relation forest 的两种边：

```text
same      1:1，复用 source identity
children  1:N，从 source 派生新 identity
```

组合后可以表示：

```text
document
├── same → document_metadata
└── children → page
    ├── same → page_metadata
    ├── same → page_image
    ├── children → block
    │   ├── same → block_bbox
    │   └── same → block_score
    └── children → figure
        └── same → figure_caption
```

只要 output 关系满足以下条件，就属于这个表达空间：

- 每条 output record 有且仅有一个直接 identity source；
- source 是 UDF input 或同一 return bundle 中的前序 output；
- relation 无环；
- relation 在本次 invocation 内可确定；
- cardinality 是 1:1 或 1:N。

不属于 relation forest 的语义继续由大 primitive 表达：

```text
0:1       Filter / Select
N:1       Reduce
M:N       Relate
fixed K-way routing  future Partition
global/window/async  future execution/workflow scope
```

## 4. 目标 API

### 4.1 普通多输出不变

```python
class ParsePages:
    def run(self, documents):
        return page_groups, page_meta_groups
```

第一个未标注 output 是隐式 child cohort，后续未标注 outputs 与它共享 identity。
现有 `num_outputs`、eager/compiled 行为和测试不变。

### 4.2 Child 与 parent-level metadata

```python
class ParsePdf:
    def run(self, documents):
        parsed = parse_pdf_batch(documents)
        pages = mg.out.children(
            parsed.pages,
            of=documents,
            grain="page",
        )
        document_metadata = mg.out.same(
            parsed.metadata,
            as_=documents,
        )
        return pages, document_metadata
```

### 4.3 Child metadata 与 sibling 对齐

```python
pages = mg.out.children(
    page_groups,
    of=documents,
    grain="page",
)
page_metadata = mg.out.same(
    page_meta_groups,
    as_=pages,
)
return pages, page_metadata
```

`page_metadata` 复用 pages 的：

- record ID；
- grain；
- ancestry；
- ordinal；
- source lineage。

运行时必须校验 group shape 完全一致。

### 4.4 多个独立 child cohorts

```python
pages = mg.out.children(
    page_groups,
    of=documents,
    grain="page",
)
images = mg.out.children(
    image_groups,
    of=documents,
    grain="image",
)
tables = mg.out.children(
    table_groups,
    of=documents,
    grain="table",
)
return pages, images, tables
```

三个 outputs 可以有不同 group lengths、grain 和 child identity space。

### 4.5 一次 UDF 产生多层 fanout

```python
pages = mg.out.children(
    page_groups,
    of=documents,
    grain="page",
)
blocks = mg.out.children(
    block_groups,
    of=pages,
    grain="block",
)
block_scores = mg.out.same(
    score_groups,
    as_=blocks,
)
return pages, blocks, block_scores
```

source 必须已经声明，禁止 forward reference 和循环。

### 4.6 稳定 child key

```python
pages = mg.out.children(
    page_groups,
    of=documents,
    grain="page",
    key=page_number_groups,
)
```

`key` 是业务稳定 evidence，框架据此派生 identity；UDF 仍不能读取或直接指定内部
`record_id`。

## 5. 大 primitive 的职责

### 5.1 Map

Map 表达 aligned row-local 1:1 enrichment：

```text
page → normalized page
page → quality score
page → embedding
```

多个 outputs 默认共享 input identity。以下情况不应通过 `mg.out.*` 扩展 Map：

- variable-length detections/chunks：使用 Expand；
- 删除 records：使用 Filter/Select；
- batch latency/throughput：使用 metrics；
- operator failure：使用 ErrorTrace。

### 5.2 Expand

Expand 表达 parent-local 1:N decomposition，是唯一需要 mixed output override 的现有
primitive。支持：

- 默认 shared child columns；
- parent-level result；
- 独立 child cohorts；
- sibling-aligned metadata；
- node-local nested fanout；
- empty child groups；
- stable child key。

一个 child 若有多个逻辑 parent，不再属于 Expand，应使用 Relate。

### 5.3 Filter / Select

它们表达 identity-preserving 0:1 selection 和 annotation。accepted/rejected/review 等
固定多路输出未来应新增大 primitive `Partition`，而不是 `out.subset/out.route`。

### 5.4 Reduce

Reduce 表达 group-complete N:1 anchor aggregation。多个 markdown/statistics/embedding
outputs 可以全部保持 anchor identity，不需要小 marker。child-level residue 应在 Reduce
前保留，失败使用 ErrorTrace。

### 5.5 Relate

Relate 表达 multi-role M:N 和多父 evidence。matched/unmatched、outer join、hot-key
salting 和分布式 join 是 Relate/Join 的逻辑或物理扩展，不属于 Expand return marker。

## 6. 现实 workload 审视

以下结论基于 2025–2026 年公开系统和数据管线。

### 6.1 Data-Juicer 2.0

Data-Juicer 已覆盖 200+ text/image/audio/video/multimodal operators；约 90% 是 Mapper 和
Filter，其余包括 Formatter、Deduplicator、Selector、Grouper、Aggregator、FusedOP、
ScriptOP 和 HumanOP。

对应关系：

```text
Mapper             Map
Filter             Filter / Select
sample decomposition Expand
Grouper/Aggregator Reduce
cross-modal match  Relate
Selector/routing   future Partition/Global Select
Deduplicator       future keyed/global scope
HumanOP            future async materialization boundary
```

参考：

- <https://github.com/datajuicer/data-juicer>
- <https://arxiv.org/html/2501.14755v2>

### 6.2 NeMo Curator

NeMo Curator 2026 版本覆盖：

- text：清洗、语言检测、分类、dedup；
- image：embedding、aesthetic/NSFW 分类、过滤、dedup；
- video：video → clip → frame、转码、embedding、caption、dedup；
- audio：ASR、WER 和质量分析。

video → clip → frame → embedding/caption 可由 Expand relation forest 表达；semantic
dedup 的 KMeans/pairwise/duplicate identification 需要 global scope。

参考：

- <https://github.com/NVIDIA-NeMo/Curator>
- <https://docs.nvidia.com/nemo/curator/curate-video/process-data>

### 6.3 Docling

Docling 的 `DoclingDocument` 包含 page、text、table、picture、key-value、document
hierarchy、bbox 和 page provenance。document → page → item 与每层 metadata 属于
`children/same`；table cell relationship、reading-order edge 等图关系需要 Relate。

参考：

- <https://docling-project-docling.mintlify.app/concepts/docling-document>
- <https://docling-project-docling.mintlify.app/concepts/architecture>

### 6.4 Ray Data

Ray Data 2.56 的主要逻辑 primitive 是：

```text
map / map_batches
flat_map
groupby / map_groups
join
repartition / sort / shuffle
```

这验证 Map/Expand/Reduce/Relate 的 cardinality 基础，但也说明 keyed/global shuffle 是
独立执行维度。

参考：<https://docs.ray.io/en/latest/data/transforming-data.html>

### 6.5 Apache Beam

Beam 使用固定 TupleTag 表达多输出，通过 Partition 做固定多路 routing，通过
GroupByKey/CoGroupByKey 表达 grouping/join，并单独建模 window、watermark、trigger
和 late data。

参考：

- <https://beam.apache.org/documentation/transforms/java/elementwise/pardo/>
- <https://beam.apache.org/documentation/transforms/java/aggregation/cogroupbykey/>

### 6.6 OmniCorpus

OmniCorpus 将 HTML、image-text pair 和 video frame/subtitle 统一为 interleaved payload，
并保存 aesthetic、NSFW、toxicity 等 metadata。这里需要区分：

- interleaved document 的普通字段：typed payload；
- 需要独立调度/恢复的 image、frame、chunk：child port；
- text-image match：Relate；
- document/image dedup：global scope。

参考：<https://github.com/OpenGVLab/OmniCorpus>

## 7. Payload 与 Port 的边界

不是每个结构字段都应该成为 port。

只有需要以下至少一项的实体才提升为 child port：

- 独立调度；
- 独立分片和重平衡；
- record-level recovery；
- lineage 查询；
- 独立物化；
- 与其他 grain 建立 relation。

否则保留为 typed payload field：

```text
page bbox              payload
page OCR workload      page port
block quality score    aligned payload/port，按独立消费需求决定
figure requiring VLM   figure port
debug latency          metric
operator failure       ErrorTrace
```

## 8. 明确不由 `mg.out.*` 表达的语义

### 8.1 Fixed routing

```text
accepted / rejected / manual_review
```

未来使用强约束 `Partition`。port 数在 graph construction 时固定，运行时每个 port 可为空。

### 8.2 Global/keyed operations

```text
exact/fuzzy/semantic dedup
clustering
ranking
sampling
sort/shuffle
global statistics
```

它们需要 shuffle、index、materialization 或多阶段算法，应新增 global/keyed execution
scope 或专用大 primitive。

### 8.3 Window/streaming

```text
event time
window
watermark
trigger
late data
repeated pane
```

未来用 `WindowSpec` 和 streaming execution semantics 表达。

### 8.4 Human/external workflow

```text
Label Studio
remote service callback
manual review
pause/resume
timeout
```

需要 materialized async boundary、external task ID 和 resume token。

### 8.5 Dynamic ports 和循环

运行时不能根据数据创建新的 port 名；固定 port 可以为空。生成→评分→重生成等循环由
外层 workflow 表达，不能把 DAG cycle 编码进 return。

## 9. 编译和运行时边界

### 9.1 编译期

编译器只做 restricted marker analysis：

- 识别最终 return tuple；
- 识别局部 `out.same/out.children` 赋值；
- 建立 input/前序 output source refs；
- 检查 branch marker topology 一致；
- 检查 output forest 无环；
- 生成逐 output `RelationSpec`。

不会：

- 实例化 operator/model；
- 用 SymbolicPort 执行黑盒 UDF；
- 分析 PDF/model/list 的普通业务计算；
- 猜测动态 helper 的真实 relation。

源码不可分析但必须使用特殊 marker 时，提供 positional fallback；普通现有代码继续使用
`num_outputs`。

### 9.2 运行时

- UDF 每个 shard 只调用一次；
- typed wrappers 只携带 values 和 invocation-local evidence；
- framework 按 source dependency 顺序物化 outputs；
- `same` 校验 shape 并复用 metadata；
- `children` 校验每个 source record 对应一个 group；
- 任一 output 失败，整个 return tuple 原子失败；
- Local/Ray 共用同一 materializer；
- 不扩大现有 record-level recovery 支持范围。

## 10. 用户、Agent 和 Debug 体验

API 只需记住：

```text
不写 marker：沿用 Expand 默认
same：共享已有 identity
children：创建 child identity
```

类型：

```python
out.Same[T]
out.Children[T]
```

IR/debug 应显示：

```text
ParsePdf.pages:
  EXPAND documents → page

ParsePdf.page_meta:
  PRESERVE ParsePdf.pages

ParsePdf.document_meta:
  PRESERVE documents
```

错误应直接指出 output、source 和 shape：

```text
ParsePdf output 'page_meta':
out.same(..., as_=pages) expected 12 values, got 11.
```

```text
ParsePdf output 'blocks':
out.children(..., of=pages) expected 12 groups, got 3.
```

不使用 dict、字符串 parent ID 或 framework record ID。

## 11. 未来扩展判定

遇到新需求时固定按以下顺序判断：

1. 单 source 且 1:1/1:N：`same/children`；
2. 0:1 或固定多路 routing：Filter/Select/Partition；
3. 必须收齐 group：Reduce；
4. 多 source/M:N：Relate；
5. 全局可见性、时间窗口或外部等待：新增 execution/workflow scope；
6. 只是 value 内部结构：保持 payload，不提升 port。

只有至少两个真实 workload 无法通过上述组合表达时，才考虑增加新的小 marker。

## 12. 若未来恢复实现，建议阶段

### R1：只实现 parent/sibling identity override

- `out.same`；
- `out.children` 仅以 graph input 为 source；
- 现有多输出 Expand 完全兼容；
- eager/compiled/Local 语义等价。

### R2：node-local nested forest

- sibling output source；
- document → page → block；
- cycle/forward-reference verifier；
- stable child key。

### R3：Ray 和重排证明

- parent-row sharding；
- contiguous/LPT 等价；
- output/lineage/recovery atomicity；
- 更新 reordering-invariance theorem。

### R4：根据真实需求评估新大 primitive

- Partition；
- distributed keyed Relate/Join；
- two-phase Reduce；
- global/window/async scope。

## 13. 当前行动

暂不实现 R1–R4。当前优先：

1. 理解并稳定现有框架结构；
2. 保持 Primitive Core、IR、handler、capability 边界清晰；
3. 完成现有实验与收敛任务；
4. 以真实 workload 驱动，而不是预先扩大公共 API。
