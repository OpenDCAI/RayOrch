# TODO：多粒度端口和基数 API

状态：正在`rayorch.experimental.multigrain`下进行实验性 MVP。

## 目标

支持记录保存、扩展、过滤、缩减、通用
多对多运算符，无需将记录 ID 或沿袭内部暴露给
用户。

中心抽象是：

> 每个输出端口都有自己的记录批次和基数关系。

因此，节点可能会以不同的粒度返回端口：

```text
port "pages"         -> page records
port "document_meta" -> document records
```

这取代了当前一个节点的每个输出端口共享的假设
相同的`MicroBatch`。

## 前端糖目标

顶级创作模型应保留 RayOrch 原生的类似 Torch 的 DAG：

```text
__init__   declares logical operator wrappers
forward    calls wrappers like ordinary modules
compiler   traces calls into relation-aware IR
```

其他 API 形式不是竞争的管道范例。它们是互补的
为相同 IR 提供相关证据的前端糖：

```text
wrapper declaration          self.op = orch.Expand(Op, parent=0)
forward relation helper      self.reduce(orch.group_by(anchor, children))
plain return shape           nested groups, masks, annotations
adapter hook                 relation_fn / mask_fn / key_fn for pure UDFs
return protocol              escape hatch for advanced relation-heavy operators
```

所有这些都应该规范化为一个小的关系感知 IR 代数：

```text
Map / Expand / Filter / Reduce / Relate / Project
```

因此，设计目标是：

> 保持面向用户的 API 灵活且符合人体工程学，但保持 IR 小巧、独特、
> 可序列化且优化器友好。

## 设计原则

- 用户运算符仍然是普通的 Python 类，具有普通的`run()`方法。
- `Pipeline.forward()`应该读起来像 Torch 模块图：操作符是
  模块属性并直接调用。
- 模块包装器，例如`orch.Map`、`orch.Expand`、`orch.Filter`、
  `orch.Reduce`和`orch.Relate`声明逻辑基数契约。
- 在运行时之外直接调用底层用户操作符返回普通值
  可用的商业价值。
- 类型注释描述业务数据类型，而不是执行语义。
- 小返回助手携带当前调用的动态关系数据
  当静态合约还不够时。
- 仅运行时包装器永远不会将参与者适配器转义到下游用户
  运营商。
- 用户从不构建或观察全局记录 ID、谱系头或父代
  边缘。
- 一次 Actor 调用仍会产生一个 Ray RPC 结果。

## 关系契约边界：这与 HYDP-dataflow 有何不同

这种设计有意与 HYDP-dataflow 的分离不同。

HYDP-dataflow 针对后端解耦进行了优化：

```text
business UDF
  -> receives neutral table/object batches
  -> returns neutral table/object batches
  -> framework owns ports, columns, execution backend, and most metadata
```

重要的 HYDP 边界是操作员是否接触后端原生
处理程序，例如`ray.data.Dataset`、Spark DataFrame 或磁盘句柄。如果它
事实并非如此，UDF 仍然是可移植的并且大多与框架无关。红外可以
推断或存储端口形状、列绑定、资源提示和引擎降低
UDF 主体之外。

RayOrch multigrain 有一个不同的难题。仅仅知道一个
节点有输入和输出端口。运行时还必须知道细粒度
UDF创建的记录关系：

```text
document -> pages
page -> kept page or business drop
pages -> document
image + caption -> matched pair
```

该关系并不总是能够从普通的 Python 值中恢复出来
事实。简单的列表并没有说明哪个父母生下了每个孩子。过滤后的列表
没有说明丢失的行是否是业务下降或失败。一对列表
没有说明哪一行图像和标题生成了每一对。

重要的结论很微妙：

> RayOrch multigrain 不必放弃独立于框架的 UDF，但它
> 无法支持所有原语的完全关系无关的 UDF 合约。

运行时需要动态关系证据。该证据可以存在于三个
地方：

```text
1. Wrapper declaration in __init__
   static contract: Map / Expand(parent=0) / Filter / Reduce / Relate roles

2. Relation expression in forward()
   routing contract: group_by(anchor, descendants), future zip_by_identity(...)

3. Ordinary UDF return shape or an external adapter
   dynamic evidence: nested groups, masks, local parent indexes, relation_fn(...)
```

这些机制是同一 IR 的互补前端。他们不应该
成为单独的管道范例。顶级的创作风格仍然是
RayOrch 原生的类似 Torch 的 DAG：

```text
__init__   declares logical operator wrappers
forward    calls wrappers like ordinary modules
compiler   traces calls into relation-aware IR
```

在该 DAG 内，框架应将关系证据表面保持为
尽可能小：

```text
static wrapper contract      Map / Expand / Filter / Reduce / Relate
forward relation helper      group_by(...), future zip_by_identity(...)
plain return shape           nested groups, masks, annotations
adapter hook                 relation_fn / mask_fn / key_fn for pure UDFs
return protocol              escape hatch for advanced Relate-like cases
```

所有这些都必须规范化为相同的小型 IR 代数：

```text
Map / Expand / Filter / Reduce / Relate / Project
```

因此，API 的决定不是“哪种风格取代其他风格”。这是：
更喜欢能够提供足够关系的侵入性最小的前端机制
独特IR的证据。

因此，最理想的边界不是“UDF 必须导入或知道
RayOrch”。它是：

```text
UDF may remain framework-independent.
The pipeline must still provide enough static and dynamic relation information
to build a unique relation-aware IR.
```

所需的边界是：

```text
User UDF:
  returns ordinary business values, or ordinary values with interpretable shape

Logical wrapper:
  orch.Map / Expand / Filter / Reduce / Relate declares static relation contract

Optional adapter:
  extracts dynamic relation evidence when plain return values are ambiguous

Runtime:
  creates internal record ids, parent edges, ordinals, lineage, replay metadata
```

这是关系契约，而不是内部元数据契约。用户可以返回
普通值、嵌套组、掩码、分组输入或本地关系
描述符。或者，包装器可以接收适配器，例如
`relation_fn=`解释完全独立于框架的 UDF 输出。用户
不应构造`PortBatch`、全局记录 ID、谱系增量、隔离区
记录或重播句柄。

实际规则是：

```text
HYDP-dataflow:
  keep UDFs decoupled from execution backend and framework metadata

RayOrch multigrain:
  keep UDFs decoupled from internal ids and, when possible, from RayOrch APIs;
  require the pipeline contract to expose cardinality/relation semantics when
  values alone are ambiguous
```

示例：

```python
class PdfToPages:
    def run(self, pdfs):
        # Nested lists are business values plus local 1:N shape.
        return [[page0, page1], [page0]]

self.pdf_to_pages = orch.Expand(PdfToPages, parent=0)
```

```python
class KeepPages:
    def run(self, pages, scores):
        # A bool mask is business validation semantics, not a failure signal.
        return [score >= 0.8 for score in scores]

self.keep_pages = orch.Filter(KeepPages)
```

```python
class MatchImagesAndCaptions:
    def run(self, images, captions):
        # This can remain a plain business return.
        return pairs

self.match = orch.Relate(
    MatchImagesAndCaptions,
    roles=("image", "caption"),
    relation_fn=lambda pairs: [
        (pair, {"image": pair.image_index, "caption": pair.caption_index})
        for pair in pairs
    ],
)
```

这种受控关系契约对于该研究的研究价值至关重要
系统。 IR 不仅仅是一个类型化端口 DAG。它是一个关系感知管道
进行沿袭跟踪、行级隔离、部分重播的合约，
关系感知重新批处理，以及无需询问即可优化的 M:N 数据治理
用户操作内部运行时元数据。

## UDF 逻辑与 IR 合约

IR 不应尝试编码或理解用户 UDF 内的所有逻辑。这是
不是 Python AST、模型图或业务规则引擎。它应该编码
框架需要正确性、跟踪、恢复和恢复的关系效果
优化。

```text
IR must know:
  grain
  cardinality effect
  parent / anchor / relation roles
  child ordinal or stable key
  business drop vs. failure quarantine
  deterministic / side-effect / retryable properties

IR may ignore:
  OCR model internals
  PDF parsing details
  layout heuristics
  scoring formula
  matching model internals
  arbitrary business logic that does not affect relation semantics
```

这与类型系统或数据库逻辑计划的分离相同：IR
记录优化器和运行时所需的语义契约，而 UDF
仍然是业务逻辑的实现。

实现应该使用三层合约模型：

```text
1. Static declaration
   What the operator promises at compile time.

2. Dynamic evidence
   What the actual invocation returns or what an adapter extracts.

3. Runtime contract check
   Assertions that dynamic evidence satisfies the static contract.
```

示例：

```python
self.pdf_to_pages = orch.Expand(PdfToPages, parent=0)
```

静态声明：

```text
EXPAND, parent input = 0
```

动态证据：

```text
UDF returns nested groups, one group per parent row
```

运行时检查：

```text
group count equals parent row count
multi-output groups share lengths unless declared otherwise
children receive parent relation and child ordinal
```

```python
self.keep_pages = orch.Filter(KeepPages)
```

静态声明：

```text
FILTER, same-grain inputs, kept rows preserve identity
```

动态证据：

```text
UDF or adapter returns a boolean mask
```

运行时检查：

```text
mask length equals input row count
mask values are bools
dropped rows are business drops, not quarantine records
kept rows preserve identity
```

```python
self.match = orch.Relate(
    MatchImagesAndCaptions,
    roles=("image", "caption"),
    relation_fn=extract_pair_parents,
)
```

静态声明：

```text
RELATE, roles = image/caption, output grain = pair
```

动态证据：

```text
adapter extracts invocation-local parent references
```

运行时检查：

```text
every parent reference points to an input row visible in this invocation
role names match declared input roles
relation ordering / multiplicity policy is explicit
no global runtime record ids are accepted from user code
```

当 UDF 内部无法静态时，这为系统提供了本地解决方案
绑定到 IR：保持 UDF 逻辑不透明，但检查其可观察的关系
运行时的效果。违反合同行为应报告为框架错误
包含节点名称、预期合约、观察到的形状和面向用户的项目
可用时的上下文。

## 首选编程模型：火炬式调用+内部关系

首选的 Surface API 是类似 Torch 的模块样式：

```python
class MineruPipeline(orch.Pipeline):
    def __init__(self):
        self.pdf_to_images = orch.Expand(PdfToImages, parent=0)
        self.layout = orch.Map(Layout)
        self.ocr = orch.Map(OCR)
        self.assemble = orch.Reduce(Assemble)

    def forward(self, pdfs):
        images, page_meta = self.pdf_to_images(pdfs)
        layouts = self.layout(images)
        texts = self.ocr(images, layouts)
        markdown = self.assemble(orch.group_by(pdfs, texts, page_meta))
        return markdown
```

`forward()`保持业务形状：用户呼叫运营商，而不是方法
端口。每个运算符的逻辑包装告诉编译器什么
期望的基数关系。内部运行时将每次调用标准化
端口本地批次加上关系代数：

```text
Port
  -> logical grain
  -> PortBatch at execution time
  -> parent relation to one or more input ports
  -> lineage heads for recovery and attribution
```

对于常见的PDF情况：

```text
pdfs                 document grain
  -- Expand -->      images/page_meta at page grain, parent = pdfs
  -- Map -->         layouts/texts preserve page grain
  -- group_by -->    group page descendants by ancestor pdf
  -- Reduce -->      markdown returns to document grain
```

用户看到的是普通的 Python 模块调用。系统看到：

```text
Expand + Map + Map + GroupByAncestor + Reduce
```

这使前端符合人体工程学，同时为运行时提供了正式的关系
用于沿袭、故障隔离和未来部分重放的模型。

## DAG IR 和优化器通行证

类似 Torch 的 API 应在物理之前编译为显式运算符 DAG
执行。该 DAG 是主要 IR 边界：

```text
user forward()
  -> traced logical DAG
  -> relation-aware IR
  -> optimizer passes
  -> physical execution plan
```

面向用户的 API 可能会保持高水平且符合人体工程学，但 IR 应该保持不变
正交且易于优化。例如，未来的便利 API，例如
`Select`可以表示为较低级别的逻辑运算：

```text
Select(quality_then_keep)
  -> Map(score / annotation)
  -> Filter(mask)
  -> Project(kept annotations)
```

运行时仍然可以选择将其作为一个融合的 Actor 调用来执行。这
重要的分离是：

```text
logical IR:    precise, analyzable, relation-preserving
physical plan: fused, reordered, resource-aware
```

### IR节点合约

每个编译节点至少应携带：

```text
node name
operator kind: Map / Expand / Filter / Reduce / Relate / helper
input ports
output ports
input and output grains
cardinality contract
parent input, for Expand
grouping anchor and descendants, for Reduce
logical output names
physical hints: replicas, resources, max_inflight, environment
```

这使得 DAG 不仅仅是一个执行顺序。就变成共享的了
用于验证、谱系生成、调度和降级的表示
雷演员。

### 初始优化器通过

第一个优化器应该是简单、显式的传递而不是广泛的成本
基于优化器：

```text
ValidateCrossGrainFanIn
  reject ordinary Map calls that mix document/page/block grains without group_by

InsertRebatchAfterExpand
  insert child-grain rebatching before GPU-heavy Map stages

FuseMapFilter
  turn Map(score) + Filter(mask) into one physical Select stage

FuseAdjacentMaps
  fuse cheap consecutive Map stages at the same grain when safe

PlanReduceGroups
  choose how group_by(anchor, descendants) materializes groups and orders children

PlaceMaterializationBoundaries
  decide which ports need persistence for tracing, replay, or expensive recompute
```

优化应该保留逻辑血统。一次通过可能会改变物理批次
布局或熔断运算符，但不得删除逻辑运算符名称和
跟踪视图和部分重播所需的关系。

### 设计规则

应该允许 API 发展便利的包装器，但 IR 应该保持
小关系感知代数：

```text
API: friendly shortcuts for users
IR:  Map / Expand / Filter / Reduce / Relate plus explicit grouping metadata
```

这保留了未来的功能，例如融合`Select`，关系感知重新批处理，
雷演员从变成一次性特例。

## 最小逻辑原语

最小的有用原语集是：

```text
Map       1:1      preserve the input record identity
Expand    1:N      one parent record produces zero or more child records
Filter    1:0/1    intentionally drop records while preserving kept identities
Reduce    N:1      group descendants by an anchor grain and produce anchor rows
Relate    M:N      explicitly attach outputs to one or more local parents
```

`flat_map`是`Expand`；`select`是`Filter`；`aggregate`是`Reduce`；
重复数据删除、集群、合并和多模式配对一旦成为它们的`Relate`
输入记录在一次调用中位于同一位置。

两个辅助表达式是图语法的一部分，但不是运算符
原语：

```python
orch.group_by(anchor, *descendants)
orch.zip_by_identity(*ports)
```

`group_by()`使跨粒度扇入变得明确。`zip_by_identity()`通常是
当同粒度端口流入`Map`时是隐式的，但是命名助手很有用
用于诊断和隐式对齐不明确的情况。

## 最小图形图案

大多数多模式数据治理 DAG 应可表示为
少量主题：

### 线性`Map`

```python
b = self.op(a)
```

`a`和`b`共享相同的粒度和记录身份。

### 扇出

```python
b = self.op1(a)
c = self.op2(a)
```

多个分支使用相同的父端口。

### 同粒度扇入

```python
d = self.op(b, c)
```

对于`Map`，所有输入端口必须按记录标识对齐。否则
编译器拒绝调用。

### 扩张

```python
children = self.expand(parent)
```

每个父行都会生成一个子组。运行时存储父关系
而不是向用户公开子 ID。

### 筛选

```python
kept = self.filter(records)
```

保留行保留标识。丢弃的行是正常的业务丢弃，而不是
检疫记录。

### 锚定减少

```python
out = self.assemble(orch.group_by(parent, children))
```

后代按其祖先在`parent`中分组。输出正常
保留锚纹。

### 一般相关

```python
out = self.cluster(records)
```

`self.cluster = orch.Relate(...)`期望用户操作符返回或发出
显式调用本地父关系。

这些图案涵盖了常见的形状：

```text
document -> pages/images/chunks/blocks
         -> enrich / score / OCR / VLM / filter
         -> group back to document or sample
```

## 运行时感知的返回助手

基数帮助程序检测它们是否在运行时调用内运行，使用
调用本地上下文，例如`contextvars.ContextVar`。

外部运行时间：

```python
orch.expanded(groups)
```

返回普通的展平值。

在运行时内部，相同的调用暂时返回一个内部 sidecar：

```text
ExpandedPort(values, offsets)
```

Actor 适配器立即将该 sidecar 转换为端口本地记录
批。它永远不会传递给另一个用户操作员。

这允许自然的独立调试：

```python
pages, meta = Pdf2Pages().run(documents, meta)
```

和更丰富的运行时解释，无需维护两个实现
操作员。

## 静态和动态合约

编译器无法通过执行`run()`发现`orch.expanded(...)`，因为
带有符号引用的编译跟踪`Pipeline.forward()`。真实用户
代码可以解码 PDF、迭代具体值、调用本机库，或者
启动 GPU 推理。

因此，静态模块包装器描述了逻辑调用：

```python
self.pdf_to_pages = orch.Expand(PdfToPages, parent=0)
```

运行时助手描述当前调用的具体关系数据：

```python
class PdfToPages:
    def run(self, documents):
        page_groups = [decode(document) for document in documents]
        return orch.expanded(page_groups)
```

这两个信号有不同的用途：

```text
orch.Expand(PdfToPages, parent=0)
  -> compile-time schema: output port is expanding from input port 0

orch.expanded(page_groups)
  -> runtime data: concrete group sizes and parent mapping
```

运行时必须交叉检查它们：

- 声明的扩展端口必须返回扩展的 sidecar 或等效项
  包装器可以明确规范化的值形状；
- 未声明的端口不得返回扩展的边车；
- 所有关系元数据必须匹配实际输出长度；
- 多输入扩展调用必须声明哪个输入端口是父端口。

包装器不会更改底层用户的`run()`方法。独立式
调试仍然可以实例化并直接调用操作符。

## 提议的用户 API

### 保存记录`1:1`

使用`orch.Map`：

```python
self.ocr = orch.Map(OCR)

texts, confidences = self.ocr(pages, layouts)
```

每个输出端口都保留对齐的输入记录标识。多个输出
可以容纳不同的列，但保持相同的粒度。

### 扩展`1:N`

```python
self.pdf_to_pages = orch.Expand(PdfToPages, parent=0)

pages, page_meta = self.pdf_to_pages(documents, meta)
```

端口有不同的颗粒：

```text
output 0 -> flattened page records with document parents
output 1 -> page metadata at the same page grain, if the wrapper declares
            shared output relation
```

来自一个`Expand`调用的多个输出应该共享一个扩展关系
默认。这支持常见的对，例如`(pages, page_meta)`。独立的
除非明确说明，否则扩展应写为单独的逻辑节点
声明：

```python
pages = self.extract_pages(documents)
figures = self.extract_figures(documents)
```

这避免了在一个节点中默默地混合不相关的偏移量。

### 过滤`1:0/1`

过滤应返回独立期间实际过滤的业务值
在运行时仅携带轻量级选择 sidecar 时进行调试：

```python
self.validate_pages = orch.Filter(ValidatePages)

valid_pages, valid_meta = self.validate_pages(pages, page_meta)
```

在运行时之外，其行为如下：

```python
filtered_pages, filtered_meta = ValidatePages().run(pages, meta)
```

在运行时内，选择位图保留保留行的记录标识
并为被拒绝的行创建轻量级删除元数据。丢弃的有效负载是
不通过 Ray RPC 返回。

由一个助手选择的所有端口共享相同的输出记录集。一个单独的
保留谷物输出必须在该助手之外返回：

```python
return (*orch.selected(keep, pages, page_meta), document_stats)
```

### 减少`N:1`

Reduce 是不同的，因为分组必须在用户代码运行之前发生。使用
当输入关系是跨粒度时，在调用点的`orch.group_by()`：

```python
self.assemble = orch.Reduce(AssembleDocument)

markdown = self.assemble(orch.group_by(documents, texts, captions))
```

`group_by()`的第一个参数是锚粒：

```text
group_by(anchor, descendants...)
```

运行时：

- 锚粒输入保持对齐`1:1`；
- 后代端口按其锚点的祖先独立分组；
- 不相关的输入被拒绝；
- 除非明确声明，否则输出将保留锚记录标识
  否则。

对于简单的情况，包装器可以允许简写：

```python
markdown = self.assemble(documents, page_texts, figure_captions)
```

但仅当声明`orch.Reduce(..., anchor=0)`且所有非锚定输入时
是那个锚的后代。否则需要`group_by()`。

独立调试仍然是普通的Python：调用者已经提供了
分组的后代列。

### 一般`M:N`

任意批量局部转换需要显式关系：

```python
self.cluster = orch.Relate(ClusterChunks)

clusters = self.cluster(chunks)
```

在用户运算符内部，`parent_groups`使用调用本地输入位置，
绝不是全局 ID：

```python
[
    [0, 3],
    [1, 2, 4],
]
```

对于多个输入端口，父组由输入参数键入：

```python
return orch.related(
    samples,
    parents={
        "chunks": chunk_groups,
        "figures": figure_groups,
    },
)
```

命令风格`ctx.emit()`可能仍然是动态控制的高级语法
流，但它必须最终确定为同一批关系，并且不得发出
每条记录的 RPC。

## 统一的物理关系

每个输出端口都标准化为：

```text
PortBatch
  columns or value
  record_ids
  lineage_heads
  parent relation
```

父关系可以使用类似 CSR 的表示：

```text
parent_offsets
parent_indices
parent_ports
```

所有基数都成为特殊情况：

```text
1:1  -> output i has parent input i
1:N  -> several outputs share one input parent
1:0  -> no output is produced for a rejected input
N:1  -> one output has several input parents
M:N  -> arbitrary output-to-input parent sets
```

## API 提供一流的运行时优势

API 不仅是拼写基数的更好方法。一旦每个端口都携带一个
粒度和关系契约，运行时可以使用与
执行控制平面。

一流的制度机制应该有两个好处：

1. 用于调试、可观察性和面向用户的自动沿袭跟踪视图
   错误报告；
2. `1:N`扩展后的关系感知物理重新分批/重新排序以减少
   GPU 气泡。

### 自动谱系跟踪视图

运行时应该将内部血统和父关系转变为可读的
诊断视图：

```text
ErrorTrace
  source item: pdf_path=paper-a.pdf
  logical item: page=17
  failed op: OCR
  grain: page
  upstream path: PdfToImages -> Layout -> OCR
  parent document: paper-a.pdf
  downstream impact: Assemble(markdown for paper-a.pdf)
  action: quarantined / retried / skipped / partial replay
```

用户不应检查全局`record_id`、`path_id`或父边缘表。
相反，运行时应该从源值派生显示标识，
逻辑端口名称和本地序号：

```text
document=paper-a.pdf
document=paper-a.pdf/page=17
document=paper-a.pdf/page=17/block=3
```

这要求每个关系改变操作符保留足够的元数据
人类可读的跟踪重建：

- 源标签列或显示键；
- 其父组内的子序号，除非运算符提供更强的
  业务关键，例如页码；
- 失败的操作员名称和逻辑端口；
- 失败调用的当前粒度；
- 上游母链和下游受影响的锚点。

预期的用户体验是：

```python
result.errors.to_table()
result.trace(error)
result.trace_item(document="paper-a.pdf", page=17)
```

确切的 API 可以改变，但功能应该是设计的一部分：

```text
lineage relation -> item-level diagnosis -> actionable recovery explanation
```

### 关系感知重新批处理和重新排序

`1:N`扩展通常会导致子项数量高度不平衡：

```text
pdf0 -> 2 pages
pdf1 -> 80 pages
pdf2 -> 1 page
pdf3 -> 40 pages
```

如果下游 GPU 阶段保持原始文档形状的批处理布局，那么速度会很快
副本很早就完成，而其他人则处理长文档。`Expand`
关系让运行时展平子颗粒记录，形成密集的子批次，
然后按祖先将它们分组：

```text
documents
  -> Expand(document -> pages)
  -> flatten/rebatch/reorder page records
  -> dense GPU Map stages over pages
  -> group_by(document, page_outputs)
  -> Reduce back to document grain
```

逻辑 DAG 不变。仅物理批次布局发生变化：

```text
logical relation: document -> page -> OCR text
physical layout:  page batches balanced across GPU replicas
restore:          group_by(document, OCR text) using parent relation
```

这是使`1:N`显式化的主要性能回报。父母关系
不仅仅是谱系元数据；它们允许运行时消除气泡
仍在重建原始文档分组。

所需的不变量：

- 重新分批必须保留子序数或稳定的子密钥，以便`Reduce`可以
  恢复确定性秩序；
- 重新排序后的同粒度扇入必须按身份对齐，而不是偶然对齐
  物理位置；
- `group_by(anchor, descendants)`必须容忍丢失、过滤、隔离、
  或根据reduce策略重试子项；
- 关系元数据必须通过调度和副本收集与记录一起移动；
- 有界重新排序范围必须定义锚点的子级何时完成。

有用的重新排序范围：

```text
within one microbatch     simplest and safest
across K microbatches     better load balance with bounded buffering
dataset/global            requires shuffle, barriers, or stateful routing
```

MVP 应从微批次内或有界 K 微批次重新排序开始。
数据集范围的路由是一个单独的物理执行问题。

## 所需类型和 API 更改

### `MicroBatch`

当前`MicroBatch`代表节点的所有输出列，并假设一个
共享行集。它应该：

1. 成为端口本地批量类型；或者
2. 在内部重命名为`PortBatch`。

重要的不变量是：

> 一个端口批次内的所有值共享一个记录集和粒度。

不同的输出端口可能具有不同的长度、身份和沿袭。

### `RuntimeResult`

当前的：

```python
RuntimeResult(batch=MicroBatch(...))
```

方向：

```python
RuntimeResult(ports=(PortBatch(...), PortBatch(...)), ...)
```

兼容性可以提供：

```python
result.batch
```

仅当结果具有一个端口，或所有最终端口共享一个记录集时
并且可以安全地形成柱批次。模糊的混合粒度结果必须要求
显式端口访问而不是默默地对齐不兼容的记录。

### 执行者上下文

执行器当前存储一个`MicroBatch`并为每个输出重复它
投币口：

```text
ctx[node] = tuple(result.batch for each output)
```

它必须存储实际的端口批次：

```text
ctx[node] = result.ports
```

`PipeRef(node, index)`已经识别出输出端口，因此编译后的DAG
参考模型不需要从根本上重新设计。

### `NodeSpec`和`RuntimeNodeSpec`

每个输出都需要一个静态基数合约：

```text
PRESERVE
EXPAND
FILTER
RELATE
```

Reduce 主要是一个输入路由合约：

```text
kind=REDUCE
anchor_input=0, if shorthand calls are allowed
grouped_inputs=(1, 2, ...), if the call site used group_by()
```

编译器从逻辑模块包装器获取这些契约，例如
`orch.Map`、`orch.Expand`、`orch.Filter`、`orch.Reduce`和`orch.Relate`，然后
将它们附加到节点规范。

### 编译器和 AST 命名

现有的 AST 分配：

```python
pages, document_meta = self.pdf2pages(documents, meta)
```

继续提供输出名称。类型注释继续提供
业务类型。

基数元数据是分开的：

```text
AST assignment       -> port names
type annotations     -> business value types
module wrapper       -> static cardinality contracts
call-site helper     -> grouping or explicit alignment intent
runtime helper       -> concrete parent mapping for this invocation
```

不需要`Grouped[T]`执行类型或用户可见的`PipeRef`。

### 逻辑包装器和`RuntimeRayModule`

公共`RuntimeRayModule`构造函数目前专注于物理
执行：

```python
RuntimeRayModule(
    Op,
    replicas=...,
    num_gpus_per_replica=...,
    max_inflight=...,
)
```

新的逻辑包装不应强迫用户重复基数
`RuntimeRayModule`。有两种可能的集成形状：

```python
self.ocr = orch.Map(OCR, replicas=4, num_gpus_per_replica=1)
```

或者：

```python
self.ocr = orch.Map(RuntimeRayModule(OCR, replicas=4, num_gpus_per_replica=1))
```

第一个更符合人体工程学；第二个保持物理执行明确并且
更接近当前的实施。这是一个开放 API 边界决定。
在这两种情况下，编译的节点必须同时携带：

```text
logical contract: Map / Expand / Filter / Reduce / Relate
physical contract: replicas, resources, max_inflight, environment
```

参与者适配器必须独立规范每个返回的端口。

### 发送和复制品收集

对于普通和扩展端口批次，输入调度仍然基于行。

副本结果集合必须独立合并匹配的输出端口：

```text
replica 0 port 0 + replica 1 port 0 -> merged port 0
replica 0 port 1 + replica 1 port 1 -> merged port 1
```

不得假设来自同一副本的不同端口具有相同的行
很重要。

### 扇入

普通的`1:1`扇入仍然保持严格的记录身份对齐。

不同谷物的投入只能通过其合同的运营商来满足
解释一下关系：

- `orch.group_by(anchor, *descendants)`将后代分组到一个锚点；
- `orch.Relate(...)`提供显式关系；
- 未来基于密钥的连接/洗牌路由独立的数据集。

普通操作员在没有此类合同的情况下接收页面和文档端口
必须明确地失败而不是隐式地广播。

### 极端情况和必要的检查

#### 多输入`Expand`

为了：

```python
crops = self.crop(images, layouts)
```

`self.crop = orch.Expand(Crop)`是不明确的。包装器必须声明哪个
输入拥有子记录：

```python
self.crop = orch.Expand(Crop, parent=0)
```

如果多个输入都可以作为父级，那么算子就不再是简单的了
`Expand`;它应该建模为`Relate`。

#### 多输出`Expand`

为了：

```python
images, page_meta = self.pdf_to_images(pdfs)
```

默认值应该是两个输出的一个共享扩展关系。这意味着
`images[i]`和`page_meta[i]`描述相同的页面记录。如果一位操作员
返回具有独立偏移量的页面和图形，它应该被分割
分成两个逻辑节点或声明为高级多重关系`Relate`。

共享关系必须包含子序数或稳定子键。下游
重新分批可能会改变物理顺序，但`group_by(pdfs, images)`必须能够
重建文档级reduce的页面顺序。

#### 十字纹`Map`

为了：

```python
out = self.op(documents, pages)
```

如果`self.op = orch.Map(Op)`，编译器必须拒绝调用，除非输入
端口按身份对齐。它不得隐式广播文档行
页行。

在关系感知重新分批之后，即使相同的粒度输入也可能以不同的方式到达
实物订单。  因此，`Map`扇入必须按记录标识对齐；身体的
仅当已知身份顺序匹配时，位置才是一种优化。

#### 减少速记

为了：

```python
markdown = self.assemble(documents, texts)
```

仅当 `self.assemble = orch.Reduce(Assemble,
锚点=0)`并且所有非锚点输入都是锚点的后代。否则，
用户应该写：

```python
markdown = self.assemble(orch.group_by(documents, texts))
```

当后代在微批次中被物理重新排序时，减少
运行时必须知道锚何时完成。微批量内减少可以使用
局部展开关系。有界或全局重新排序需要水印，
亲本结束标记，或其他完成协议。

减少政策还应该定义如何处理失踪儿童：

```text
fail-open    assemble with missing/error metadata
fail-closed  quarantine the whole anchor
retry-first  run child-grain partial replay before reduce
partial      emit output plus structured warnings
```

#### 过滤与隔离

过滤是有意的数据管理。隔离是执行失败。他们
可能都从健康路径中删除行，但它们需要单独的元数据和
复苏政策。

#### 全节点提交

一次 Actor 调用返回一个结果。具有多个输出端口的节点应该
将其所有端口关系集中在一起。每个端口的部分提交需要
单独的未来提交协议。

### 滴落和检疫

正常过滤和执行失败仍然不同：

```text
drop       -> intentional removal with lightweight metadata
quarantine -> execution failure for one record
```

两者都应默认引用和元数据，而不是嵌入大量内容
RPC 响应中的图像、张量或文档有效负载。

### 失败和重试语义

`BadRecordError(index=...)`保留在提交给的记录的本地
当前运算符调用：

```text
document-grain op -> index identifies a document
page-grain op     -> index identifies a page
reduce op         -> index identifies an anchor group
```

因此，减少失败会隔离或重试受影响的锚定组
默认。它不会隐式识别该组中的一个孩子。精确的
子隔离通常应该发生在上游子粒操作符中。
未来的高级错误可能会携带调用本地子引用，但它
不得公开全局 ID。

拆分并重试必须保留当前的调用粒度：

- 页面批次按页面分割；
- 基于锚点的reduce批次被完整的锚点组分割；
- 扩展运算符由源输入记录重试，而不是部分重试
  产生了孩子的价值观。

一家不断扩张的运营商，希望保留部分成功的孩子
失败的源记录需要显式的`related()`或缓冲的发出协议。
基本的`expanded()`合约对于每个源输入记录都是原子的。

系统级重试必须重放一个节点的完整输出端口集
调用。成功的页面端口和失败的文档端口来自同一个
尝试不得独立提交，除非未来提交协议
明确允许每个端口提交。

### 血统

路径沿袭对于保存操作员历史记录仍然有用，但是
基数变化还需要显式的记录父边缘。

因此，内部模型分离：

```text
execution path lineage
record parent relation
```

两者都对普通用户 API 隐藏，并且可以在以后进行影响分析
和部分重播。

## 什么不会改变

- 用户操作员仍然暴露`run()`。
- 没有 Ray 的情况下，操作符仍然可以被实例化和调试。
- `Pipeline.forward()`仍然是一个正常的商业 DAG。
- `PipeRef(node, index)`仍然是内部输出端口参考。
- 现有的`1:1`运算符不需要超出逻辑的关系帮助器
  `Map`包装器。
- 类型注释仍然是正常的应用程序类型。
- Ray RPC 仍然对每个 actor 调用进行一次调用并输出一个结果。
- 运行时物理设置保留在`RuntimeRayModule`上。
- 现有的`BadRecordError(index=...)`仍然是本地调用，尽管
  表示的颗粒现在可以是文档、页面、块或减少锚点。

## 明确分离问题

该 API 描述了一个参与者中已经存在的记录之间的关系
调用。它本身并不实现：

- 跨微批次分组；
- 全局重复数据删除；
- 分布式连接；
- 重新分区或洗牌；
- 窗口、障碍或数据集范围的状态。

这些功能需要端口关系之上的路由和恢复语义
模型。一旦记录位于同一位置，它们的本地输出关系仍然可以使用
相同的`related()`表示。

## 原型问题尚未解决

当前的提案有意关注本地多粒度 API。这
以下设计问题在实施之前仍然需要明确的决定：

### 包装器和物理运行时边界

用户应该写：

```python
self.ocr = orch.Map(OCR, replicas=4)
```

或者显式地组成逻辑层和物理层：

```python
self.ocr = orch.Map(RuntimeRayModule(OCR, replicas=4))
```

第一个比较友好。第二个使当前的`RuntimeRayModule`
生命周期更容易保存。该决定会影响 API 稳定性、文档以及影响方式
物理设置浮出水面。

### 操作员主体返回形状

对于`Expand`，用户操作符是否应该返回普通嵌套组：

```python
return page_groups, page_meta_groups
```

并让`orch.Expand`标准化它们，或者它们应该始终使用：

```python
return orch.expanded(page_groups, page_meta_groups)
```

前者对用户来说更干净。后者更容易验证和支持
具有较少隐藏约定的独立调试。

### 共享与独立多输出关系

该提案将多输出`Expand`默认为共享关系。我们还需要
罕见的独立多输出扩展的语法，或者这样的策略
运算符必须拆分为多个逻辑节点。

### 减少API表面

`orch.group_by(anchor, *descendants)`是明确且安全的。`Reduce(anchor=0)`
速记简洁。文档应该决定哪一个是推荐的路径
以及只允许哪一种。

### 相关人体工程学

`Relate`是`M:N`的完整性逃生舱口，但写入父索引
列表是低级别的。对于常见情况，我们仍然需要一个用户友好的模式：

- 本地重复数据删除；
- 聚类；
- 通过本地键合并；
- 多模态样本配对。

`ctx.emit()`可能是高级形式，但常见的`Relate`工作负载不应该
强制用户输入逐条记录的命令代码。

### 全局路由边界

关系代数可以表示记录被记录后的局部父边
位于同一地点。它不决定如何进行全局重复数据删除、连接、洗牌、
窗口，或有状态索引路由记录。这些运营商需要单独
物理执行语义。

### 重放和确定性合约

对于沿袭引导的部分重放，每个操作员最终都需要声明
用于确定性、副作用、物化安全和重放策略。这
文档仅定义了使重播范围可计算的关系 API。

## 初始验收工作量

使用多粒度Flash-MinerU形管道：

```python
class MineruPipeline(orch.Pipeline):
    def __init__(self):
        self.pdf_to_images = orch.Expand(PdfToImages, parent=0)
        self.layout = orch.Map(Layout)
        self.ocr = orch.Map(OCR)
        self.assemble = orch.Reduce(Assemble)

    def forward(self, pdfs):
        images, page_meta = self.pdf_to_images(pdfs)
        layouts = self.layout(images)
        texts = self.ocr(images, layouts)
        return self.assemble(orch.group_by(pdfs, texts, page_meta))
```

证实：

- `images`和`page_meta`在具有共享父级的页面粒度上留下一个节点
  与`pdfs`的关系；
- 子序号或稳定子密钥在调度、重新分批和
  收藏;
- 直接操作符调用返回普通列表；
- 编译器知道包装器声明的静态端口协定；
- 运行时验证具体的动态关系元数据；
- 页面失败不会隔离整个文档，除非减少策略
  选择失败关闭；
- 故障跟踪标识源文档、子项、失败的操作员、
  上游路径和受影响的锚；
- 页面粒度重新批处理可减少 PDF 页数不平衡的空闲时间；
- 减少仅将健康页面后代分组到每个文档锚点；
- 空组对归约操作员仍然可见；
- 不同的输出端口在副本之间正确合并；
- 除非明确分组或相关，否则跨粒度`Map`调用会失败；
- 重新分批后的同粒度扇入按记录标识对齐；
- 不会发生每条记录的 Ray RPC；
- lineage 可以追踪 Markdown 到参与的页面和源文档。

## 实施令

1. 引入逻辑模块包装器（`Map`，`Expand`，`Filter`，`Reduce`，
   `Relate`）和编译时基数合约而不改变执行。
2. 添加符号关系助手（`group_by`，可选`zip_by_identity`）和
   编译器检查非法的跨粒度扇入。
3. 添加调用本地返回助手和`Expand`的独立测试，
   `Filter`和`Relate`。
4. 将执行器存储从重复的节点批次转换为真正的每端口批次。
5. 实现具有共享关系的保留和扩展的混合粒度输出
   多输出支持。
6. 添加显示身份元数据和自动项目级跟踪视图
   隔离和运行时错误。
7. 实施轻量级过滤并删除元数据。
8. 物理重新排序后实施基于身份的同粒度扇入。
9. 为`1:N`输出实现微批次内关系感知重新批处理。
10. 通过`group_by`实现基于锚点的缩减路由，包括子路由
    订购和失踪儿童政策。
11. 仅在锚完成后实施有界 K-microbatch 重新分批
    协议已定义。
12. 实现通用批量本地`related()`父映射和/或缓冲发射。
13. 将父关系连接到部分重播和持久沿袭。
