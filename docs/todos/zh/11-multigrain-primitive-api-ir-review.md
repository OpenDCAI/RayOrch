# Multigrain 原始 API 和 IR 审查

> 历史评审。工作负载分析仍可参考，但旧 IR 类名不再是当前 API。已实现的
> `ExecutionGraph` 与 relation algebra 请见
> [`rayorch/experimental/multigrain/README.md`](../../../rayorch/experimental/multigrain/README.md)。

本文回顾了来自
原始水平。关键问题不在于系统是否可以追踪 DAG。
关键问题是每个面向用户的原语是否都有一个清晰的语义
也就是说，降低到一种规范的 IR 形状，并为未来留出足够的空间
谱系、恢复、调度和优化。

## 审核标准

每个原语都应该通过相同的检查：

- 用户 API：常见情况应该像普通的 Torch 样式模块调用一样。
- 语义契约：基数和关系含义应明确。
- 规范 IR：等效的用户语法应降低到一个标准化的 IR 形状。
- 验证：非法或不明确的情况应尽早拒绝并具有可读性
  诊断。
- 运行时检查：UDF 输出或适配器输出应根据
  当关系效果是动态时声明的关系契约。
- 运行时值：IR 应公开足够的元数据用于沿袭跟踪，
  行级恢复、重新分批、物化和光线降低。
- 扩展点：高级行为应该通过显式扩展原语
  元数据或更通用的原语，而不是通过隐藏的约定。
- 关系证据：图元必须获得足够的静态和动态
  唯一 IR 的关系信息，最好不将 UDF 耦合到
  框架 API。

API 和 IR 有不同的工作。 API 应该很小并且符合人体工程学。这
IR 应该完整、严格且对优化器友好。

本次审查假设一种前端架构：RayOrch 原生的类似 Torch 的 DAG 调用
是主要的创作表面，而包装器、助手、适配器、返回形状、
和返回协议都是相同规范 IR 的语法糖。一个原始的
仅当它可以降低到小的 IR 代数而不添加
前端特定的特殊情况。

## 关系证据审查

该原型使用与 HYDP-dataflow 不同的关系边界，但是
并不意味着 UDF 应该急切地导入 RayOrch API。 HYDP 式运营商可以
通常保持几乎纯粹的 UDF，因为框架可以获得足够的执行
来自表端口、列键和引擎元数据的含义。多粒对象
管道并不总是能够做到这一点：`Expand`、`Filter`和`Relate`创建
仅从普通值无法恢复的细粒度记录关系。

每个原语的规则是优先选择侵入性最小的关系证据
足够的来源：

```text
Preferred evidence sources:
  wrapper declaration
  forward helper
  business values
  nested groups
  boolean masks
  adapter hooks such as relation_fn / mask_fn / key_fn

Escape-hatch evidence:
  local parent references
  role names

Forbidden evidence:
  global record IDs
  lineage heads
  PortBatch construction
  quarantine records
  replay handles
  Ray/runtime internals
```

这将原始的审查变成了一个更尖锐的问题：API 是否询问用户
构建独特的关系感知 IR 所需的最少语义信息？
如果是，则证据源是模型的一部分。如果不是，则原语是
框架内部泄漏，应该重新设计。

## 原始集

最小逻辑集仍然是：

```text
Map       1:1      preserve aligned record identity
Expand    1:N      one parent produces zero or more children
Filter    1:0/1    intentionally drop records while preserving kept identities
Reduce    N:1      group descendants by an anchor and return anchor-grain rows
Relate    M:N      explicitly emit arbitrary local parent-child relations
```

图形助手是语法级关系表达式，而不是完整的用户运算符：

```text
group_by(anchor, *descendants)
zip_by_identity(*same_grain_ports)
```

`Select`应被视为友好的高级 API，可降低为
`Map + Filter + Project`或集成到一个融合的物理操作员中。它不应该添加
新的逻辑关系类型，除非它最终证明在语义上是不同的
来自注释加过滤。

## 地图

用户形态：

```python
self.ocr = orch.Map(OCR)

texts = self.ocr(images)
texts, scores = self.ocr(images, layouts)
```

语义契约：

- 所有输入端口必须具有相同的粒度。
- 输入按逻辑记录标识对齐，而不是按物理行位置对齐。
- 每个输出都保留与对齐输入相同的记录标识和粒度。
- 多个输出表示同一逻辑行上的多个列。

规范红外：

```text
IRNode.kind = MAP
contract.kind = MAP
relation = PRESERVE for every output
input_grains = (G, G, ...)
output_grains = (G, G, ...)
```

为什么 API 很优雅：

- 普通的调用语法符合用户的直觉。
- 同粒度扇入是隐含的，因为它是常见情况。
- 多输出是自然的 Python 解包。

所需验证：

- 拒绝跨粒度输入，除非关系助手或关系原语是
  用过的。
- 如果身份域不兼容，则拒绝同粒度扇入。
- 物理重新排序后，在调用用户之前需要进行身份对齐
  代码。

极端情况：

- 多输出`Map`必须在所有输出之间保留一个共享的身份关系
  输出。
- 更改行数的`Map`声明不正确，应被拒绝
  在运行时或通过形状元数据。
- 具有副作用的`Map`应在`OperatorProperties`中标记，以便恢复
  不会盲目重试。

当前原型状态：

- 为急切的本地执行和符号跟踪而实现。
- IR 记录`PRESERVE`关系、操作员配方、属性和物理
  提示。
- 验证通过检查同粒度扇入。

## 扩张

用户形态：

```python
self.pdf_to_pages = orch.Expand(PdfToPages, parent=0, child_label="page")

pages, page_meta = self.pdf_to_pages(documents, meta)
```

语义契约：

- 一个已声明的父输入拥有生成的子身份。
- 每个父行生成一个长度为零或更大的子组。
- 每个孩子都有稳定的父母关系和有序或稳定的孩子密钥。
- 默认情况下，一个`Expand`的多个输出共享相同的子关系。

规范红外：

```text
IRNode.kind = EXPAND
contract.kind = EXPAND
relation = EXPAND for every output
parent_input = declared parent index
ordinal = CHILD_INDEX or explicit stable key
physical.prefer_rebatch = true by default
```

为什么 API 很优雅：

- 基数在`__init__`中声明一次。
- 前进路径仍然是普通呼叫。
- `parent=...`仅在需要时才是显式的，并防止隐藏的广播规则。

所需验证：

- `parent`必须在范围内。
- 多输出`Expand`必须共享一个关系或显式声明
  多重关系。
- 如果多个输入可以合理地拥有子身份，请使用`Relate`
  而不是猜测。

极端情况：

- 带有侧面输入的`parent=1`必须产生与
  `parent=0`，父索引除外。
- `pages`和`figures`等独立扩展应该是单独的节点
  或未来的多重关系`Relate`。
- 空子组对于`Reduce`完成和重播必须仍然可见
  逻辑。

当前原型状态：

- 为急切的本地执行和符号跟踪而实现。
- 多输出共享关系在 eager 模式下强制执行。
- IR 记录`EXPAND`、`parent_input`、输出端口和重新批处理提示。

## 筛选

首选用户体型：

```python
self.keep_good_pages = orch.Filter(ValidatePages)

good_pages, good_meta = self.keep_good_pages(pages, page_meta)
```

语义契约：

- 输入必须是同粒度且一致的。
- 保存记录可以保留身份。
- 丢弃的记录是业务丢弃，而不是隔离记录。
- 选择掩码是与谱系相关的元数据，即使它没有返回到
  用户作为正常输出。

规范红外：

```text
IRNode.kind = FILTER
contract.kind = FILTER
relation = FILTER for kept outputs
input_grains = (G, G, ...)
output_grains = (G, G, ...)
selection = mask sidecar or explicit predicate output
```

为什么 API 很棘手：

- 在真实的数据管道中，过滤通常还计算分数、原因或
  注解。
- 纯粹的`Filter`对于简单验证来说可以很优雅，但当
  谓词在下游很有用。

推荐的 API 拆分：

```python
self.score = orch.Map(ScorePages)
self.keep = orch.Filter(KeepHighQuality)

scores = self.score(pages)
good_pages = self.keep(pages, scores)
```

和一个方便的 API：

```python
self.select_good_pages = orch.Select(ScoreAndKeep)

good_pages, scores = self.select_good_pages(pages)
```

降低：

```text
Select
  -> Map(annotation / score)
  -> Filter(mask)
  -> Project(kept annotations)
```

或者，优化后：

```text
Map + Filter + Project
  -> one fused physical Select stage
```

所需验证：

- 丢弃的行不得报告为错误。
- 隔离仍然保留用于意外故障。
- 下游`group_by`必须区分“父母没有养孩子”和
  “父母从未被处理过”。

当前原型状态：

- `orch.Filter`是为了急切的本地执行和符号跟踪而实现的。
- `orch.Select`作为高级 API 实现，可降低至
  符号 IR 中的`Map + Filter + Project`。
- 测试涵盖保留身份保存、业务下降与隔离以及
  规范降低。

## 减少

用户形态：

```python
self.assemble = orch.Reduce(AssembleDocument)

markdown = self.assemble(orch.group_by(documents, texts, page_meta))
```

语义契约：

- 第一个分组参数是锚点。
- 后代端口按其与锚点的祖先关系进行分组。
- 输出通常返回到锚定颗粒。
- 失踪后代遵循明确的失踪儿童政策。

规范红外：

```text
IRNode.kind = REDUCE
contract.kind = REDUCE
relation = REDUCE for every output
anchor = input_refs[0]
grouped = true
parent_input = 0
missing = fail_open / fail_closed / partial / retry_first
```

为什么 API 很优雅：

- `group_by(anchor, descendants...)`使横纹扇入清晰可见
  重要的是。
- 用户不操作行 ID 或父子映射。
- 一旦分组表达式是，该操作读起来就像正常的模块调用
  提供。

所需验证：

- 拒绝未分组的`Reduce`，除非启用了精心设计的速记。
- 验证每个后代都有一条已知的到锚点的祖先路径。
- 除非明确声明，否则验证输出颗粒是锚颗粒。

极端情况：

- 故障打开程序集对于部分文档很有用，但策略必须在
  IR所以恢复和评估就明白了。
- 在用户减少之前，重新排序的子项必须按序数或稳定键排序
  代码可以看到它们。
- 除非声明，多输出`Reduce`应该共享相同的锚关系
  否则。

当前原型状态：

- 通过显式的`group_by`实现。
- 原型拒绝直接的`self.reduce(documents)`误用并出现可读错误。
- 验证通过检查分组减少和锚合同。

## 涉及

潜在用户形态：

```python
self.cluster = orch.Relate(ClusterChunks)

clusters = self.cluster(chunks)
```

或对于多个输入：

```python
self.match = orch.Relate(MatchImagesAndCaptions)

pairs = self.match(images, captions)
```

语义契约：

- 用户操作符发出新记录以及显式的本地父引用。
- 输出可以与来自一个或多个输入的零个、一个或多个父级相关
  端口。
- 调用中的基数是任意的：M:N、重复数据删除、聚类、合并、
  对生成，或图构建。

规范红外：

```text
IRNode.kind = RELATE
contract.kind = RELATE
relation = RELATE
parents = explicit input refs
relation_schema = parent ports, local ids, optional role names
```

为什么 API 是最难的：

- 它是所有不干净的 1:1、1:N 或 N:1 的逃生口。
- 这里太多的便利可能会隐藏昂贵的连接或不明确的全局
  匹配。

推荐约束：

- MVP`Relate`应该只表达调用本地关系。
- 全局连接、洗牌和跨批次匹配应该是单独的物理连接
  具有显式分区或关键语义的规划功能。

所需验证：

- 每个发出的父引用必须指向调用中可见的行。
- 当多个父端口具有相同的粒度时，应命名关系角色。
- 下游`Reduce`必须知道关系是有序的、无序的还是
  多父母。

当前原型状态：

- `orch.Relate`是为了符号跟踪而实现的。
- IR 记录`RELATE`、输出颗粒、输入参考和可选的关系角色。
- 热切和被动 IR 执行支持三个关系证据层
  （参见[`12-relation-model-three-tiers.md`](12-relation-model-three-tiers.md)）：
  1. 声明式`on={role: field}`键连接（内部等连接；纯数据
     出处；常见的跨分支情况，`forward`中没有关系代码）；
  2. by-ref`relation_adapter="pkg.mod:fn"`（点路径，在执行时解析，
     可序列化）对于任意非平等关系；
  3. live`relation_fn`供本地即时使用（未序列化到 IR 中）。
- 本地 MVP 将调用本地父引用存储在`PortBatch.relations`中；
  `on=`key-join 另外合并了两个分支的祖先，因此下游
  `Reduce`可以由仅携带一根分支的祖先重新组合。运行时检查
  拒绝超出范围/未声明的父引用。分布式/全局加入和
  外连接语义仍然是未来的工作。

## group_by 助手

用户形态：

```python
markdown = self.assemble(orch.group_by(documents, texts, page_meta))
```

语义契约：

- 这不是数据操作员。
- 它创建一个关系表达式：后代应该由祖先路由
  与锚点的关系。
- 它必须降低到`REDUCE`输入路由元数据，而不是单独的物理
  阶段，除非通行证决定实现组。

规范红外：

```text
REDUCE node:
  inputs = (anchor, descendants...)
  grouped = true
  anchor = input_refs[0]
```

为什么 API 很优雅：

- 它仅出现在隐式跨粒度扇入的调用站点上
  危险的。
- 它避免将内部 ID 暴露给用户。

所需验证：

- 后代必须与锚有血统。
- 应拒绝多个祖先路径或需要显式路径选择器。

当前原型状态：

- 在急切的本地执行和符号跟踪中实现。

## zip_by_identity 帮助程序

潜在用户形态：

```python
texts = self.ocr(orch.zip_by_identity(images, layouts))
```

或者更可能是隐含的：

```python
texts = self.ocr(images, layouts)
```

语义契约：

- 端口具有相同的粒度并共享相同的逻辑身份域。
- 物理行顺序可能不同；对齐应该通过记录标识进行。

规范红外：

```text
MAP node:
  inputs = same-grain ports
  relation = PRESERVE
  alignment = identity
```

推荐：

- 对于常见的同粒度`Map`扇入，请保持隐式。
- 稍后添加显式`zip_by_identity()`仅用于诊断、消歧、
  或用户控制的对齐策略。

## 规范化规则

编译器应该将等效的用户表达式标准化为一个 IR 形状：

- `Map`同粒度扇入始终与`PRESERVE`成为一个`MAP`节点
  关系和身份一致性。
- `Expand`始终为每个简单扩展关系准确记录一个父输入。
- 多输出`Expand`默认共享一个关系。
- `Reduce(group_by(anchor, ...))`总是成为一个`REDUCE`节点，其锚点为
  `input_refs[0]`。
- 优化前，`Select`降低至`Map + Filter + Project`。
- 优化器融合改变物理节点或物理提示，而不是逻辑
  血统契约。
- 物化是对端口或逻辑边界的注释，而不是新的
  用户可见的数据依赖性。

这些规则使 IR 足够独特以用于测试：如果两个 API 含义相同
事情，他们的标准化`MultigrainIR.to_dict()`应该匹配，除了名字
以及来源出处。

## 优先角落案例测试

原型中已经涵盖了：

- `Map`拒绝跨粒度扇入。
- `Expand`支持非零`parent`。
- 多输出`Expand`记录共享关系。
- `Reduce`需要`group_by`。
- IR 暴露部门、消费者、关系合同、物理暗示，以及
  物化注释。

现在涵盖（之前是“下一个”）：

- `Filter`保留保留的身份并将业务下降与隔离区分开
  （`test_upper_primitives.py`）。
- `Select`降低为规范的`Map + Filter + Project`（`test_dummy_e2e.py`，
  `test_upper_primitives.py`）。
- `Relate`发出显式的本地父引用，加上`on=`key-join 和
  通过参考适配器（`test_relate_key_join.py`，`test_upper_primitives.py`）。
- 在 LPT 跨分片重新排序下，谱系是不变的，包括。误隔离
  （`test_lineage_under_parallelism.py`）。

还需要补充一下：

- `Reduce`验证锚点的后代血统（显式阴性测试）。
- 物理重新排序后（显式测试），同粒度扇入按身份对齐。
- 空的子组和全部过滤的父组对下游仍然可见
  完成/恢复。
- 多个祖先路径需要显式的路径选择。
- 规范化等价：具有相同含义的不同前端糖
  产生相等的标准化`to_dict()`。

## 目前的判决

当前类似火炬的包装器 API 仍然是正确的主要路径：

```python
images = self.pdf_to_images(pdfs)
layouts = self.layout(images)
texts = self.ocr(images, layouts)
markdown = self.assemble(orch.group_by(pdfs, texts))
```

它是用户友好的，因为普通的转换看起来很普通，并且关系
助手仅出现在图表不明确的地方。 IR 也是
朝着正确的方向前进：它已经记录了grain、关系类型、父级
输入、输出规格、操作员配方、属性、物理提示以及
物化。

剩余的设计风险集中在两个方面：

- `Filter/Select`：API必须避免强迫用户分割自然
  得分然后过滤逻辑成尴尬的样板，同时仍然降低到
  规范关系感知 IR。
- `Relate`：API 必须足够强大，能够进行 M:N 数据治理，而无需
  成为隐式连接/洗牌系统。

这两个原语应该驱动下一个原型迭代。
