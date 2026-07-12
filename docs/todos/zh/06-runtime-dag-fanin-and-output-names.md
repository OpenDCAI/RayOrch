# TODO：Runtime DAG 扇入与输出命名

状态：文档级扇入已实现；确定性的直接返回命名
仍待解决。

## 范围决策

近期目标是保留记录的执行：

```text
one healthy input document -> one healthy output document
one failed input document  -> one quarantine record
```

这是记录级别的 `1:1`，异常情况为 `1:0`。这并不意味着一个节点只能有一个 DAG 输入或一个输出列。只要所有列都表示相同且对齐的文档记录，节点便可以消费多个分支并返回多个列。

在此阶段，页面、块、图像和文本跨度仍是文档记录内的嵌套值：

```python
list[document]
list[list[page]]
list[list[block]]
```

页面级调度和任意记录基数变更推迟至多粒度端口 API 中，该 API 记录在
[`09-multi-grain-port-cardinality-api.md`](09-multi-grain-port-cardinality-api.md)。
高级命令式发射功能仍单独记录在
[`07-runtime-emit-api.md`](07-runtime-emit-api.md)。

## 问题 1：路径分歧后的扇入

一个真实的 Flash-MinerU OCR 阶段可能同时消费原始图像和布局结果：

```python
images, meta = self.pdf2image(pdf, meta)
layouts, meta = self.layout(images, meta)
texts, meta = self.ocr(images, layouts, meta)
```

Runtime 按 `row_id` 对齐必需分支。不同的 `path_ids` 是预期行为，并会产生一个内部的多父节点 join 谱系节点。

## 已实现的语义

一次扇入应当：

- 按 `row_id` 对齐必需输入；
- 仅传递存在于每个必需输入分支中的行；
- 保留节点所使用的每条父谱系路径；
- 创建一条引用完整父集合的输出路径；
- 让隔离记录可通过相关输入分支进行追踪；
- 区分已隔离的行与健康且对齐的行；
- 显式拒绝重复或其他存在歧义的行标识。

谱系已从单父边：

```text
path_id -> parent_path_id + op
```

改为多父边：

```text
path_id -> parent_path_ids + op
```

当前 `path_id` 仍是每条记录对应的一个不透明字符串。在扇入时，Runtime 会创建一个内部 join 谱系节点，并将所有不同的上游路径头作为父节点。用户 op 继续只接收普通批列。

此表示方式有意为未来的 emit 输出做准备，同时不向当前 MVP 添加记录基数 API。

## 待完成：直接返回的输出名称

编译器可从以下代码推断出一个可读的输出名称：

```python
markdown = self.markdown(texts, meta)
return markdown
```

但直接返回：

```python
return self.markdown(texts, meta)
```

会回退为诸如 `markdown.out0` 的生成名称。Runtime 调用方随后无法可靠地假定：

```python
result.batch.columns["markdown"]
```

该行为在内部是有效的，但在公共 API 中令人意外。

## 期望方向

为直接返回调用定义确定性的命名规则。候选方案：

- 对单一输出使用 DAG 属性/节点名称：`markdown`；
- 仅对未命名的多输出调用保留 `node.out0`；
- 当语义名称重要时，允许显式的编译期覆盖。

该规则必须区分节点身份和变量身份，并且当同一 `RuntimeRayModule` 实例复用于多个 DAG 节点时保持稳定。

## 验收测试

- 一个 Runtime OCR 节点可以消费来自不同路径、已对齐的 `images`、`layouts` 和 `meta`，且不丢失谱系。
- 已经在一个必需分支中缺失的行不会进入扇入节点。
- 出错的 OCR 行会报告失败节点及其完整的上游历史。
- 必需分支以稳定顺序使用健康 `row_id` 的交集。
- 重复的 `row_ids` 会在 actor 执行前失败。
- `return self.markdown(...)` 会生成已文档化、确定性的最终列名。
- 赋值和直接返回形式会编译为等价的拓扑和类型。
- API 导览可以直接使用真实的 Flash-MinerU 扇入，无需变通方案。
