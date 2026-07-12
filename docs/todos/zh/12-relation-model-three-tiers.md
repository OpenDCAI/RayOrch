# 关系表征的三档模型(Relation Model: Three Tiers)

## 目标

用户在表达数据间的父子 / 多对多关系时,心智负担要尽量小,`forward()`
要像 torch 一样干净。本文把关系表征分成三档,把"重语义"压到几乎不出现,
并对标 VERL 的优雅设计确认这套分层是可接受的。

核心结论:

- **80% 的关系根本不需要用户写关系代码**(Expand/Reduce 血缘全自动)。
- **~19% 的真跨分支 join** 只需在 `__init__` 写**一行声明** `on={role: field}`。
- **<1% 的任意非等值关系** 才用 by-ref adapter 逃生口。
- 三档的 `forward()` 写法**完全一致**(都是普通的 torch 式调用),差异全压在
  `__init__` 的一行声明里 —— 这正是 VERL "重语义在声明处、调用点干净" 的同构。

> 反面教训:如果把 adapter(第③档)讲成主路径,那就是设计失败。它只能是罕见逃生口。

## 对标 VERL:抽出两条优雅原则

1. **重语义挂在"定义处",调用点保持顺序式干净。** VERL 用
   `@register(dispatch_mode=…)` 把分发/收集语义挂在方法定义上,用户在驱动代码里
   写 `actor.generate(data)` 这种普通调用,不手写 index 数学。
2. **关系靠"随行的对齐键",不靠代码。** VERL 的 `DataProto` 里同一 batch index =
   同一样本,分组/聚合是"按某列 group",而不是手工连线。

我们对这两条同构:关系语义挂在 operator 的 `__init__` 声明处;跨分支 join 靠声明
join 键(等值连接),而不是让用户在 `forward` 里手工连线。

## 第①档:纯 Expand / Reduce —— 零关系代码(已实现)

**场景**:一份 pdf 拆成多页、逐页 OCR、再按原顺序拼回。父子关系天然由 Expand
产生,Reduce 按 anchor 自动重组,用户全程不碰关系。

```python
class Parse(orch.Pipeline):
    def __init__(self):
        self.split = orch.Expand(PdfToPages)         # 1 pdf -> N pages
        self.ocr   = orch.Map(Ocr)                   # N pages -> N texts(同粒度)
        self.merge = orch.Reduce(Assemble, anchor=0) # N texts -> 1 doc

    def forward(self, pdfs):
        pages = self.split(pdfs)
        texts = self.ocr(pages)
        # anchor=pdfs, descendants=texts;血缘 / 顺序框架自动补
        return self.merge(orch.group_by(pdfs, texts))
```

用户写的只有算子逻辑。**关系代码:0 行。** page 属于哪个 pdf、第几页、失败落在
哪一页,全自动。

实现依据:`Expand._make_outputs` 自动填 `ancestors[parent.name]=parent_id` 与
`ordinals[parent.name]=child_index`;`Reduce._groups_for` 读 `ancestors`/`ordinals`
自动重组并按 ordinal 排序。见 `rayorch/experimental/multigrain/expand_reduce.py`。

## 第②档:真跨分支 join —— 重语义挂在 `__init__`(VERL 式)

**场景**:pages 和 figures 是**两条独立分支**(figure 不是 page 直接 Expand 出来
的,而是另一个抽取器产出,自带 `page_id`)。需要把 figure 连回它所属的 page ——
这是跨分支 join,必须声明 join 键。

```python
class ParseWithFigs(orch.Pipeline):
    def __init__(self):
        self.split   = orch.Expand(PdfToPages)
        self.ocr     = orch.Map(Ocr)
        self.extract = orch.Expand(ExtractFigures)         # 另一条分支:pages -> figures(带 page_id)
        # 重语义在声明处:按 page 的 page_id 把两路对齐
        self.link    = orch.Relate(LinkFigs, on={"page": "page_id"})
        self.merge   = orch.Reduce(Assemble, anchor=0)

    def forward(self, pdfs):
        pages = self.split(pdfs)
        texts = self.ocr(pages)
        figs  = self.extract(pages)
        # forward 依旧 torch 干净:一次普通调用,不写任何 index / 键的连线
        page_figs = self.link(pages, figs)
        return self.merge(orch.group_by(pdfs, texts, page_figs))
```

跟①的差别:多了 `self.link = orch.Relate(..., on=...)` **一行声明**;`forward` 里
`self.link(pages, figs)` 和普通算子调用长得一样。`on={"page": "page_id"}` 等价于
SQL `JOIN ON page_id` —— 这已经是 join 的最小诚实表达,再简就得靠"猜键",那是
静默错连的坑。

设计要点:

- `on={role: field}` 是**纯数据**,静态可序列化,passive IR 不破,优化器可规划
  join 策略(hash join / broadcast)。
- 内部 ID 翻译永远在框架侧;用户只提供字段名,不碰 record_id。
- 可选糖(默认关闭):两侧若都带同名字段,可省略 `on=` 按同名键自动连。默认
  仍要求显式 `on=`,因为隐式连键是 footgun。

状态:**已实现**。`Relate(op, on={role: field})` 走 `_make_key_join_batch`:按声明
字段建索引做 inner equi-join(缺键的记录被丢弃),对每个匹配组合调用
`op.run(by_role_dict)`,并把两侧血缘(ancestors / ordinals / lineage)合并进关系行。
`on` 以纯数据存入 recipe provenance,IR 可 pickle 往返、可被 executor 复原执行。
见 `rayorch/experimental/multigrain/relate.py` 与
`test/experimental/multigrain/test_relate_key_join.py`。

## 第③档:任意非等值关系 —— by-ref adapter 逃生口(极少见)

**场景**:关系不是等值连接(比如"每个 figure 关联到坐标最近的 3 个文本块"这种
几何 / 模糊匹配),无键可 join。此时给一个**可导入的纯函数**(只吐局部下标,不碰
内部 ID)。

```python
# pkg/adapters.py —— 独立、可序列化、可 import 的纯函数
def link_nearest(raw):
    figs, blocks = raw["fig"], raw["block"]
    out = []
    for i, fig in enumerate(figs):
        for j in nearest_k(fig, blocks, k=3):          # 用户自定义的任意匹配
            out.append((make_pair(fig, blocks[j]), {"fig": i, "block": j}))
    return out                                          # list[(value, {role: local_index})]
```

```python
class ParseFuzzy(orch.Pipeline):
    def __init__(self):
        self.split  = orch.Expand(PdfToPages)
        self.figs   = orch.Expand(ExtractFigures)
        self.blocks = orch.Expand(ExtractBlocks)
        # 逃生口:by-ref dotted path,不是闭包 —— IR 仍可序列化
        self.link   = orch.Relate(LinkNearest,
                                  roles=("fig", "block"),
                                  relation_adapter="pkg.adapters:link_nearest")

    def forward(self, pdfs):
        pages  = self.split(pdfs)
        figs   = self.figs(pages)
        blocks = self.blocks(pages)
        return self.link(figs, blocks)                  # forward 仍然一行,干净
```

跟②的差别:声明处从 `on={...}`(声明式键)换成 `relation_adapter="pkg:fn"`
(命令式引用);`forward` 写法完全一样。多出来的重量只在那个独立 adapter 文件里,
且**极少触发**。

设计要点 / 纪律:

- adapter 必须是 **by-ref dotted path**(可导入函数),**禁止内联 lambda / 闭包**
  进 IR —— 否则 passive IR 不可序列化。
- adapter 只说 `{role: local_index}`(局部下标),**禁止碰内部 record_id**;
  local_index → ParentRef/record_id 的翻译永远在框架侧
  (见 `relate.py:_make_relation_batch`)。
- 运行时契约校验已就位:role 必须声明、local_index 必须在界内、长度必须对齐
  (`relate.py`)。

状态:**已实现**。新增 `relation_adapter="pkg.mod:fn"` 参数,`_resolve_adapter`
在 execute 时 `import` 回可调用对象;dotted path 以纯字符串存入 provenance,IR
可序列化往返。旧的 `relation_fn`(活对象)保留作本地即时用,不进 IR。
见 `test/experimental/multigrain/relate_adapters.py`(可导入 adapter)与
`test_relate_key_join.py::test_relate_adapter_by_ref_resolves_dotted_path`。

## 三档对照

| 档 | 场景 | `__init__` 增量 | `forward` | 用户写的关系代码 | 触发频率 | 状态 |
|---|---|---|---|---|---|---|
| ① Expand/Reduce | 拆分后按血缘归并 | 无 | `self.split(pdfs)` / `group_by` | **0 行** | ~80% | 已实现 |
| ② `on=` key-join | 跨分支等值连接 | `Relate(..., on={role: field})` 一行 | 普通调用 | 1 行声明 | ~19% | 已实现 |
| ③ by-ref adapter | 任意非等值关系 | `Relate(..., relation_adapter="pkg:fn")` | 普通调用 | 一个独立纯函数 | <1% | 已实现 |

三档的 `forward()` 写法完全一致(都是 torch 式普通调用),差异全压在 `__init__`
的一行声明里。

## 红线(所有档共用)

1. **annotation 只装死数据**:`on={role: field}` 或 `relation_adapter="pkg:fn"`,
   都是纯数据 / 纯字符串,不装活对象。passive IR 可序列化、可下沉。
2. **UDF 保持框架无关**:算子只返回 raw 值,不碰关系、不碰内部 ID。
3. **内部 ID 翻译永远在框架侧**:用户 / adapter 只说字段名或局部下标。
4. **优先零代码 → 声明式 → 命令式**:能自动就自动,能声明就别命令式。

## 下沉到 HYDP 注入模型

HYDP 的 annotation 袋里带的要么是 `on={role: field}`(纯数据),要么是
`relation_adapter="pkg:fn"`(纯字符串)。两者都是死数据,passive IR 不破;
RayOrch lowering 时:key-join → 生成 RELATE 节点内建 join;adapter → resolve 引用。
运行时证据出来后,由 `_make_relation_batch` 的界内 / role / 长度校验充当动态契约检查。

## 待办

- [x] 第②档:给 `Relate` 加 `on={role: field}` 声明式 key-join 路径。
- [x] 第③档:新增 `relation_adapter="pkg:fn"` dotted-path,lower/execute 时 resolve。
- [x] 跑通 `pages ↔ figures by page_id` 的 M:N 例子(含 join→reduce 按 pdf 归并、
      pickle 往返执行)以实锤第②档。
- [ ] 可选糖:两侧同名字段时允许省略 `on=`(默认关闭,显式开启)。
- [ ] outer join / 左连接语义(当前仅 inner equi-join)。
- [ ] 把 M:N 关系接入 Flash-MinerU 类负载做 rebalancing 收益测量(见下)。

## 下一步:对标 Flash-MinerU 负载的关系 + 重排联合优化

现有 `bench_gpu_mineru.py` / `bench_gpu_complex.py` 只测了 Expand→Map→Reduce 的
1:N 长尾重排(LPT 消空泡)。把第②档 key-join 接进去后,可测一个更贴近 Flash-MinerU
的负载:`pdf → pages(1:N) → {ocr 文本分支, figure 抽取分支}`,两分支用
`on={"page": "page_id"}` 连回页,再按 pdf 归并。观测点:

- join 产出的关系行数据依赖(每页 figure 数长尾),会不会引入新的 stage 空泡;
- LPT 重排能否同时覆盖"OCR 分支"和"figure 分支"的不均衡;
- 关系行的血缘在跨分支 join + 重排下是否仍然可反查(复用
  `test_lineage_under_parallelism.py` 的不变量思路)。
