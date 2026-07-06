"""自驱动 source(有状态流式 reader)+ StopIteration 哨兵驱动终止 —— 正确性 / 边界 / 时间账。

**这是什么能力**:一个算子的 ``run`` 自己 ``return next(self._it)`` 从内部迭代器(如
``ray.data.read_parquet(uri).iter_batches()``)流式吐 batch,迭代器枯竭时 ``run`` 抛 ``StopIteration``。
把这样的算子放进 ``CompiledGraph`` 的 **root 节点**(``deps==()``、无 ``input_keys``),``DagExecutor`` 就
**乐观 admit、靠哨兵终止**——不预知 batch 总数、TB 级源不 OOM,多个无依赖 reader 各自 actor **并行读**。

**算子零耦合**:reader 只写 ``return next(self._it)``,不 import 任何引擎符号;``StopIteration → SOURCE_EXHAUSTED``
的转换收在 :class:`RunnerActor`,哨兵识别收在 ``_Scheduler``(见 ``docs/dag_new_pipeline_architecture.md``)。

────────────────────────── API 用法(最小示例) ──────────────────────────
    class StreamReader:                       # 有状态 reader:零引擎耦合
        def __init__(self, uri):
            self._it = iter(ray.data.read_parquet(uri).iter_batches(batch_size=1000))
        def run(self):                         # root 节点 run 无参;枯竭时 next() 自然抛 StopIteration
            return next(self._it)

    r = RayModule(StreamReader, replicas=1, max_inflight=4).pre_init(uri)
    graph = CompiledGraph(
        nodes={"read": NodeSpec("read", r, args=(), kw_args={}, max_inflight=4, num_outputs=1), ...},
        ..., input_keys=())                    # ★ 空 input_keys = 纯自驱动图
    out = DagExecutor(max_batches_inflight=4).execute(graph, {})   # 无需喂 columns
────────────────────────────────────────────────────────────────────────

跑法:``python -m pytest test/test_selfdriven_source.py -q``(``-s`` 看 bench 时间账)。
"""
from __future__ import annotations

import time

import ray

from rayorch.dag_new_pipeline import CompiledGraph, DagExecutor, NodeSpec, PipeRef
from rayorch.ray_module import RayModule, SOURCE_EXHAUSTED, _SourceExhausted


# --------------------------------------------------------------------------- #
# 可 import 的算子(Ray by-reference 还原需模块级可 import;零引擎耦合)
# --------------------------------------------------------------------------- #
class _ListReader:
    """有状态 reader:init 持一个 list 的迭代器,每次 run next() 吐一 batch,枯竭抛 StopIteration。

    真实场景里 ``self._it`` 是 ``iter(ray.data.read_*(uri).iter_batches())``;这里用 list 模拟,
    语义一致(``next()`` + 自然 ``StopIteration``),不依赖 ray.data、测试快且确定。
    """

    def __init__(self, values, sleep_s: float = 0.0):
        self._it = iter(values)
        self._sleep = sleep_s

    def run(self):
        if self._sleep:
            time.sleep(self._sleep)
        return next(self._it)                  # 枯竭 → StopIteration(Python 原生,零引擎耦合)


class _TsReader:
    """记 (enter, exit) 绝对时间戳的 reader —— 证明多 source 真并行(执行区间重叠)用。"""

    def __init__(self, n_batches: int, sleep_s: float):
        self._it = iter(range(n_batches))
        self._sleep = sleep_s

    def run(self):
        t0 = time.perf_counter()
        time.sleep(self._sleep)
        next(self._it)                         # 枯竭即停
        return [[t0, time.perf_counter()]]


class _Double:
    def run(self, x):
        return [v * 2 for v in x]


class _MergePair:
    """fan-in:两路逐元素打包 tuple(看是否对齐/串位)。"""
    def run(self, a, b):
        return [(x, y) for x, y in zip(a, b)]


class _Collect:
    """收两路时间戳区间成 [ra_iv, rb_iv]。"""
    def run(self, a, b):
        return [[a[0], b[0]]]


class _SleepStage:
    """流水线阶段:sleep 固定时长透传。类属性 S 可调(bench 用)。"""
    S = 0.1

    def run(self, x):
        time.sleep(self.S)
        return x


def _cleanup(*modules: RayModule) -> None:
    for m in modules:
        for actor in getattr(m, "actors", []):
            try:
                ray.kill(actor)
            except Exception:
                pass


def _rm(cls, *init, inflight=4):
    return RayModule(cls, replicas=1, max_inflight=inflight, num_outputs=1).pre_init(*init)


def _single_graph(reader_module):
    """[read]→[double]→out,无 input_keys(纯自驱动)。"""
    dmod = _rm(_Double)
    g = CompiledGraph(
        nodes={
            "read": NodeSpec(name="read", module=reader_module, args=(), kw_args={}, max_inflight=4, num_outputs=1),
            "double": NodeSpec(name="double", module=dmod, args=(PipeRef(node="read", index=0),), kw_args={}, max_inflight=4, num_outputs=1),
        },
        topo_order=("read", "double"), deps={"read": (), "double": ("read",)},
        consumers={"read": ("double",), "double": ()},
        graph_outputs=(PipeRef(node="double", index=0),), input_keys=())
    return g, dmod


def _fanin_graph(ra_module, rb_module, merge_cls=_MergePair):
    """a=[ra] b=[rb] → Merge(a,b),两自驱动 source 无依赖(可并行);无 input_keys。"""
    mm = _rm(merge_cls)
    g = CompiledGraph(
        nodes={
            "ra": NodeSpec(name="ra", module=ra_module, args=(), kw_args={}, max_inflight=4, num_outputs=1),
            "rb": NodeSpec(name="rb", module=rb_module, args=(), kw_args={}, max_inflight=4, num_outputs=1),
            "merge": NodeSpec(name="merge", module=mm, args=(), kw_args={"a": PipeRef(node="ra", index=0), "b": PipeRef(node="rb", index=0)}, max_inflight=4, num_outputs=1),
        },
        topo_order=("ra", "rb", "merge"),
        deps={"ra": (), "rb": (), "merge": ("ra", "rb")},
        consumers={"ra": ("merge",), "rb": ("merge",), "merge": ()},
        graph_outputs=(PipeRef(node="merge", index=0),), input_keys=())
    return g, mm


# =========================================================================== #
# 正确性
# =========================================================================== #
def test_single_selfdriven_source_runs_and_stops():
    """单自驱动 source:N 批全产出、数据对、迭代器枯竭即停(不预知 N)。"""
    ray.init(ignore_reinit_error=True, num_cpus=4)
    ra = _rm(_ListReader, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])   # 3 batch
    g, dmod = _single_graph(ra)
    try:
        out = DagExecutor(max_batches_inflight=4).execute(g, {})
        assert out == [[0, 2, 4], [6, 8, 10], [12, 14, 16]]
    finally:
        _cleanup(ra, dmod)
        ray.shutdown()


def test_empty_source_yields_nothing():
    """空 source(reader 立即 StopIteration)→ 空结果,不崩不挂。"""
    ray.init(ignore_reinit_error=True, num_cpus=4)
    ra = _rm(_ListReader, [])
    g, dmod = _single_graph(ra)
    try:
        assert DagExecutor(max_batches_inflight=4).execute(g, {}) == []
    finally:
        _cleanup(ra, dmod)
        ray.shutdown()


def test_inflight_one_no_deadlock():
    """inflight=1:1-ahead admit 门控在最小并发下仍推进,不死锁。"""
    ray.init(ignore_reinit_error=True, num_cpus=4)
    ra = _rm(_ListReader, [[0], [1], [2]], inflight=1)
    g, dmod = _single_graph(ra)
    try:
        assert DagExecutor(max_batches_inflight=1).execute(g, {}) == [[0], [2], [4]]
    finally:
        _cleanup(ra, dmod)
        ray.shutdown()


def test_backward_compat_driver_list_inputs():
    """向后兼容:非空 input_keys(driver 侧喂 list)完全走老路径,行为不变(哨兵/门控不触发)。"""
    ray.init(ignore_reinit_error=True, num_cpus=4)
    dmod = _rm(_Double)
    g = CompiledGraph(
        nodes={"d": NodeSpec(name="d", module=dmod, args=(PipeRef(node="__input__x", index=0),), kw_args={}, max_inflight=4, num_outputs=1)},
        topo_order=("d",), deps={"d": ()}, consumers={"d": ()},
        graph_outputs=(PipeRef(node="d", index=0),), input_keys=("__input__x",))
    try:
        assert DagExecutor(max_batches_inflight=4).execute(g, {"__input__x": [[1, 2], [3, 4]]}) == [[2, 4], [6, 8]]
    finally:
        _cleanup(dmod)
        ray.shutdown()


# =========================================================================== #
# ★ Corner cases:多 source 不等长 / 一路先枯竭另一路在途
# =========================================================================== #
def test_unequal_length_stops_at_shortest_and_aligns():
    """不等长 fan-in(A=5, B=3):停在最短(3);抢先多读的 A[3]/A[4] 被丢弃、产出严格对齐不串位。"""
    ray.init(ignore_reinit_error=True, num_cpus=4)
    ra = _rm(_ListReader, [[i] for i in range(5)])            # A: [0][1][2][3][4]
    rb = _rm(_ListReader, [[100 + i] for i in range(3)])      # B: [100][101][102]
    g, mm = _fanin_graph(ra, rb)
    try:
        out = DagExecutor(max_batches_inflight=4).execute(g, {})
        # 停在短的(3),逐批严格对齐 (A[i], B[i]),A 多读的两批不产出、不串位
        assert out == [[(0, 100)], [(1, 101)], [(2, 102)]]
    finally:
        _cleanup(ra, rb, mm)
        ray.shutdown()


def test_one_source_exhausts_while_sibling_inflight():
    """竞态:rb 立即枯竭(0 批),ra 慢(在途 sleep)。作废 batch 须等在途的 ra 回来才收尾,不崩不挂、无产出。

    这逼出 `_retire_if_drained` 的在途路径:哨兵先到 → batch 入 `_void` 但 `_batch_inflight>0` 不释放 →
    ra 完成回来走 void 分支收尾。若收尾错(提前 pop / KeyError / 挂起)这条会崩或超时。
    """
    ray.init(ignore_reinit_error=True, num_cpus=4)
    ra = _rm(_ListReader, [[i] for i in range(5)], 0.4)   # 慢、在途(第3个 init 参数 = sleep_s)
    rb = _rm(_ListReader, [])                                     # 立即枯竭
    g, mm = _fanin_graph(ra, rb)
    try:
        t = time.perf_counter()
        out = DagExecutor(max_batches_inflight=4).execute(g, {})
        dt = time.perf_counter() - t
        assert out == []                    # 一路空 → 全图 0 产出
        assert dt < 10.0                    # 不挂(宽松上界,真挂会撞 pytest 超时)
    finally:
        _cleanup(ra, rb, mm)
        ray.shutdown()


def test_shorter_source_slow_still_aligns():
    """反向竞态:ra 短(2 批)、rb 长且慢(5 批 sleep)→ 停在 2,rb 抢先多读丢弃、对齐不串。"""
    ray.init(ignore_reinit_error=True, num_cpus=4)
    ra = _rm(_ListReader, [[i] for i in range(2)])
    rb = _rm(_ListReader, [[100 + i] for i in range(5)], 0.2)   # 长且慢(第3个 init 参数 = sleep_s)
    g, mm = _fanin_graph(ra, rb)
    try:
        out = DagExecutor(max_batches_inflight=4).execute(g, {})
        assert out == [[(0, 100)], [(1, 101)]]
    finally:
        _cleanup(ra, rb, mm)
        ray.shutdown()


def test_three_sources_unequal():
    """3-source fan-in 不等长(5/2/4):停在最短(2)。"""
    ray.init(ignore_reinit_error=True, num_cpus=4)
    ra = _rm(_ListReader, [[i] for i in range(5)])
    rb = _rm(_ListReader, [[100 + i] for i in range(2)])
    rc = _rm(_ListReader, [[200 + i] for i in range(4)])

    class Merge3:
        def run(self, a, b, c):
            return [(x, y, z) for x, y, z in zip(a, b, c)]
    mm = _rm(Merge3)
    g = CompiledGraph(
        nodes={
            "ra": NodeSpec(name="ra", module=ra, args=(), kw_args={}, max_inflight=4, num_outputs=1),
            "rb": NodeSpec(name="rb", module=rb, args=(), kw_args={}, max_inflight=4, num_outputs=1),
            "rc": NodeSpec(name="rc", module=rc, args=(), kw_args={}, max_inflight=4, num_outputs=1),
            "merge": NodeSpec(name="merge", module=mm, args=(), kw_args={"a": PipeRef(node="ra", index=0), "b": PipeRef(node="rb", index=0), "c": PipeRef(node="rc", index=0)}, max_inflight=4, num_outputs=1),
        },
        topo_order=("ra", "rb", "rc", "merge"),
        deps={"ra": (), "rb": (), "rc": (), "merge": ("ra", "rb", "rc")},
        consumers={"ra": ("merge",), "rb": ("merge",), "rc": ("merge",), "merge": ()},
        graph_outputs=(PipeRef(node="merge", index=0),), input_keys=())
    try:
        out = DagExecutor(max_batches_inflight=4).execute(g, {})
        assert out == [[(0, 100, 200)], [(1, 101, 201)]]
    finally:
        _cleanup(ra, rb, rc, mm)
        ray.shutdown()


# =========================================================================== #
# 哨兵语义
# =========================================================================== #
def test_sentinel_identity_survives_ray_serialization():
    """哨兵 SOURCE_EXHAUSTED 跨 Ray 序列化后 isinstance 仍 True(按模块路径引用,非对象 identity)。"""
    ray.init(ignore_reinit_error=True, num_cpus=2)

    @ray.remote
    def _echo(x):
        return x
    try:
        got = ray.get(_echo.remote(SOURCE_EXHAUSTED))
        assert isinstance(got, _SourceExhausted)
    finally:
        ray.shutdown()


# =========================================================================== #
# ★ 时间账 bench:多 source 真并行 + 自驱动流水线重叠(带断言)
# =========================================================================== #
def test_multisource_reads_in_parallel():
    """★两个无依赖 reader 各自 actor → 同批 ra/rb 执行区间**重叠**(直接证据,非 wall 差值)。

    直接看执行区间:若两 actor 同时跑,同批 ra/rb 的 [enter,exit] 重叠 ≈ 满 sleep;接力串行则 ≈ 0。
    """
    ray.init(ignore_reinit_error=True, num_cpus=8)
    S, NB = 0.3, 4
    ra = _rm(_TsReader, NB, S)
    rb = _rm(_TsReader, NB, S)
    g, mm = _fanin_graph(ra, rb, merge_cls=_Collect)
    try:
        batches = DagExecutor(max_batches_inflight=4).execute(g, {})
        ivs = [pair for b in batches for pair in b]
        assert len(ivs) == NB
        overlaps = [min(a[1], b[1]) - max(a[0], b[0]) for a, b in ivs]
        n_par = sum(1 for o in overlaps if o > 0.5 * S)
        print(f"\n[bench] 多 source 真并行: {n_par}/{NB} 批 ra/rb 区间重叠 (overlaps={[round(o,2) for o in overlaps]}, S={S})")
        assert n_par == NB, f"多 source 未并行: overlaps={overlaps}"
    finally:
        _cleanup(ra, rb, mm)
        ray.shutdown()


def test_selfdriven_pipeline_overlap():
    """★自驱动 source + 3 阶段线性下游,inflight 流水线重叠显著快于串行。

    信号 >> 噪声的配置:N=40 batch、K=3 阶段、每阶段 S=0.1s。overlap 收益随 N 线性涨(省 ≈(K-1)/K·N·K·S
    量级),actor 冷启是固定常数——N 大到收益(理论省 ≈ 2·N·S = 8s)远压过冷启噪声(~1s)。inflight 上限取
    K+1=4(=流水线深度,不超订 8 CPU)。断言加速 > 1.8×(实测 ~2.4×,理论上限 K=3×)。
    """
    ray.init(ignore_reinit_error=True, num_cpus=8)
    N, S, INFLIGHT = 40, 0.1, 4
    _SleepStage.S = S

    def build(inflight):
        r = _rm(_ListReader, [[i] for i in range(N)], inflight=inflight)
        s1 = _rm(_SleepStage, inflight=inflight)
        s2 = _rm(_SleepStage, inflight=inflight)
        s3 = _rm(_SleepStage, inflight=inflight)
        g = CompiledGraph(
            nodes={
                "r": NodeSpec(name="r", module=r, args=(), kw_args={}, max_inflight=inflight, num_outputs=1),
                "s1": NodeSpec(name="s1", module=s1, args=(PipeRef(node="r", index=0),), kw_args={}, max_inflight=inflight, num_outputs=1),
                "s2": NodeSpec(name="s2", module=s2, args=(PipeRef(node="s1", index=0),), kw_args={}, max_inflight=inflight, num_outputs=1),
                "s3": NodeSpec(name="s3", module=s3, args=(PipeRef(node="s2", index=0),), kw_args={}, max_inflight=inflight, num_outputs=1),
            },
            topo_order=("r", "s1", "s2", "s3"),
            deps={"r": (), "s1": ("r",), "s2": ("s1",), "s3": ("s2",)},
            consumers={"r": ("s1",), "s1": ("s2",), "s2": ("s3",), "s3": ()},
            graph_outputs=(PipeRef(node="s3", index=0),), input_keys=())
        return g, (r, s1, s2, s3)

    g1, m1 = build(1)
    gN, mN = build(INFLIGHT)
    try:
        t = time.perf_counter(); o1 = DagExecutor(max_batches_inflight=1).execute(g1, {}); t1 = time.perf_counter() - t
        t = time.perf_counter(); oN = DagExecutor(max_batches_inflight=INFLIGHT).execute(gN, {}); tN = time.perf_counter() - t
        speedup = t1 / tN
        print(f"\n[bench] 自驱动流水线重叠 (N={N}, K=3阶段, S={S}s): "
              f"serial(inflight=1) {t1:.2f}s → overlap(inflight={INFLIGHT}) {tN:.2f}s | "
              f"加速 {speedup:.2f}× (理论上限3×) | 省 {t1-tN:.2f}s (>> 冷启~1s)")
        assert len(o1) == len(oN) == N
        assert speedup > 1.8, f"跨 batch 未充分重叠: serial {t1:.2f}s vs overlap {tN:.2f}s (加速仅 {speedup:.2f}×)"
    finally:
        _cleanup(*m1, *mN)
        ray.shutdown()
