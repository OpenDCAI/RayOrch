"""容器操作 —— 让 DAG 的 batch 单元支持任意「可切片容器」,不止 list/tuple。

RayOrch 原本只把 list/tuple 当可切分 batch(_is_shardable / _submit / _validate_output 硬编码 isinstance list)。
本模块把「切 shard + 拼回」收口成两个鸭子类型 helper,让 pandas.DataFrame / pyarrow.Table / numpy.ndarray
也能当 batch 单元(表 + 多模态 object 统一)。**不硬依赖 pandas/pyarrow/numpy**——按需惰性 import,只在真遇到
该类型时才 import,保持 RayOrch 依赖轻。

设计(hydp-dataflow 四探针坐实,见 rayorch-engine.md §4.5):
  is_sliceable(x)      —— 能否当 batch 切分:list/tuple 或「有 __len__ + 支持 [s:e] 行切/或 .slice」的容器
  uni_slice(x, s, e)   —— 位置切 [s,e):arrow 用 .slice(offset,length),其余用 x[s:e]
  uni_concat(parts)    —— 按 rank 顺序拼回:type→拼接分派(list/tuple/DataFrame/Table/ndarray)
向后兼容:list/tuple 走原路径,旧调用零改动。新增一种容器 = uni_concat 加一支。
"""

from __future__ import annotations

from typing import Any, List, Sequence


def _is_arrow_table(x: Any) -> bool:
    # 鸭子:pyarrow.Table 有 .slice + .num_rows + .schema,且不硬 import pyarrow
    return (hasattr(x, "slice") and hasattr(x, "num_rows")
            and hasattr(x, "schema") and hasattr(x, "column_names"))


def _is_pandas_df(x: Any) -> bool:
    # 鸭子:DataFrame 有 .iloc + .columns + .reset_index,避免硬 import pandas
    return hasattr(x, "iloc") and hasattr(x, "columns") and hasattr(x, "reset_index")


def _is_numpy(x: Any) -> bool:
    return type(x).__module__ == "numpy" and type(x).__name__ == "ndarray"


def is_sliceable(x: Any) -> bool:
    """能否当作 batch 切分单元。str/bytes 不算(它们有 __getitem__ 但不是 batch)。"""
    if isinstance(x, (str, bytes)):
        return False
    if isinstance(x, (list, tuple)):
        return True
    if _is_arrow_table(x) or _is_pandas_df(x) or _is_numpy(x):
        return True
    return False


def uni_slice(x: Any, start: int, end: int) -> Any:
    """按位置切 [start, end)。arrow 用 .slice(offset,length);list/tuple/numpy/DataFrame 用 x[s:e] 行切。"""
    if _is_arrow_table(x):
        return x.slice(start, end - start)
    return x[start:end]


def uni_concat(parts: List[Any]) -> Any:
    """按 rank 顺序拼回。空 → None。按首元素类型分派(惰性 import 对应库)。"""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    x0 = parts[0]

    if isinstance(x0, list):
        out: List[Any] = []
        for p in parts:
            out.extend(p)
        return out
    if isinstance(x0, tuple):
        out_t: tuple = ()
        for p in parts:
            out_t = out_t + tuple(p)
        return out_t
    if _is_pandas_df(x0):
        import pandas as pd
        return pd.concat(parts, ignore_index=True)
    if _is_arrow_table(x0):
        import pyarrow as pa
        return pa.concat_tables(parts)
    if _is_numpy(x0):
        import numpy as np
        return np.concatenate(parts)
    raise TypeError(f"uni_concat: 无法拼接类型 {type(x0).__name__}(非 list/tuple/DataFrame/Table/ndarray)")
