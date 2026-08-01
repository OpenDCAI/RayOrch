"""V3 各层共享的轻量公共合同。

该模块位于 API、Arena、Execution 和 Worker 之下，只包含异常类型与稳定 sentinel，
不依赖 DAG 或任何运行时组件，用于避免底层模块反向导入 ``api.py``。
"""

from __future__ import annotations


class ExecutionError(RuntimeError):
    """表示一次 run 或 bounded Arena 在控制层失败。"""


class BadRecordError(Exception):
    """用户显式把 UDF 错误归因到物理 batch 中某一行。

    `index` 只用于 Worker 将错误映射回 AttemptToken；最终语义归因仍然是 GrainId。
    """

    def __init__(self, message: str, *, index: int) -> None:
        """保存错误信息并校验 batch 行号非负。"""

        super().__init__(message)
        if index < 0:
            raise ValueError("bad record index must be non-negative")
        self.index = index


class _Missing:
    """可 pickle 的 optional-input 缺失 sentinel。

    使用独立 sentinel 而不是 `None`，因为 `None` 可能是合法业务值。MISSING 只允许作为
    Worker UDF 输入，不能作为 UDF 输出或写入 ItemTable/ValueTable。
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """返回稳定、易读的调试表示。"""

        return "MISSING"

    def __reduce__(self):
        """保证跨 Ray 序列化后仍恢复为进程内单例。"""

        return (_missing_singleton, ())


def _missing_singleton() -> "_Missing":
    """返回当前进程唯一的 MISSING 对象，供 pickle 反序列化使用。"""

    return MISSING


MISSING = _Missing()

