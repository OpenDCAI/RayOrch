"""TableFormer 表格阶段的保守 batching 作业层。

这个模块不导入 Docling、torch 或任何模型实现；它只规划已经准备好的表格作业，并将
实际执行交给调用方注入的 callable。当前 TableFormer decoder 的已知执行语义是
``batch=1``。因此，本辅助层**不宣称能带来加速**：只有调用方提供并验证了真实的
batch kernel 后，才会调用 ``batch_callable``。即使提供了 batch kernel，默认也会
逐 job 执行 reference 路径作 shadow 对比，语义不一致时回退到该 job 的 reference
结果。

``EncoderBatchedKernel`` 与其 adapter 只定义未来可验证的 encoder 批接口，不调用真实
Docling 或 torch，也不把它误当作已经支持 batch 的 TableFormer decoder。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Literal, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class TableJob:
    """一个表格推理作业及其稳定恢复位置。

    ``page``、``table``、``order`` 定义输出恢复键。``payload`` 完全由调用方解释，
    本模块既不会读取其图像内容，也不会修改、复制或序列化它。
    """

    page: int
    table: int
    order: int
    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def page_index(self) -> int:
        """兼容使用 ``*_index`` 命名的调用点。"""

        return self.page

    @property
    def table_index(self) -> int:
        """兼容使用 ``*_index`` 命名的调用点。"""

        return self.table

    @property
    def order_key(self) -> tuple[int, int, int]:
        """返回 page/table/order 的确定性恢复顺序。"""

        return (self.page, self.table, self.order)


@dataclass(frozen=True, slots=True)
class TableBatch:
    """同一兼容 bucket 中、保持输入相对顺序的一组表格作业。"""

    bucket_key: Hashable | None
    jobs: tuple[TableJob, ...]

    @property
    def key(self) -> Hashable | None:
        """提供简短的 bucket key 别名。"""

        return self.bucket_key


def metadata_bucket_key(job: TableJob) -> Hashable | None:
    """读取调用方显式标注的 bucket key；未标注时统一使用 ``None``。"""

    return job.metadata.get("batch_key")


class TableBatchPlanner:
    """按调用方给出的兼容 key 稳定分桶并可限制单 batch 大小。

    planner 不猜测 payload 的 shape、模型配置或 decoder 能力。相同 key 仅表示调用方
    认为这些作业可交给同一个 *候选* batch kernel；是否真的能执行仍由执行器处理。
    bucket 的排列按 key 首次出现的顺序，bucket 内以及被切分后的作业顺序均保持稳定。
    """

    def __init__(
        self,
        bucket_key: Callable[[TableJob], Hashable | None] = metadata_bucket_key,
        *,
        max_batch_size: int | None = None,
    ) -> None:
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError("max_batch_size 必须为正数或 None")
        self.bucket_key = bucket_key
        self.max_batch_size = max_batch_size

    def plan(self, jobs: Sequence[TableJob]) -> list[TableBatch]:
        """稳定分桶；此方法不按 page/table 排序原始输入。"""

        buckets: dict[Hashable | None, list[TableJob]] = {}
        for job in jobs:
            key = self.bucket_key(job)
            try:
                hash(key)
            except TypeError as error:
                raise TypeError("TableBatchPlanner 的 bucket key 必须可 hash") from error
            buckets.setdefault(key, []).append(job)

        batches: list[TableBatch] = []
        for key, bucket_jobs in buckets.items():
            size = self.max_batch_size or len(bucket_jobs)
            for start in range(0, len(bucket_jobs), size):
                batches.append(TableBatch(key, tuple(bucket_jobs[start : start + size])))
        return batches


@dataclass(frozen=True, slots=True)
class TablePrediction:
    """表格推理的语义值及可选审计元数据。

    shadow 比较只比较 ``value``，不会因为耗时、日志、原始对象等 ``metadata`` 不同而
    将两个语义相同的预测判为不一致。
    """

    value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def semantic_value(self) -> Any:
        """明确返回用于语义比较的预测值。"""

        return self.value


TablePredictionLike = TablePrediction | Any


def coerce_table_prediction(value: TablePredictionLike) -> TablePrediction:
    """把裸语义值统一封装为 :class:`TablePrediction`。"""

    return value if isinstance(value, TablePrediction) else TablePrediction(value=value)


@dataclass(frozen=True, slots=True)
class TablePredictionComparator:
    """只针对 :class:`TablePrediction` 语义值的可注入比较器。"""

    semantic_equal: Callable[[Any, Any], bool] = lambda expected, actual: expected == actual

    def equivalent(self, expected: TablePrediction, actual: TablePrediction) -> bool:
        """比较 prediction value；比较器抛错或不给出真值时视为不一致。"""

        try:
            return bool(self.semantic_equal(expected.value, actual.value))
        except Exception:
            return False


# 更短的名字便于调用方在注入执行器时使用。
TableComparator = TablePredictionComparator


class TableReferenceCallable(Protocol):
    """原有 batch=1 表格路径的 reference callable 协议。"""

    def __call__(self, job: TableJob) -> TablePredictionLike:
        """对一个 job 返回其语义预测。"""


class TableBatchCallable(Protocol):
    """经过验证的表格 batch kernel 的适配协议。

    callable 接收同一 :class:`TableBatch` 内的 job 序列，并必须返回同长度的结果序列，
    顺序与输入完全一致。
    """

    def __call__(self, jobs: Sequence[TableJob]) -> Sequence[TablePredictionLike]:
        """执行候选 batch kernel。"""


class EncoderBatchedKernel(Protocol):
    """未来 encoder batching kernel 的极小接口，不代表 decoder 已支持 batching。"""

    def __call__(self, payloads: Sequence[Any]) -> Sequence[TablePredictionLike]:
        """对按输入顺序排列的 payload 返回同顺序的候选预测。"""


class EncoderBatchedKernelAdapter:
    """把 payload 级 ``EncoderBatchedKernel`` 转为 job 级 ``TableBatchCallable``。

    adapter 只负责保持 job/payload 顺序并转交 kernel。它不会导入 torch，不会拼接
    tensor，也不会执行或声称支持 TableFormer decoder batching。
    """

    def __init__(self, kernel: EncoderBatchedKernel) -> None:
        self.kernel = kernel

    def __call__(self, jobs: Sequence[TableJob]) -> Sequence[TablePredictionLike]:
        """提取 payload 并保留其与 jobs 完全相同的顺序。"""

        return self.kernel([job.payload for job in jobs])


def _prediction_sequence(
    value: Any,
    *,
    expected_size: int,
) -> list[TablePrediction]:
    """校验 batch 返回长度，并把每项统一为 ``TablePrediction``。"""

    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise TypeError("table batch callable 必须返回与输入等长的结果序列")
    if len(value) != expected_size:
        raise ValueError(
            f"table batch callable 返回 {len(value)} 项，期望 {expected_size} 项"
        )
    return [coerce_table_prediction(item) for item in value]


@dataclass(frozen=True, slots=True)
class TableExecutionResult:
    """一个 job 的最终结果及 batch/shadow 审计信息。"""

    job: TableJob
    prediction: TablePrediction
    bucket_key: Hashable | None
    source: Literal["reference", "batch", "shadow_fallback"]
    batched: TablePrediction | None = None
    reference: TablePrediction | None = None
    batch_error: str | None = None

    @property
    def used_fallback(self) -> bool:
        """最终结果是否没有采用候选 batch 输出。"""

        return self.source != "batch"


def restore_table_order(
    results: Sequence[TableExecutionResult],
) -> list[TableExecutionResult]:
    """按 page/table/order 恢复稳定的文档输出顺序。"""

    return sorted(results, key=lambda result: result.job.order_key)


class TableBatchExecutor:
    """保守地运行表格 batch，并默认以 reference 路径做严格 shadow 回退。

    当 ``batch_callable`` 为 ``None`` 时，执行器只调用 ``reference_callable``，不作任何
    batching 声明。当提供候选 batch kernel 后，``strict_shadow_fallback`` 默认仍为
    ``True``：每个候选输出都会与同一 job 的 reference 语义比较；不一致时仅该 job
    采用 reference。若整个 batch 抛异常，则所有 job 各自调用 reference。该设计优先
    保持当前 TableFormer decoder 的 batch=1 语义，而非宣称未经验证的加速。
    """

    def __init__(
        self,
        planner: TableBatchPlanner,
        reference_callable: TableReferenceCallable,
        *,
        batch_callable: TableBatchCallable | None = None,
        comparator: TablePredictionComparator | None = None,
        strict_shadow_fallback: bool = True,
    ) -> None:
        self.planner = planner
        self.reference_callable = reference_callable
        self.batch_callable = batch_callable
        self.comparator = comparator or TablePredictionComparator()
        self.strict_shadow_fallback = strict_shadow_fallback

    def run(self, jobs: Sequence[TableJob]) -> list[TableExecutionResult]:
        """执行所有 bucket，并在结束时恢复 page/table/order 顺序。"""

        results: list[TableExecutionResult] = []
        for batch in self.planner.plan(jobs):
            results.extend(self._run_batch(batch))
        return restore_table_order(results)

    def _reference_result(
        self,
        job: TableJob,
        *,
        bucket_key: Hashable | None,
        batch_error: str | None = None,
        batched: TablePrediction | None = None,
        source: Literal["reference", "shadow_fallback"] = "reference",
    ) -> TableExecutionResult:
        """逐 job 运行 reference，并构造统一审计结果。"""

        reference = coerce_table_prediction(self.reference_callable(job))
        return TableExecutionResult(
            job=job,
            prediction=reference,
            bucket_key=bucket_key,
            source=source,
            batched=batched,
            reference=reference,
            batch_error=batch_error,
        )

    def _run_batch(self, batch: TableBatch) -> list[TableExecutionResult]:
        """运行一个候选 batch；异常和语义漂移均收敛到 job 级 reference。"""

        if self.batch_callable is None:
            return [
                self._reference_result(job, bucket_key=batch.bucket_key)
                for job in batch.jobs
            ]

        try:
            batched = _prediction_sequence(
                self.batch_callable(batch.jobs),
                expected_size=len(batch.jobs),
            )
        except Exception as error:
            error_summary = f"{type(error).__name__}: {error}"
            return [
                self._reference_result(
                    job,
                    bucket_key=batch.bucket_key,
                    batch_error=error_summary,
                )
                for job in batch.jobs
            ]

        if not self.strict_shadow_fallback:
            return [
                TableExecutionResult(
                    job=job,
                    prediction=prediction,
                    bucket_key=batch.bucket_key,
                    source="batch",
                    batched=prediction,
                )
                for job, prediction in zip(batch.jobs, batched)
            ]

        results: list[TableExecutionResult] = []
        for job, batch_prediction in zip(batch.jobs, batched):
            reference = coerce_table_prediction(self.reference_callable(job))
            if self.comparator.equivalent(reference, batch_prediction):
                results.append(
                    TableExecutionResult(
                        job=job,
                        prediction=batch_prediction,
                        bucket_key=batch.bucket_key,
                        source="batch",
                        batched=batch_prediction,
                        reference=reference,
                    )
                )
            else:
                results.append(
                    TableExecutionResult(
                        job=job,
                        prediction=reference,
                        bucket_key=batch.bucket_key,
                        source="shadow_fallback",
                        batched=batch_prediction,
                        reference=reference,
                    )
                )
        return results
