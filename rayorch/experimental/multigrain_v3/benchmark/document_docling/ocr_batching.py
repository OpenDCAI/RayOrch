"""RapidOCR 识别阶段早期 tensor-planner 的 Ray-free 契约工具。

本模块刻意不依赖 Docling、RapidOCR、Ray 或 NumPy。它只处理已经由 detector
切出的 crop，并把真正的分类器、预处理器、识别器以 callable 注入。因此它可以在
接入真实 ``TextRecognizer`` 前独立做语义验证。

当前 V3 生产 adapter 不使用这个 planner：它把稳定排序的原始 crops 一次交给
``RapidOCR.recognize_txt``，由上游拥有 resize、排序、minibatch 和 decode。本模块保留为
历史低层 kernel 的契约测试，不能据此推断当前生产调用边界。

P1 的关键约束是：每个 crop 必须先独立完成与原识别器一致的预处理，只有归一化后的
tensor shape、dtype 和调用方给出的 normalization key 全部相同，才允许放入同一个
recognition batch。不能先按宽高猜测，也不能为了凑 batch 重编码 image bytes。

RapidOCR 的 ``TextRecognizer`` 会从一个 batch 的最大宽高比推导 ``max_wh_ratio``；
混入不同宽度 crop 可能改变所有输入 tensor 的 shape。本模块通过“先逐 crop 归一化、
再按实际 tensor key 分桶”避免伪造这项语义。但即使 shape 相同，ONNX/decoder 仍可能有
浮点级 decode 漂移，所以执行器默认开启严格 shadow 对比，并在不一致时按 job 回退。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Hashable, Literal, Mapping, Protocol, Sequence

from .core_values import OcrCropJob


@dataclass(frozen=True, slots=True)
class RecognizerInput:
    """一个 crop 的单独识别器预处理结果。

    ``key`` 应包含影响 tensor 语义、但无法仅从 ``tensor.shape`` 推断的条件，例如
    recognizer 的 image shape、语言/字典、固定的 ``max_wh_ratio`` 或预处理版本。
    它必须可 hash。planner 还会额外把实际 shape 和 dtype 加入 bucket key。
    """

    tensor: Any
    key: Hashable | None = None


@dataclass(frozen=True, slots=True)
class RecognizerBatchKey:
    """可安全拼接为同一 recognizer batch 的精确兼容键。"""

    shape: tuple[int, ...]
    dtype: str | None
    normalization_key: Hashable | None


@dataclass(frozen=True, slots=True)
class NormalizedOcrCrop:
    """保留原 job 关联的逐 crop 归一化识别输入。"""

    job: OcrCropJob
    recognizer_input: RecognizerInput
    batch_key: RecognizerBatchKey


@dataclass(frozen=True, slots=True)
class RecognitionBatch:
    """一个只含精确兼容 tensor 的 recognizer batch。"""

    key: RecognizerBatchKey
    jobs: tuple[NormalizedOcrCrop, ...]


class RecognizerPreprocessor(Protocol):
    """将一个 job 单独变换为 recognizer 输入的 fake/adapter 协议。"""

    def __call__(self, job: OcrCropJob) -> RecognizerInput | Any:
        """不得依赖候选 batch 的其他 crop。"""


class ClassifierCallable(Protocol):
    """单 crop 分类器协议，例如方向分类及其对应的 crop 旋转。"""

    def __call__(self, image: Any) -> Any:
        """返回供 ``ClassifierCallableAdapter`` 提取图像的分类结果。"""


class RecognizerBatchCallable(Protocol):
    """接收 tensor 序列、返回同长度识别结果序列的 fake/adapter 协议。"""

    def __call__(self, tensors: Sequence[Any]) -> Any:
        """实现者可返回结果序列，或 ``(结果序列, 耗时)``。"""


class RecognizerSingleCallable(Protocol):
    """严格 shadow 或 per-job fallback 所用的单 tensor 识别协议。"""

    def __call__(self, tensor: Any) -> Any:
        """返回一个识别结果。"""


@dataclass(frozen=True, slots=True)
class OcrRecognition:
    """统一后的文本识别结果。"""

    text: str
    score: float | None = None
    raw: Any = None


RecognitionLike = OcrRecognition | tuple[str, float | None] | Mapping[str, Any] | str


def coerce_ocr_recognition(value: RecognitionLike) -> OcrRecognition:
    """把常见 fake/RapidOCR 风格结果转换为 :class:`OcrRecognition`。

    支持 ``OcrRecognition``、``(text, score)``、带 ``text``/``score`` 的 mapping
    以及只含文本的 ``str``。真实接入遇到不同返回类型时，可以在 adapter 外先转换，
    不应在这里引入 RapidOCR 依赖。
    """

    if isinstance(value, OcrRecognition):
        return value
    if isinstance(value, str):
        return OcrRecognition(text=value, raw=value)
    if isinstance(value, Mapping):
        if "text" not in value:
            raise TypeError("识别结果 mapping 必须包含 'text'")
        score = value.get("score")
        return OcrRecognition(
            text=str(value["text"]),
            score=None if score is None else float(score),
            raw=value,
        )
    if isinstance(value, tuple) and len(value) == 2:
        text, score = value
        return OcrRecognition(
            text=str(text),
            score=None if score is None else float(score),
            raw=value,
        )
    raise TypeError(
        "无法转换识别结果；请返回 OcrRecognition、(text, score)、mapping 或 str"
    )


def _tensor_shape(tensor: Any) -> tuple[int, ...]:
    """提取 tensor 的实际 shape；不猜测 image 的宽高。"""

    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise TypeError(
            "recognizer 预处理结果必须暴露 .shape；"
            "请传入实际 normalized tensor，而不是原始 crop"
        )
    try:
        values = tuple(int(item) for item in shape)
    except TypeError as error:
        raise TypeError("recognizer tensor 的 .shape 必须可迭代") from error
    if not values:
        raise ValueError("recognizer tensor 的 .shape 不能为空")
    return values


def _tensor_dtype(tensor: Any) -> str | None:
    """提取 dtype，使同 shape 但不同数值语义的输入不被合并。"""

    dtype = getattr(tensor, "dtype", None)
    return None if dtype is None else str(dtype)


def recognizer_batch_key(recognizer_input: RecognizerInput) -> RecognizerBatchKey:
    """从逐 crop 预处理结果生成严格的 recognizer bucket key。"""

    try:
        hash(recognizer_input.key)
    except TypeError as error:
        raise TypeError("RecognizerInput.key 必须可 hash") from error
    return RecognizerBatchKey(
        shape=_tensor_shape(recognizer_input.tensor),
        dtype=_tensor_dtype(recognizer_input.tensor),
        normalization_key=recognizer_input.key,
    )


class RecognitionBatchPlanner:
    """先逐 crop 预处理，再按真实 recognizer tensor key 稳定分桶。

    ``max_batch_size`` 只负责切分已经兼容的 bucket；它绝不用于重新计算任一 crop 的
    预处理参数。因此即使 batch 被切成不同大小，已观察到的 normalized tensor shape
    也保持不变。
    """

    def __init__(
        self,
        preprocessor: RecognizerPreprocessor,
        *,
        max_batch_size: int | None = None,
    ) -> None:
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError("max_batch_size 必须为正数或 None")
        self.preprocessor = preprocessor
        self.max_batch_size = max_batch_size

    def normalize(self, jobs: Sequence[OcrCropJob]) -> list[NormalizedOcrCrop]:
        """按输入顺序逐个预处理，绝不把 image bytes 交叉混合。"""

        normalized: list[NormalizedOcrCrop] = []
        for job in jobs:
            prepared = self.preprocessor(job)
            if not isinstance(prepared, RecognizerInput):
                prepared = RecognizerInput(tensor=prepared)
            normalized.append(
                NormalizedOcrCrop(
                    job=job,
                    recognizer_input=prepared,
                    batch_key=recognizer_batch_key(prepared),
                )
            )
        return normalized

    def plan(self, jobs: Sequence[OcrCropJob]) -> list[RecognitionBatch]:
        """返回按首次出现顺序排列、每桶内部稳定的 batch 列表。"""

        buckets: dict[RecognizerBatchKey, list[NormalizedOcrCrop]] = {}
        for normalized in self.normalize(jobs):
            buckets.setdefault(normalized.batch_key, []).append(normalized)

        batches: list[RecognitionBatch] = []
        for key, bucket_jobs in buckets.items():
            size = self.max_batch_size or len(bucket_jobs)
            for start in range(0, len(bucket_jobs), size):
                batches.append(
                    RecognitionBatch(key=key, jobs=tuple(bucket_jobs[start : start + size]))
                )
        return batches


class ClassifierCallableAdapter:
    """把单 crop classifier callable 接到 batching helper，但不批处理 classifier。

    P1 的目标仅是 recognizer batching。classifier 仍逐 crop 调用，避免改变方向分类的
    batch 语义。默认把 callable 返回值直接视为新的 image；若 callable 返回的是复杂
    对象，可用 ``image_from_result`` 提取旋转后的 image。
    """

    def __init__(
        self,
        classifier: ClassifierCallable,
        *,
        image_from_result: Callable[[Any], Any] | None = None,
        metadata_key: str = "rapidocr_classifier",
    ) -> None:
        self.classifier = classifier
        self.image_from_result = image_from_result or (lambda result: result)
        self.metadata_key = metadata_key

    def classify(self, job: OcrCropJob) -> OcrCropJob:
        """分类并返回关联不变、image 可能被方向校正的新 job。"""

        result = self.classifier(job.image)
        metadata = dict(job.metadata)
        metadata[self.metadata_key] = result
        return replace(
            job,
            image=self.image_from_result(result),
            metadata=metadata,
        )


def _is_result_sequence(value: Any) -> bool:
    """判断值能否作为结果序列，同时排除文本和 mapping。"""

    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, Mapping)
    )


def _as_result_sequence(value: Any, expected_size: int) -> Sequence[Any]:
    """兼容结果序列及 RapidOCR 常见的 ``(结果序列, elapsed)`` 返回。"""

    if (
        isinstance(value, tuple)
        and len(value) == 2
        and _is_result_sequence(value[0])
        and len(value[0]) == expected_size
    ):
        return value[0]
    if _is_result_sequence(value) and len(value) == expected_size:
        return value
    if expected_size == 1:
        return (value,)
    raise ValueError(
        f"recognizer 返回 {type(value)!r}，无法对应期望的 {expected_size} 个结果"
    )


class RecognizerCallableAdapter:
    """把 tensor callable 包装成可供 shadow/fallback 使用的识别器执行器。

    ``batch_callable`` 只收到 planner 已验证兼容的 tensor 列表。没有提供
    ``single_callable`` 时，单 job shadow/fallback 会以长度为一的 batch 调用
    ``batch_callable``；真实接入时建议提供原 RapidOCR 的单 crop 路径作为
    ``single_callable``，以便影子比较的是原有语义。
    """

    def __init__(
        self,
        batch_callable: RecognizerBatchCallable,
        *,
        single_callable: RecognizerSingleCallable | None = None,
        result_converter: Callable[[Any], OcrRecognition] = coerce_ocr_recognition,
    ) -> None:
        self.batch_callable = batch_callable
        self.single_callable = single_callable
        self.result_converter = result_converter

    def recognize_batch(
        self, jobs: Sequence[NormalizedOcrCrop]
    ) -> list[OcrRecognition]:
        """执行一个已验证 shape/key 相同的 recognizer batch。"""

        tensors = [job.recognizer_input.tensor for job in jobs]
        raw_results = _as_result_sequence(self.batch_callable(tensors), len(tensors))
        results = [self.result_converter(result) for result in raw_results]
        if len(results) != len(jobs):
            raise ValueError("recognizer 返回结果数与输入 job 数不一致")
        return results

    def recognize_one(self, job: NormalizedOcrCrop) -> OcrRecognition:
        """执行原语义的单 job recognizer 调用。"""

        tensor = job.recognizer_input.tensor
        if self.single_callable is None:
            raw_result = _as_result_sequence(self.batch_callable([tensor]), 1)[0]
        else:
            raw_result = self.single_callable(tensor)
        return self.result_converter(raw_result)


class RecognitionExecutor(Protocol):
    """供 runner 使用的 recognizer fake/adapter 协议。"""

    def recognize_batch(
        self, jobs: Sequence[NormalizedOcrCrop]
    ) -> list[OcrRecognition]:
        """运行一个已分桶的 batch。"""

    def recognize_one(self, job: NormalizedOcrCrop) -> OcrRecognition:
        """运行一个原语义的单 job。"""


@dataclass(frozen=True, slots=True)
class RecognitionComparator:
    """文本严格相等、score 按容差比较的 shadow 对比规则。"""

    score_atol: float = 0.0
    score_rtol: float = 0.0
    text_normalizer: Callable[[str], str] | None = None

    def __post_init__(self) -> None:
        if self.score_atol < 0 or self.score_rtol < 0:
            raise ValueError("score 容差不能为负数")

    def equivalent(self, expected: OcrRecognition, actual: OcrRecognition) -> bool:
        """比较文本与置信度；任一缺失 score 都要求二者同时缺失。"""

        normalizer = self.text_normalizer or (lambda text: text)
        if normalizer(expected.text) != normalizer(actual.text):
            return False
        if expected.score is None or actual.score is None:
            return expected.score is None and actual.score is None
        return math.isclose(
            expected.score,
            actual.score,
            rel_tol=self.score_rtol,
            abs_tol=self.score_atol,
        )


@dataclass(frozen=True, slots=True)
class OcrBatchResult:
    """一个识别结果及其 shadow/fallback 审计信息。"""

    job: OcrCropJob
    recognition: OcrRecognition
    batch_key: RecognizerBatchKey
    source: Literal["batch", "shadow_fallback", "fallback"]
    batched: OcrRecognition | None = None
    shadow: OcrRecognition | None = None
    batch_error: str | None = None

    @property
    def used_fallback(self) -> bool:
        """是否没有采用原 batch 的输出。"""

        return self.source != "batch"


def restore_page_rect_order(
    results: Sequence[OcrBatchResult],
) -> list[OcrBatchResult]:
    """按 page/rect/crop/local_order 恢复确定性 Docling 输出顺序。"""

    return sorted(results, key=lambda result: result.job.order_key)


class RapidOCRRecognitionBatcher:
    """P1 recognizer batching 的保守执行器。

    默认 ``strict_shadow_fallback=True``：每一个 batched 输出都会与单 job shadow 输出
    比较。比较失败时，默认直接采用 shadow（``source='shadow_fallback'``）；若传入
    ``fallback_recognizer``，则只对失败的 job 再调用一次该 fallback，并采用其输出。
    batch 调用异常时，也会逐 job fallback。这样调用方无需相信“shape 相同”就必然得到
    bit-exact decode。
    """

    def __init__(
        self,
        planner: RecognitionBatchPlanner,
        recognizer: RecognitionExecutor,
        *,
        classifier: ClassifierCallableAdapter | None = None,
        comparator: RecognitionComparator | None = None,
        strict_shadow_fallback: bool = True,
        fallback_recognizer: RecognitionExecutor | None = None,
    ) -> None:
        self.planner = planner
        self.recognizer = recognizer
        self.classifier = classifier
        self.comparator = comparator or RecognitionComparator()
        self.strict_shadow_fallback = strict_shadow_fallback
        self.fallback_recognizer = fallback_recognizer

    def run(self, jobs: Sequence[OcrCropJob]) -> list[OcrBatchResult]:
        """批量识别并恢复到稳定 page/rect 顺序。"""

        original_by_object_id: dict[int, OcrCropJob] = {}
        classified_jobs: list[OcrCropJob] = []
        for job in jobs:
            classified = self.classifier.classify(job) if self.classifier else job
            original_by_object_id[id(classified)] = job
            classified_jobs.append(classified)

        results: list[OcrBatchResult] = []
        for batch in self.planner.plan(classified_jobs):
            results.extend(
                self._run_one_batch(batch, original_by_object_id=original_by_object_id)
            )
        return restore_page_rect_order(results)

    def _run_one_batch(
        self,
        batch: RecognitionBatch,
        *,
        original_by_object_id: Mapping[int, OcrCropJob],
    ) -> list[OcrBatchResult]:
        """运行一个 bucket，并把异常或漂移收敛到 job 级 fallback。"""

        try:
            batched = self.recognizer.recognize_batch(batch.jobs)
        except Exception as error:
            return [
                self._fallback_after_batch_error(
                    normalized,
                    original_job=original_by_object_id[id(normalized.job)],
                    error=error,
                )
                for normalized in batch.jobs
            ]

        if len(batched) != len(batch.jobs):
            raise ValueError("recognizer 返回结果数与 batch job 数不一致")

        outputs: list[OcrBatchResult] = []
        for normalized, batch_result in zip(batch.jobs, batched):
            original_job = original_by_object_id[id(normalized.job)]
            shadow = (
                self.recognizer.recognize_one(normalized)
                if self.strict_shadow_fallback
                else None
            )
            if shadow is None or self.comparator.equivalent(shadow, batch_result):
                outputs.append(
                    OcrBatchResult(
                        job=original_job,
                        recognition=batch_result,
                        batch_key=batch.key,
                        source="batch",
                        batched=batch_result,
                        shadow=shadow,
                    )
                )
                continue

            if self.fallback_recognizer is None:
                outputs.append(
                    OcrBatchResult(
                        job=original_job,
                        recognition=shadow,
                        batch_key=batch.key,
                        source="shadow_fallback",
                        batched=batch_result,
                        shadow=shadow,
                    )
                )
                continue

            fallback = self.fallback_recognizer.recognize_one(normalized)
            outputs.append(
                OcrBatchResult(
                    job=original_job,
                    recognition=fallback,
                    batch_key=batch.key,
                    source="fallback",
                    batched=batch_result,
                    shadow=shadow,
                )
            )
        return outputs

    def _fallback_after_batch_error(
        self,
        normalized: NormalizedOcrCrop,
        *,
        original_job: OcrCropJob,
        error: Exception,
    ) -> OcrBatchResult:
        """batch 异常时按 job 回退，且不丢失原始异常摘要。"""

        executor = self.fallback_recognizer or self.recognizer
        fallback = executor.recognize_one(normalized)
        return OcrBatchResult(
            job=original_job,
            recognition=fallback,
            batch_key=normalized.batch_key,
            source="fallback",
            batch_error=f"{type(error).__name__}: {error}",
        )
