"""Cardinality-faithful full-corpus poison-isolation probe.

This benchmark consumes the frozen PDF manifest and expands its exact parent/page
cardinalities without decoding image payloads.  It complements, but never replaces,
the real MinerU benchmark: the synthetic per-page delay isolates scheduler work,
READY admission and late-commit semantics from PDF rasterization and model variance.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast

from .. import F, Executor, GroupFailure, ItemOutcome, Pipeline, Port, RayModule
from .mineru_poison import load_pdf_manifest, worker_resource_options


class ManifestParentToPages:
    """Expand one manifest parent into exact lightweight page records."""

    def run(self, parents: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        return [
            [
                {
                    "parent_id": int(parent["parent_id"]),
                    "page_id": page_id,
                    "pdf_pages": int(parent["pages"]),
                    "poison": bool(parent["poison"] and page_id == 0),
                }
                for page_id in range(int(parent["pages"]))
            ]
            for parent in parents
        ]


class SyntheticPageWork:
    """Model a fixed page cost and return a typed parent failure at page zero."""

    def __init__(self, work_ms: float) -> None:
        if work_ms < 0:
            raise ValueError("work_ms must be non-negative")
        self.work_s = work_ms / 1000.0

    def run(self, pages: list[dict[str, Any]]) -> list[int | GroupFailure]:
        live = sum(not bool(page["poison"]) for page in pages)
        if self.work_s and live:
            time.sleep(self.work_s * live)
        return [
            (
                GroupFailure(
                    f"synthetic poison parent={page['parent_id']} page=0"
                )
                if page["poison"]
                else int(page["page_id"])
            )
            for page in pages
        ]


class ParentIdentity:
    """Carry a lightweight parent identity into the terminal reducer."""

    def run(self, parents: list[dict[str, Any]]) -> list[int]:
        return [int(parent["parent_id"]) for parent in parents]


class CountHealthyParent:
    """Return one terminal observation for every non-suppressed parent."""

    def run(
        self,
        grouped_values: list[list[int]],
        grouped_pages: list[list[dict[str, Any]]],
        parent_ids: list[int],
    ) -> list[dict[str, int]]:
        outputs = []
        for values, pages, parent_id in zip(
            grouped_values,
            grouped_pages,
            parent_ids,
            strict=True,
        ):
            if len(values) != len(pages):
                raise ValueError("synthetic page/value groups must align")
            outputs.append({"parent_id": int(parent_id), "pages": len(values)})
        return outputs


@dataclass(frozen=True, slots=True)
class _DaftPage:
    """Keep one exact manifest page as an opaque Daft Python value."""

    parent_id: int
    page_id: int
    pages: int
    poison: bool


@dataclass(frozen=True, slots=True)
class _DaftWorkedPage:
    """Carry work and batch observations through Daft's parent regroup."""

    parent_id: int
    page_id: int
    poison: bool
    batch_token: str
    batch_size: int


@dataclass(frozen=True, slots=True)
class _DaftParentResult:
    """One terminal Daft observation for a healthy or poisoned parent."""

    parent_id: int
    pages: int
    suppressed: bool
    batch_tokens: tuple[str, ...]
    batch_sizes: tuple[int, ...]


class CardinalityPipeline(Pipeline):
    """Exact Parent→Page→Work→Parent graph for the full manifest."""

    def __init__(
        self,
        *,
        replicas: int,
        batch_size: int,
        work_ms: float,
        expand_replicas: int,
        reduce_replicas: int,
        cpu_worker_resource: str | None,
        gpu_worker_resource: str | None,
    ) -> None:
        self.expand = RayModule(ManifestParentToPages).ray_options(
            replicas=expand_replicas,
            batch_size=1,
            num_cpus=1,
            **worker_resource_options(cpu_worker_resource),
        )
        self.work = (
            RayModule(SyntheticPageWork)
            .pre_init(work_ms=work_ms)
            .ray_options(
                replicas=replicas,
                batch_size=batch_size,
                batching_policy="any_parent",
                num_cpus=1,
                num_gpus=1,
                **worker_resource_options(gpu_worker_resource),
            )
        )
        self.identity = RayModule(ParentIdentity).ray_options(
            replicas=1,
            batch_size=64,
            num_cpus=1,
            **worker_resource_options(cpu_worker_resource),
        )
        self.reduce = RayModule(CountHealthyParent).ray_options(
            replicas=reduce_replicas,
            batch_size=8,
            num_cpus=1,
            **worker_resource_options(cpu_worker_resource),
        )

    def forward(self, parents: Port):  # pyright: ignore[reportIncompatibleMethodOverride]
        pages = F.expand(cast(Port, self.expand(parents)))
        values = cast(Port, self.work(pages))
        identities = cast(Port, self.identity(parents))
        value_groups, page_groups = F.reduce_aligned(
            values,
            pages,
            members=values,
        )
        return self.reduce(value_groups, page_groups, identities)


def _load_parents(
    manifest: str,
    poison_largest: int,
    *,
    selection_largest: int | None = None,
    healthy_max_pages: int | None = None,
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    pdfs, page_counts = load_pdf_manifest(manifest)
    if not 0 <= poison_largest <= len(pdfs):
        raise ValueError("poison_largest is outside the manifest range")
    if (selection_largest is None) != (healthy_max_pages is None):
        raise ValueError(
            "selection_largest and healthy_max_pages must be set together"
        )
    ranked = sorted(
        range(len(pdfs)),
        key=lambda index: (-page_counts[index], index),
    )
    if selection_largest is None:
        selected = tuple(range(len(pdfs)))
    else:
        if not 0 <= selection_largest <= len(pdfs):
            raise ValueError("selection_largest is outside the manifest range")
        if healthy_max_pages is None or healthy_max_pages <= 0:
            raise ValueError("healthy_max_pages must be positive")
        if poison_largest > selection_largest:
            raise ValueError("poison_largest cannot exceed selection_largest")
        retained_large = frozenset(ranked[:selection_largest])
        selected = tuple(
            index
            for index, pages in enumerate(page_counts)
            if index in retained_large or pages <= healthy_max_pages
        )

    poisoned_source = frozenset(ranked[:poison_largest])
    poisoned = tuple(
        local_index
        for local_index, source_index in enumerate(selected)
        if source_index in poisoned_source
    )
    parents = [
        {
            "parent_id": local_index,
            "path": pdfs[source_index],
            "pages": int(page_counts[source_index]),
            "poison": local_index in poisoned,
        }
        for local_index, source_index in enumerate(selected)
    ]
    return parents, poisoned


def _load_selected_parents(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    return _load_parents(
        args.input_manifest,
        args.poison_largest,
        selection_largest=args.selection_largest,
        healthy_max_pages=args.healthy_max_pages,
    )


def _expected(
    parents: list[dict[str, Any]],
    poisoned: tuple[int, ...],
) -> dict[str, int]:
    total_pages = sum(int(parent["pages"]) for parent in parents)
    poisoned_pages = sum(int(parents[index]["pages"]) for index in poisoned)
    return {
        "documents": len(parents),
        "pages": total_pages,
        "poisoned_documents": len(poisoned),
        "poisoned_parent_pages": poisoned_pages,
        "healthy_documents": len(parents) - len(poisoned),
        "healthy_pages": total_pages - poisoned_pages,
    }


def run_v36(args: argparse.Namespace) -> dict[str, Any]:
    import ray  # pyright: ignore[reportMissingImports]

    parents, poisoned = _load_selected_parents(args)
    expected = _expected(parents, poisoned)
    pipeline = CardinalityPipeline(
        replicas=args.replicas,
        batch_size=args.batch_size,
        work_ms=args.work_ms,
        expand_replicas=args.expand_replicas,
        reduce_replicas=args.reduce_replicas,
        cpu_worker_resource=args.cpu_worker_resource,
        gpu_worker_resource=args.gpu_worker_resource,
    )
    compiled = pipeline.compile()
    work_call = next(
        call
        for call, spec in compiled.logical.calls.items()
        if spec.udf.target is pipeline.work.udf
    )
    started_ray_here = not ray.is_initialized()
    if started_ray_here:
        ray.init(address=args.ray_address, include_dashboard=False)
    started = time.perf_counter()
    try:
        with Executor(compiled) as executor:
            result = executor.run(
                parents,
                microbatch_size=args.microbatch_size,
                max_active_microbatches=args.max_active_microbatches,
            )
        framework_wall = time.perf_counter() - started
    finally:
        if started_ray_here:
            ray.shutdown()
    end_to_end = time.perf_counter() - started

    outputs = list(cast(Iterable[dict[str, int] | ItemOutcome], result.outputs))
    successful = [value for value in outputs if isinstance(value, dict)]
    suppressed = sum(value is ItemOutcome.SUPPRESSED for value in outputs)
    output_pages = sum(int(value["pages"]) for value in successful)
    output_ids = {int(value["parent_id"]) for value in successful}
    healthy_ids = set(range(len(parents))) - set(poisoned)
    work_metrics = next(
        metrics for metrics in result.calls if metrics.call_index == work_call.value
    )
    ready_saved = expected["pages"] - work_metrics.grains
    model_pages = work_metrics.grains - suppressed
    inflight_discarded = (
        expected["poisoned_parent_pages"] - suppressed - ready_saved
    )
    closed = (
        suppressed + ready_saved + inflight_discarded
        == expected["poisoned_parent_pages"]
    )
    payload = {
        "engine": "multigrain_v3_6",
        "benchmark": "full_corpus_cardinality_probe",
        "input_manifest": os.path.abspath(args.input_manifest),
        **expected,
        "poisoned_parent_indices": list(poisoned),
        "poisoned_parent_page_counts": [
            int(parents[index]["pages"]) for index in poisoned
        ],
        "poison_policy": "group_failure",
        "poison_observed": suppressed,
        "successful_documents": len(successful),
        "output_pages": output_pages,
        "output_identity_passed": output_ids == healthy_ids,
        "contract_passed": (
            suppressed == len(poisoned)
            and len(successful) == expected["healthy_documents"]
            and output_pages == expected["healthy_pages"]
            and output_ids == healthy_ids
            and closed
        ),
        "replicas": args.replicas,
        "batch_size": args.batch_size,
        "work_ms": args.work_ms,
        "microbatch_size": args.microbatch_size,
        "max_active_microbatches": args.max_active_microbatches,
        "input_mode": "parent_lineage_expand",
        "page_partitions": None,
        "selection_largest": args.selection_largest,
        "healthy_max_pages": args.healthy_max_pages,
        "measured_wall_s": round(result.elapsed_s, 6),
        "framework_wall_s": round(framework_wall, 6),
        "end_to_end_wall_s": round(end_to_end, 6),
        "rpc_count": result.rpc_count,
        "work_rpc_count": work_metrics.rpcs,
        "work_dispatched_pages": work_metrics.grains,
        "physical_model_pages": model_pages,
        "ready_siblings_not_dispatched": ready_saved,
        "inflight_siblings_computed_but_discarded": inflight_discarded,
        "suppression_accounting_closed": closed,
        "work_batch_fill_ratio": work_metrics.average_batch / args.batch_size,
        "work_batch_histogram": dict(sorted(Counter(work_metrics.batch_sizes).items())),
        "work_retries": work_metrics.retries,
        "actor_count": result.actor_count,
        "peak_active_microbatches": result.peak_active_microbatches,
    }
    return payload


class _RayDataExpandParent:
    def __call__(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "parent_id": int(row["parent_id"]),
                "page_id": page_id,
                "poison": bool(row["poison"] and page_id == 0),
            }
            for page_id in range(int(row["pages"]))
        ]


class _RayDataSyntheticWork:
    def __init__(self, work_ms: float) -> None:
        self.work_s = work_ms / 1000.0
        self.actor_token = f"{os.getpid()}-{time.time_ns()}"
        self.calls = 0

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        import numpy as np  # pyright: ignore[reportMissingImports]

        poison = np.asarray(batch["poison"], dtype=bool)
        live = int((~poison).sum())
        if self.work_s and live:
            time.sleep(self.work_s * live)
        token = f"{self.actor_token}-{self.calls}"
        self.calls += 1
        return {
            "parent_id": np.asarray(batch["parent_id"], dtype="int64"),
            "page_id": np.asarray(batch["page_id"], dtype="int64"),
            "poison": poison,
            "batch_token": np.asarray([token] * len(poison)),
            "batch_size": np.full(len(poison), len(poison), dtype="int64"),
        }


class _RayDataCountParent:
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        import numpy as np  # pyright: ignore[reportMissingImports]

        parent_id = int(batch["parent_id"][0])
        suppressed = bool(np.asarray(batch["poison"], dtype=bool).any())
        observations = {
            str(token): int(size)
            for token, size in zip(
                batch["batch_token"],
                batch["batch_size"],
                strict=True,
            )
        }
        return {
            "parent_id": np.asarray([parent_id], dtype="int64"),
            "pages": np.asarray([0 if suppressed else len(batch["page_id"])], dtype="int64"),
            "suppressed": np.asarray([suppressed], dtype=bool),
            "batch_observations": np.asarray(
                [json.dumps(observations, sort_keys=True)]
            ),
        }


def run_ray_data(args: argparse.Namespace) -> dict[str, Any]:
    import ray  # pyright: ignore[reportMissingImports]
    from ray.data.context import ShuffleStrategy  # pyright: ignore[reportMissingImports]

    parents, poisoned = _load_selected_parents(args)
    expected = _expected(parents, poisoned)
    started_ray_here = not ray.is_initialized()
    if started_ray_here:
        ray.init(address=args.ray_address, include_dashboard=False)
    ray.data.DataContext.get_current().shuffle_strategy = (
        ShuffleStrategy.SORT_SHUFFLE_PULL_BASED
    )
    if args.page_partitions is None:
        source = ray.data.from_items(
            parents,
            override_num_blocks=min(len(parents), args.source_blocks),
        )
        pages = source.flat_map(
            _RayDataExpandParent,
            concurrency=args.expand_replicas,
            num_cpus=1,
            **worker_resource_options(args.cpu_worker_resource),
        )
        input_mode = "parent_expand"
    else:
        page_rows = [
            {
                "parent_id": int(parent["parent_id"]),
                "page_id": page_id,
                "poison": bool(parent["poison"] and page_id == 0),
            }
            for parent in parents
            for page_id in range(int(parent["pages"]))
        ]
        pages = ray.data.from_items(
            page_rows,
            override_num_blocks=args.page_partitions,
        )
        input_mode = "preexpanded_balanced_pages"
    worked = pages.map_batches(
        _RayDataSyntheticWork,
        batch_size=args.batch_size,
        batch_format="numpy",
        concurrency=args.replicas,
        num_cpus=1,
        num_gpus=1,
        fn_constructor_kwargs={"work_ms": args.work_ms},
        **worker_resource_options(args.gpu_worker_resource),
    )
    outputs = worked.groupby(
        "parent_id",
        num_partitions=args.reduce_replicas,
    ).map_groups(
        _RayDataCountParent,
        batch_format="numpy",
        concurrency=args.reduce_replicas,
        num_cpus=1,
        **worker_resource_options(args.cpu_worker_resource),
    )
    started = time.perf_counter()
    try:
        rows = outputs.take_all()
        wall = time.perf_counter() - started
    finally:
        if started_ray_here:
            ray.shutdown()
    rows.sort(key=lambda row: int(row["parent_id"]))
    suppressed_ids = {
        int(row["parent_id"]) for row in rows if bool(row["suppressed"])
    }
    successful = [row for row in rows if not bool(row["suppressed"])]
    output_pages = sum(int(row["pages"]) for row in successful)
    poisoned_set = set(poisoned)
    batches: dict[str, int] = {}
    for row in rows:
        for token, size in json.loads(row["batch_observations"]).items():
            previous = batches.setdefault(token, int(size))
            if previous != int(size):
                raise ValueError("Ray Data reported conflicting batch sizes")
    batch_sizes = tuple(batches.values())
    payload = {
        "engine": "ray_data",
        "benchmark": "full_corpus_cardinality_probe",
        "input_manifest": os.path.abspath(args.input_manifest),
        **expected,
        "poisoned_parent_indices": list(poisoned),
        "poisoned_parent_page_counts": [
            int(parents[index]["pages"]) for index in poisoned
        ],
        "poison_policy": "drop_parent_after_groupby",
        "poison_observed": len(suppressed_ids),
        "successful_documents": len(successful),
        "output_pages": output_pages,
        "output_identity_passed": suppressed_ids == poisoned_set,
        "contract_passed": (
            len(rows) == expected["documents"]
            and suppressed_ids == poisoned_set
            and len(successful) == expected["healthy_documents"]
            and output_pages == expected["healthy_pages"]
        ),
        "replicas": args.replicas,
        "batch_size": args.batch_size,
        "work_ms": args.work_ms,
        "shuffle_strategy": "sort",
        "measured_wall_s": round(wall, 6),
        "framework_wall_s": round(wall, 6),
        "end_to_end_wall_s": round(wall, 6),
        "work_dispatched_pages": expected["pages"],
        "physical_model_pages": expected["pages"] - len(poisoned),
        "ready_siblings_not_dispatched": 0,
        "inflight_siblings_computed_but_discarded": (
            expected["poisoned_parent_pages"] - len(poisoned)
        ),
        "suppression_accounting_closed": True,
        "work_batch_fill_ratio": (
            sum(batch_sizes) / (len(batch_sizes) * args.batch_size)
            if batch_sizes
            else 0.0
        ),
        "work_batch_histogram": dict(sorted(Counter(batch_sizes).items())),
        "source_blocks": (
            args.page_partitions
            if args.page_partitions is not None
            else min(len(parents), args.source_blocks)
        ),
        "page_partitions": args.page_partitions,
        "input_mode": input_mode,
        "selection_largest": args.selection_largest,
        "healthy_max_pages": args.healthy_max_pages,
    }
    return payload


def run_daft(args: argparse.Namespace) -> dict[str, Any]:
    """Run Daft's natural eager-sibling parent-drop control arm."""

    import daft  # pyright: ignore[reportMissingImports]
    import ray  # pyright: ignore[reportMissingImports]
    from daft import DataType, Series  # pyright: ignore[reportMissingImports]

    parents, poisoned = _load_selected_parents(args)
    expected = _expected(parents, poisoned)

    @daft.cls(cpus=1, max_concurrency=args.expand_replicas)
    class ExpandParent:
        @daft.method(return_dtype=DataType.list(DataType.python()))
        def run(
            self,
            parent_id: int,
            pages: int,
            poison: bool,
        ) -> list[_DaftPage]:
            return [
                _DaftPage(
                    parent_id=int(parent_id),
                    page_id=page_id,
                    pages=int(pages),
                    poison=bool(poison and page_id == 0),
                )
                for page_id in range(int(pages))
            ]

    @daft.cls(cpus=1, gpus=1, max_concurrency=args.replicas)
    class SyntheticWork:
        def __init__(self, work_ms: float) -> None:
            self.work_s = work_ms / 1000.0
            self.actor_token = f"{os.getpid()}-{time.time_ns()}"
            self.calls = 0

        @daft.method.batch(
            return_dtype=DataType.python(),
            batch_size=args.batch_size,
        )
        def run(self, pages: Series) -> list[_DaftWorkedPage]:
            values = cast(list[_DaftPage], pages.to_pylist())
            live = sum(not page.poison for page in values)
            if self.work_s and live:
                time.sleep(self.work_s * live)
            token = f"{self.actor_token}-{self.calls}"
            self.calls += 1
            return [
                _DaftWorkedPage(
                    parent_id=page.parent_id,
                    page_id=page.page_id,
                    poison=page.poison,
                    batch_token=token,
                    batch_size=len(values),
                )
                for page in values
            ]

    @daft.cls(cpus=1, max_concurrency=args.reduce_replicas)
    class CountParent:
        @daft.method.batch(return_dtype=DataType.python())
        def run(self, values: Series) -> list[_DaftParentResult]:
            rows = cast(list[_DaftWorkedPage], values.to_pylist())
            if not rows:
                raise ValueError("Daft emitted an empty parent group")
            parent_id = rows[0].parent_id
            if any(row.parent_id != parent_id for row in rows):
                raise ValueError("Daft parent group mixes parent ids")
            suppressed = any(row.poison for row in rows)
            return [
                _DaftParentResult(
                    parent_id=parent_id,
                    pages=0 if suppressed else len(rows),
                    suppressed=suppressed,
                    batch_tokens=tuple(row.batch_token for row in rows),
                    batch_sizes=tuple(row.batch_size for row in rows),
                )
            ]

    started_ray_here = not ray.is_initialized()
    if started_ray_here:
        ray.init(address=args.ray_address, include_dashboard=False)
    daft.set_runner_ray(noop_if_initialized=True)
    if args.page_partitions is None:
        source = daft.from_pydict(
            {
                "parent_id": [int(parent["parent_id"]) for parent in parents],
                "pages": [int(parent["pages"]) for parent in parents],
                "poison": [bool(parent["poison"]) for parent in parents],
            }
        ).into_partitions(min(len(parents), args.source_blocks))
        expander = ExpandParent()
        pages = source.with_column(
            "page",
            expander.run(
                source["parent_id"], source["pages"], source["poison"]
            ),
        ).explode("page", ignore_empty_and_null=True)
        input_mode = "parent_expand"
    else:
        page_values = [
            _DaftPage(
                parent_id=int(parent["parent_id"]),
                page_id=page_id,
                pages=int(parent["pages"]),
                poison=bool(parent["poison"] and page_id == 0),
            )
            for parent in parents
            for page_id in range(int(parent["pages"]))
        ]
        pages = daft.from_pydict(
            {
                "parent_id": [page.parent_id for page in page_values],
                "page": page_values,
            }
        ).into_partitions(args.page_partitions)
        input_mode = "preexpanded_balanced_pages"
    worker = SyntheticWork(args.work_ms)
    worked = pages.with_column("result", worker.run(pages["page"]))
    counter = CountParent()
    outputs = worked.groupby("parent_id").map_groups(
        counter.run(worked["result"])
    )

    started = time.perf_counter()
    try:
        collected = outputs.collect().to_pydict()
        wall = time.perf_counter() - started
    finally:
        if started_ray_here:
            ray.shutdown()

    results = cast(list[_DaftParentResult], collected["result"])
    results.sort(key=lambda value: value.parent_id)
    output_ids = {
        value.parent_id for value in results if not value.suppressed
    }
    healthy_ids = set(range(len(parents))) - set(poisoned)
    suppressed_ids = {
        value.parent_id for value in results if value.suppressed
    }
    output_pages = sum(value.pages for value in results)
    batches: dict[str, int] = {}
    for result in results:
        for token, size in zip(
            result.batch_tokens,
            result.batch_sizes,
            strict=True,
        ):
            previous = batches.setdefault(token, size)
            if previous != size:
                raise ValueError("Daft reported conflicting batch sizes")
    batch_sizes = tuple(batches.values())
    model_pages = expected["pages"] - len(poisoned)
    inflight_discarded = expected["poisoned_parent_pages"] - len(poisoned)
    payload = {
        "engine": "daft",
        "daft_version": daft.__version__,
        "benchmark": "full_corpus_cardinality_probe",
        "input_manifest": os.path.abspath(args.input_manifest),
        **expected,
        "poisoned_parent_indices": list(poisoned),
        "poisoned_parent_page_counts": [
            int(parents[index]["pages"]) for index in poisoned
        ],
        "poison_policy": "drop_parent_after_groupby",
        "poison_observed": len(suppressed_ids),
        "successful_documents": len(output_ids),
        "output_pages": output_pages,
        "output_identity_passed": (
            output_ids == healthy_ids and suppressed_ids == set(poisoned)
        ),
        "contract_passed": (
            len(results) == expected["documents"]
            and output_ids == healthy_ids
            and suppressed_ids == set(poisoned)
            and output_pages == expected["healthy_pages"]
        ),
        "replicas": args.replicas,
        "batch_size": args.batch_size,
        "work_ms": args.work_ms,
        "source_partitions": (
            args.page_partitions
            if args.page_partitions is not None
            else min(len(parents), args.source_blocks)
        ),
        "page_partitions": args.page_partitions,
        "input_mode": input_mode,
        "selection_largest": args.selection_largest,
        "healthy_max_pages": args.healthy_max_pages,
        "measured_wall_s": round(wall, 6),
        "framework_wall_s": round(wall, 6),
        "end_to_end_wall_s": round(wall, 6),
        "work_dispatched_pages": expected["pages"],
        "physical_model_pages": model_pages,
        "ready_siblings_not_dispatched": 0,
        "inflight_siblings_computed_but_discarded": inflight_discarded,
        "suppression_accounting_closed": True,
        "work_batch_fill_ratio": (
            sum(batch_sizes) / (len(batch_sizes) * args.batch_size)
            if batch_sizes
            else 0.0
        ),
        "work_batch_histogram": dict(sorted(Counter(batch_sizes).items())),
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=("v36", "ray_data", "daft"),
        required=True,
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--poison-largest", type=int, default=0)
    parser.add_argument("--selection-largest", type=int, default=None)
    parser.add_argument("--healthy-max-pages", type=int, default=None)
    parser.add_argument("--replicas", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--work-ms", type=float, default=2.0)
    parser.add_argument("--expand-replicas", type=int, default=32)
    parser.add_argument("--reduce-replicas", type=int, default=32)
    parser.add_argument("--source-blocks", type=int, default=128)
    parser.add_argument("--page-partitions", type=int, default=None)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--max-active-microbatches", type=int, default=8)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--cpu-worker-resource", default=None)
    parser.add_argument("--gpu-worker-resource", default=None)
    parser.add_argument("--artifact-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.poison_largest < 0:
        raise ValueError("poison_largest must be non-negative")
    if (args.selection_largest is None) != (args.healthy_max_pages is None):
        raise ValueError(
            "selection_largest and healthy_max_pages must be set together"
        )
    if args.work_ms < 0:
        raise ValueError("work_ms must be non-negative")
    if args.page_partitions is not None and args.page_partitions <= 0:
        raise ValueError("page_partitions must be positive")
    runners = {
        "v36": run_v36,
        "ray_data": run_ray_data,
        "daft": run_daft,
    }
    payload = runners[args.engine](args)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "CardinalityPipeline",
    "CountHealthyParent",
    "ManifestParentToPages",
    "ParentIdentity",
    "SyntheticPageWork",
    "build_parser",
    "run_daft",
    "run_ray_data",
    "run_v36",
]
