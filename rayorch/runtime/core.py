"""Core rowwise runtime primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence


def _id(*parts: Any) -> str:
    """Readable id for MVP lineage demos."""
    text: List[str] = []
    for part in parts:
        if isinstance(part, (tuple, list)):
            if part:
                text.append(",".join(str(x) for x in part))
            continue
        text.append(str(part))
    return ":".join(text)


class BadRecordError(Exception):
    """Mark one input row as bad.

    ``index`` is local to the microbatch passed to the current op. If omitted,
    the runtime falls back to split-and-retry isolation.
    """

    def __init__(
        self,
        message: str,
        *,
        index: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.index = index
        # Classification only; the multigrain RecoveryPolicy decides whether
        # and when to retry. False preserves deterministic-poison semantics.
        self.retryable = bool(retryable)


@dataclass
class MicroBatch:
    """Row-aligned columns plus hidden runtime metadata."""

    columns: Dict[str, List[Any]]
    row_ids: List[str]
    path_ids: List[str]

    @classmethod
    def source(
        cls,
        columns: Mapping[str, Sequence[Any]],
        *,
        dataset: str = "source",
    ) -> "MicroBatch":
        sizes = {name: len(values) for name, values in columns.items()}
        if not sizes:
            raise ValueError("columns cannot be empty")
        if len(set(sizes.values())) != 1:
            raise ValueError(f"columns must have the same length, got {sizes}")
        n = next(iter(sizes.values()))
        return cls(
            columns={name: list(values) for name, values in columns.items()},
            row_ids=[_id(dataset, i) for i in range(n)],
            path_ids=["source"] * n,
        )

    def __len__(self) -> int:
        return len(self.row_ids)

    def slice(self, start: int, end: int) -> "MicroBatch":
        """Slice rows while preserving their ids and current paths."""
        return MicroBatch(
            columns={name: values[start:end] for name, values in self.columns.items()},
            row_ids=self.row_ids[start:end],
            path_ids=self.path_ids[start:end],
        )

    def row(self, index: int) -> Dict[str, Any]:
        return {name: values[index] for name, values in self.columns.items()}


@dataclass
class QuarantineRecord:
    """A bad row removed from the healthy data path."""

    row_id: str
    path_id: str
    op: str
    error: str
    values: Dict[str, Any]


@dataclass(frozen=True)
class LineageNode:
    """One internal lineage edge with one or more upstream path heads."""

    parents: tuple[str, ...]
    op: str | None


@dataclass
class RuntimeResult:
    """Healthy rows plus runtime metadata emitted by one op call."""

    batch: MicroBatch
    quarantined: List[QuarantineRecord]
    paths: Dict[str, LineageNode]
    row_path: Dict[str, str]

    def trace(self, path_id: str) -> List[str]:
        """Return the de-duplicated operator history behind one path."""
        return trace_lineage(self.paths, path_id)

    def trace_row(self, row_id: str) -> List[str]:
        """Return the operator history for one healthy output row."""
        return self.trace(self.row_path[row_id])


@dataclass(frozen=True)
class RuntimeNodeSpec:
    """DAG-bound runtime port spec for one node."""

    node: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


class LineageStore:
    """Small in-memory lineage store for MVP tests and debug output."""

    def __init__(self) -> None:
        self.paths: Dict[str, LineageNode] = {}
        self.row_path: Dict[str, str] = {}
        self.quarantined: List[QuarantineRecord] = []

    def advance(
        self,
        row_id: str,
        parent_path: str,
        op: str,
    ) -> str:
        """Move one healthy row through a rowwise op."""
        path = _id(parent_path, op)
        self.paths.setdefault(path, LineageNode((parent_path,), op))
        self.row_path[row_id] = path
        return path

    def quarantine(self, record: QuarantineRecord) -> None:
        self.quarantined.append(record)

    def trace(self, path: str) -> List[str]:
        """Recover the op trajectory for a path id."""
        return trace_lineage(self.paths, path)

    def result(self, batch: MicroBatch, bad: List[QuarantineRecord]) -> RuntimeResult:
        """Freeze the current in-memory delta into a Ray-serializable result."""
        return RuntimeResult(
            batch=batch,
            quarantined=bad,
            paths=dict(self.paths),
            row_path=dict(self.row_path),
        )


def merge_runtime_results(results: Sequence[RuntimeResult]) -> RuntimeResult:
    """Merge replica-local RuntimeResult objects in order."""
    if not results:
        return RuntimeResult(MicroBatch({}, [], []), [], {}, {})

    columns = {name: [] for name in results[0].batch.columns}
    row_ids: List[str] = []
    path_ids: List[str] = []
    quarantined: List[QuarantineRecord] = []
    paths: Dict[str, LineageNode] = {}
    row_path: Dict[str, str] = {}

    for result in results:
        for name in columns:
            columns[name].extend(result.batch.columns[name])
        row_ids.extend(result.batch.row_ids)
        path_ids.extend(result.batch.path_ids)
        quarantined.extend(result.quarantined)
        paths.update(result.paths)
        row_path.update(result.row_path)

    return RuntimeResult(
        batch=MicroBatch(columns, row_ids, path_ids),
        quarantined=quarantined,
        paths=paths,
        row_path=row_path,
    )


def merge_lineage_heads(
    parents: Sequence[str],
) -> tuple[str, Dict[str, LineageNode]]:
    """Collapse one or more upstream heads into one opaque path id."""
    unique = tuple(dict.fromkeys(parents))
    if not unique:
        raise ValueError("lineage join requires at least one parent")
    if len(unique) == 1:
        return unique[0], {}

    path = _id("join", unique)
    return path, {path: LineageNode(unique, None)}


def trace_lineage(paths: Mapping[str, LineageNode], path_id: str) -> List[str]:
    """Traverse a lineage DAG and return each operator once in parent order."""
    ops: List[str] = []
    seen_paths: set[str] = set()
    seen_ops: set[str] = set()

    def visit(path: str) -> None:
        if path == "source" or path in seen_paths:
            return
        seen_paths.add(path)
        node = paths[path]
        for parent in node.parents:
            visit(parent)
        if node.op is not None and node.op not in seen_ops:
            seen_ops.add(node.op)
            ops.append(node.op)

    visit(path_id)
    return ops


def run_rowwise(
    fn: Callable[..., Any],
    batch: MicroBatch,
    *,
    op: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    lineage: LineageStore | None = None,
) -> tuple[MicroBatch, List[QuarantineRecord]]:
    """Run a rowwise op, quarantine bad rows, and return healthy outputs."""
    runner = _RowwiseRunner(
        fn=fn,
        op=op,
        inputs=inputs,
        outputs=outputs,
        lineage=lineage or LineageStore(),
    )
    return runner.run(batch)


class _RowwiseRunner:
    """Internal executor object for one rowwise op invocation."""

    def __init__(
        self,
        *,
        fn: Callable[..., Any],
        op: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        lineage: LineageStore,
    ) -> None:
        self.fn = fn
        self.op = op
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self.lineage = lineage

    def run(self, batch: MicroBatch) -> tuple[MicroBatch, List[QuarantineRecord]]:
        if len(batch) == 0:
            return self.empty(), []

        try:
            out = self.as_columns(
                self.fn(*[batch.columns[name] for name in self.inputs]),
                len(batch),
            )
        except BadRecordError as exc:
            if exc.index is not None:
                return self.drop_one_and_retry(batch, exc.index, str(exc))
            return self.split_and_retry(batch, str(exc))
        except Exception as exc:
            return self.split_and_retry(batch, f"{exc.__class__.__name__}: {exc}")

        path_ids = [
            self.lineage.advance(row_id, parent_path, self.op)
            for row_id, parent_path in zip(batch.row_ids, batch.path_ids)
        ]
        return MicroBatch(out, list(batch.row_ids), path_ids), []

    def as_columns(self, value: Any, n: int) -> Dict[str, List[Any]]:
        values = (value,) if len(self.outputs) == 1 else tuple(value)
        if len(values) != len(self.outputs):
            raise ValueError(
                f"{self.op} returned {len(values)} outputs, "
                f"expected {len(self.outputs)}"
            )
        columns: Dict[str, List[Any]] = {}
        for name, col in zip(self.outputs, values):
            if not isinstance(col, list):
                raise TypeError(f"{self.op}.{name} must be a list")
            if len(col) != n:
                raise ValueError(f"rowwise {self.op}.{name} length {len(col)} != {n}")
            columns[name] = col
        return columns

    def empty(self) -> MicroBatch:
        return MicroBatch({name: [] for name in self.outputs}, [], [])

    def quarantine(self, batch: MicroBatch, index: int, error: str) -> QuarantineRecord:
        record = QuarantineRecord(
            row_id=batch.row_ids[index],
            path_id=batch.path_ids[index],
            op=self.op,
            error=error,
            values=batch.row(index),
        )
        self.lineage.quarantine(record)
        return record

    def drop_one_and_retry(
        self,
        batch: MicroBatch,
        bad: int,
        error: str,
    ) -> tuple[MicroBatch, List[QuarantineRecord]]:
        if bad < 0 or bad >= len(batch):
            raise IndexError(
                f"bad record index {bad} out of range for batch size {len(batch)}"
            )

        pieces: List[tuple[MicroBatch, List[QuarantineRecord]]] = []
        if bad:
            pieces.append(self.run(batch.slice(0, bad)))
        bad_record = self.quarantine(batch, bad, error)
        if bad + 1 < len(batch):
            pieces.append(self.run(batch.slice(bad + 1, len(batch))))

        healthy, quarantined = self.merge(pieces)
        return healthy, [bad_record] + quarantined

    def split_and_retry(
        self,
        batch: MicroBatch,
        error: str,
    ) -> tuple[MicroBatch, List[QuarantineRecord]]:
        if len(batch) == 1:
            record = self.quarantine(batch, 0, error)
            return self.empty(), [record]

        mid = len(batch) // 2
        return self.merge([
            self.run(batch.slice(0, mid)),
            self.run(batch.slice(mid, len(batch))),
        ])

    def merge(
        self,
        pieces: Sequence[tuple[MicroBatch, List[QuarantineRecord]]],
    ) -> tuple[MicroBatch, List[QuarantineRecord]]:
        columns = {name: [] for name in self.outputs}
        row_ids: List[str] = []
        path_ids: List[str] = []
        quarantined: List[QuarantineRecord] = []
        for batch, bad in pieces:
            for name in self.outputs:
                columns[name].extend(batch.columns[name])
            row_ids.extend(batch.row_ids)
            path_ids.extend(batch.path_ids)
            quarantined.extend(bad)
        return MicroBatch(columns, row_ids, path_ids), quarantined
