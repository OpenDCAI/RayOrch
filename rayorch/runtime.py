"""Minimal record-local runtime for rowwise fault isolation."""
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

    def __init__(self, message: str, *, index: int | None = None) -> None:
        super().__init__(message)
        self.index = index


@dataclass
class MicroBatch:
    """Row-aligned columns plus hidden runtime metadata.

    User ops still receive plain Python lists from ``columns``. ``row_ids`` and
    ``path_ids`` stay internal and move alongside the rows whenever we split,
    retry, or merge healthy output.
    """

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
        """Create the first microbatch at a pipeline source."""
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


class LineageStore:
    """Small in-memory lineage store for MVP tests and debug output.

    ``paths`` is a compressed trajectory table: many rows that go through the
    same parent path and op share the same path id. Full per-row edges are not
    recorded for rowwise ops.
    """

    def __init__(self) -> None:
        self.paths: Dict[str, tuple[str, str]] = {}
        self.row_path: Dict[str, str] = {}
        self.mutations: List[tuple[str, str, str]] = [] # store all inplace mutations as (row_id, op, port)
        self.quarantined: List[QuarantineRecord] = [] # store bad rows

    def advance(
        self,
        row_id: str,
        parent_path: str,
        op: str,
        mutates: Sequence[str] = (),
    ) -> str:
        """Move one healthy row through a rowwise op."""
        path = _id(parent_path, op, tuple(mutates))
        self.paths.setdefault(path, (parent_path, op))
        self.row_path[row_id] = path
        for port in mutates:
            self.mutations.append((row_id, op, port))
        return path

    def quarantine(self, record: QuarantineRecord) -> None:
        self.quarantined.append(record)

    def trace(self, path: str) -> List[str]:
        """Recover the op trajectory for a path id."""
        ops: List[str] = []
        seen: set[str] = set()
        while path != "source" and path not in seen:
            seen.add(path)
            parent, op = self.paths[path]
            ops.append(op)
            path = parent
        return list(reversed(ops))


def run_rowwise(
    fn: Callable[..., Any],
    batch: MicroBatch,
    *,
    op: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    lineage: LineageStore | None = None,
    mutates: Sequence[str] = (),
) -> tuple[MicroBatch, List[QuarantineRecord]]:
    """Run a rowwise op, quarantine bad rows, and return healthy outputs.

    Contract for this MVP:
    - every output column must have exactly one value per input row;
    - healthy rows keep their RowID and get a new PathID;
    - bad singleton rows are omitted from the returned MicroBatch;
    - ordinary exceptions are isolated by split-and-retry.
    """
    runner = _RowwiseRunner(
        fn=fn,
        op=op,
        inputs=inputs,
        outputs=outputs,
        lineage=lineage or LineageStore(),
        mutates=mutates,
    )
    return runner.run(batch)


class _RowwiseRunner:
    """Internal executor object for one rowwise op invocation.

    Keeping this private lets the public API stay tiny while avoiding a long
    tail of module-level helpers that all pass the same context around.
    """

    def __init__(
        self,
        *,
        fn: Callable[..., Any],
        op: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        lineage: LineageStore,
        mutates: Sequence[str],
    ) -> None:
        self.fn = fn
        self.op = op
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self.lineage = lineage
        self.mutates = tuple(mutates)

    def run(self, batch: MicroBatch) -> tuple[MicroBatch, List[QuarantineRecord]]:
        if len(batch) == 0:
            return self.empty(), []

        try:
            # User code sees only normal list columns. Runtime metadata is
            # advanced only after the op succeeds for these rows.
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
            self.lineage.advance(row_id, parent_path, self.op, self.mutates)
            for row_id, parent_path in zip(batch.row_ids, batch.path_ids)
        ]
        return MicroBatch(out, list(batch.row_ids), path_ids), []

    def as_columns(self, value: Any, n: int) -> Dict[str, List[Any]]:
        """Normalize and validate the rowwise return value."""
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
        """Record one bad row without advancing its path through the failed op."""
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

        # BadRecordError(index=...) is the fast path: user code already told us
        # which row failed, so we quarantine it and retry only the healthy sides.
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

        # Generic exception path: keep bisecting until only the bad singleton
        # row fails. Healthy halves continue through the same op.
        mid = len(batch) // 2
        return self.merge([
            self.run(batch.slice(0, mid)),
            self.run(batch.slice(mid, len(batch))),
        ])

    def merge(
        self,
        pieces: Sequence[tuple[MicroBatch, List[QuarantineRecord]]],
    ) -> tuple[MicroBatch, List[QuarantineRecord]]:
        """Concatenate healthy outputs and quarantine records in row order."""
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


def _demo_print_batch(name: str, batch: MicroBatch, lineage: LineageStore) -> None:
    print(f"\n[{name}] healthy rows = {len(batch)}")
    print("  row_ids :", batch.row_ids)
    print("  path_ids:", batch.path_ids)
    if batch.path_ids:
        print("  trace   :", lineage.trace(batch.path_ids[0]))
    for col, values in batch.columns.items():
        print(f"  {col}: {values}")


def _demo_print_bad(name: str, bad: Sequence[QuarantineRecord]) -> None:
    print(f"[{name}] quarantined = {len(bad)}")
    for record in bad:
        print(
            "  bad row:",
            {
                "row_id": record.row_id,
                "path_id": record.path_id,
                "op": record.op,
                "error": record.error,
                "values": record.values,
            },
        )


if __name__ == "__main__":
    lineage = LineageStore()

    source = MicroBatch.source(
        {
            "pdf": ["paper0.pdf", "corrupt.pdf", "paper2.pdf"],
            "meta": [
                {"name": "paper0"},
                {"name": "corrupt"},
                {"name": "paper2"},
            ],
        },
        dataset="flash-mineru-demo",
    )
    _demo_print_batch("source", source, lineage)

    def pdf2img(pdfs, meta):
        images = []
        for i, (pdf, item) in enumerate(zip(pdfs, meta)):
            if pdf == "corrupt.pdf":
                raise BadRecordError("pdf parser failed", index=i)
            item["pages"] = 2
            images.append([f"img<{pdf}:0>", f"img<{pdf}:1>"])
        return images, meta

    def layout(images, meta):
        blocks = []
        for pages, item in zip(images, meta):
            item["layout_blocks"] = len(pages)
            blocks.append([[{"type": "text", "page": i}] for i, _ in enumerate(pages)])
        return blocks, meta

    def ocr(blocks, meta):
        text = []
        for per_pdf_blocks, item in zip(blocks, meta):
            item["ocr_pages"] = len(per_pdf_blocks)
            text.append(
                [
                    f"ocr<{item['name']}>:page{page_blocks[0]['page']}"
                    for page_blocks in per_pdf_blocks
                ]
            )
        return text, meta

    def convert(text, meta):
        return [
            f"{item['name']}.md pages={item['ocr_pages']} "
            f"blocks={item['layout_blocks']} text={text_rows}"
            for text_rows, item in zip(text, meta)
        ]

    b1, bad1 = run_rowwise(
        pdf2img,
        source,
        op="pdf2img",
        inputs=("pdf", "meta"),
        outputs=("images", "meta"),
        lineage=lineage,
        mutates=("meta",),
    )
    _demo_print_batch("after pdf2img", b1, lineage)
    _demo_print_bad("after pdf2img", bad1)

    b2, bad2 = run_rowwise(
        layout,
        b1,
        op="layout",
        inputs=("images", "meta"),
        outputs=("blocks", "meta"),
        lineage=lineage,
        mutates=("meta",),
    )
    _demo_print_batch("after layout", b2, lineage)
    _demo_print_bad("after layout", bad2)

    b3, bad3 = run_rowwise(
        ocr,
        b2,
        op="ocr",
        inputs=("blocks", "meta"),
        outputs=("text", "meta"),
        lineage=lineage,
        mutates=("meta",),
    )
    _demo_print_batch("after ocr", b3, lineage)
    _demo_print_bad("after ocr", bad3)

    b4, bad4 = run_rowwise(
        convert,
        b3,
        op="convert",
        inputs=("text", "meta"),
        outputs=("markdown",),
        lineage=lineage,
    )
    _demo_print_batch("after convert", b4, lineage)
    _demo_print_bad("after convert", bad4)

    print("\n[lineage store]")
    print("  paths      :", lineage.paths)
    print("  row_path   :", lineage.row_path)
    print("  mutations  :", lineage.mutations)
    print("  quarantined:", lineage.quarantined)
