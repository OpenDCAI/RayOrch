"""End-to-end V3 execution through native Pipeline and local Ray actors."""

from __future__ import annotations

import time

import pytest

import rayorch.experimental.multigrain_v3 as mg


pytestmark = pytest.mark.usefixtures("ray_cluster")


class ExpandPages:
    def run(self, parents):
        return [
            [f"{parent}:{ordinal}" for ordinal in range(count)]
            for parent, count in parents
        ]


class Upper:
    def run(self, pages):
        return [page.upper() for page in pages]


class Keep:
    def run(self, pages):
        return [not page.endswith(":1") for page in pages]


class Assemble:
    def run(self, members):
        return [tuple(group) for group in members]


class FanoutPipeline(mg.Pipeline):
    def __init__(self):
        self.expand = mg.Expand(ExpandPages).ray_options(batch_size=2)
        self.map = mg.Map(Upper).ray_options(batch_size=8)
        self.filter = mg.Filter(Keep).ray_options(batch_size=8)
        self.reduce = mg.Reduce(Assemble).ray_options(batch_size=2)

    def forward(self, parents):
        pages = self.expand(parents)
        mapped = self.map(pages)
        kept = self.filter(mapped)
        return self.reduce(anchor=parents, members=kept)


def test_expand_filter_reduce_restores_order_and_empty_groups():
    """N=0 and filtered children preserve exact ordered Reduce groups."""

    result = mg.Executor(FanoutPipeline()).run(
        [("a", 3), ("empty", 0)]
    )
    assert result.get() == (("A:0", "A:2"), ())
    assert result.metrics["grains_per_rpc"] > 1


class TupleMask:
    def run(self, left, right):
        return [value % 2 == 0 for value in left]


class TuplePipeline(mg.Pipeline):
    def __init__(self):
        self.filter = mg.Filter(TupleMask).ray_options(batch_size=4)

    def forward(self, left, right):
        return self.filter(left, right)


def test_filter_masks_all_aligned_inputs_without_copying_payload():
    """One bool mask filters every aligned input Port synchronously."""

    result = mg.Executor(TuplePipeline()).run(
        [0, 1, 2, 3],
        ["a", "b", "c", "d"],
    )
    assert result.get() == (0, 2, "a", "c")


class PickOptional:
    def run(self, kind, pdf, image):
        outputs = []
        for value_kind, pdf_value, image_value in zip(kind, pdf, image):
            outputs.append(
                pdf_value if value_kind == "pdf" else image_value
            )
        return outputs


class KindFilter:
    def __init__(self, target):
        self.target = target

    def run(self, kinds, values):
        return [kind == self.target for kind in kinds]


class OptionalPipeline(mg.Pipeline):
    def __init__(self):
        self.pdf = mg.Filter(KindFilter).pre_init("pdf")
        self.image = mg.Filter(KindFilter).pre_init("image")
        self.normalize = mg.Map(PickOptional).ray_options(batch_size=4)

    def forward(self, kinds, values):
        _, pdf_values = self.pdf(kinds, values)
        _, image_values = self.image(kinds, values)
        return self.normalize(
            kinds,
            pdf=mg.optional(pdf_values),
            image=mg.optional(image_values),
        )


def test_optional_mutually_exclusive_branches_normalize_with_missing():
    """Normal branch absence becomes MISSING while failures remain fail-closed."""

    result = mg.Executor(OptionalPipeline()).run(
        ["pdf", "image", "pdf"],
        ["p0", "i0", "p1"],
    )
    assert result.get() == ("p0", "i0", "p1")


class Timed:
    def __init__(self, delay):
        self.delay = delay

    def run(self, rows):
        time.sleep(self.delay)
        return list(rows)


class TwoStagePipeline(mg.Pipeline):
    def __init__(self):
        self.first = mg.Map(Timed).pre_init(0.12).ray_options(batch_size=1)
        self.second = mg.Map(Timed).pre_init(0.12).ray_options(batch_size=1)

    def forward(self, rows):
        return self.second(self.first(rows))


def _timed(max_inflight):
    return mg.Executor(
        TwoStagePipeline(),
        microbatch_size=1,
        max_inflight_arenas=max_inflight,
    ).run(["a", "b", "c", "d"])


def test_multi_arena_inflight_overlaps_pipeline_stages():
    """Persistent Stage pools overlap Arena N+1 front with Arena N back."""

    serial = _timed(1)
    overlap = _timed(3)
    assert serial.get() == overlap.get() == ("a", "b", "c", "d")
    assert overlap.metrics["active_arenas_high_watermark"] == 3
    assert overlap.metrics["measured_wall_time_s"] < (
        serial.metrics["measured_wall_time_s"] * 0.82
    )
    first = {event.arena: event for event in overlap.timeline if event.stage == 1}
    second = {event.arena: event for event in overlap.timeline if event.stage == 2}
    assert max(
        first[1].worker_started_at,
        second[0].worker_started_at,
    ) < min(
        first[1].worker_finished_at,
        second[0].worker_finished_at,
    )
