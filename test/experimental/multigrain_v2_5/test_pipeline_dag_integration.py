from __future__ import annotations

import time

import pytest

import rayorch.experimental.multigrain_v2_5 as mg


pytestmark = pytest.mark.usefixtures("ray_cluster")


class DagExpand:
    def run(self, parents):
        counts = {"zero": 0, "all": 2, "partial": 3}
        return [
            [f"{parent}:{ordinal}" for ordinal in range(counts[parent])]
            for parent in parents
        ]


class DagMap:
    def run(self, children):
        return [f"mapped({child})" for child in children]


class DagFilter:
    def run(self, children):
        return [
            "mapped(partial:" in child and not child.endswith(":1)")
            for child in children
        ]


class DagReduce:
    def run(self, anchors, members):
        return [
            f"{anchor}=>[{','.join(group)}]"
            for anchor, group in zip(anchors, members)
        ]


class FanoutPipeline(mg.Pipeline):
    def __init__(self):
        self.expand = mg.Expand(DagExpand).ray_options(
            replicas=1,
            batch_size=3,
        )
        self.map = mg.Map(DagMap).ray_options(
            replicas=1,
            batch_size=8,
        )
        self.filter = mg.Filter(DagFilter).ray_options(
            replicas=1,
            batch_size=8,
        )
        self.reduce = mg.Reduce(DagReduce).ray_options(
            replicas=1,
            batch_size=3,
        )

    def forward(self, parents):
        children = self.expand(parents)
        mapped = self.map(children)
        survivors = self.filter(mapped)
        return self.reduce(anchor=parents, members=survivors)


def test_native_pipeline_dag_zero_all_filtered_partial_and_reduce_order():
    result = mg.Executor(FanoutPipeline()).run(
        ["zero", "all", "partial"]
    )
    assert result.get() == (
        "zero=>[]",
        "all=>[]",
        "partial=>[mapped(partial:0),mapped(partial:2)]",
    )
    assert result.metrics["grains_per_rpc"] > 2.0


class Left:
    def run(self, values):
        return [f"L:{value}" for value in values]


class Right:
    def run(self, values):
        return [f"R:{value}" for value in values]


class MultiJoin:
    def run(self, left, right):
        return (
            [f"{a}|{b}" for a, b in zip(left, right)],
            [len(a) + len(b) for a, b in zip(left, right)],
        )


class DiamondPipeline(mg.Pipeline):
    def __init__(self):
        self.left = mg.Map(Left).ray_options(batch_size=3)
        self.right = mg.Map(Right).ray_options(batch_size=3)
        self.join = mg.Map(MultiJoin).ray_options(
            batch_size=3,
            num_outputs=2,
        )

    def forward(self, rows):
        left = self.left(rows)
        right = self.right(rows)
        return self.join(left, control=right)


def test_native_pipeline_diamond_multi_role_and_multi_output():
    result = mg.Executor(DiamondPipeline()).run(["a", "b", "c"])
    assert result.get() == (
        "L:a|R:a",
        "L:b|R:b",
        "L:c|R:c",
        6,
        6,
        6,
    )


class KeepFlag:
    def run(self, rows):
        return [row[1] for row in rows]


class KeptId:
    def run(self, rows):
        return [row[0] for row in rows]


class FilterPipeline(mg.Pipeline):
    def __init__(self):
        self.filter = mg.Filter(KeepFlag).ray_options(batch_size=3)
        self.map = mg.Map(KeptId).ray_options(batch_size=9)

    def forward(self, rows):
        return self.map(self.filter(rows))


def test_native_pipeline_filter_absence_skips_downstream_grains():
    result = mg.Executor(FilterPipeline()).run(
        [
            ("all-true-0", True),
            ("all-true-1", True),
            ("all-true-2", True),
            ("all-false-0", False),
            ("all-false-1", False),
            ("all-false-2", False),
            ("partial-0", True),
            ("partial-1", False),
            ("partial-2", True),
        ]
    )
    assert result.get() == (
        "all-true-0",
        "all-true-1",
        "all-true-2",
        "partial-0",
        "partial-2",
    )


class KeyRows:
    def run(self, rows):
        return [row[0] for row in rows]


class PairRows:
    def run(self, left, right):
        return [f"{a[1]}+{b[1]}" for a, b in zip(left, right)]


class RelatePipeline(mg.Pipeline):
    def __init__(self):
        self.left_key = mg.Map(KeyRows).ray_options(batch_size=4)
        self.right_key = mg.Map(KeyRows).ray_options(batch_size=4)
        self.relate = mg.Relate(PairRows).ray_options(batch_size=8)

    def forward(self, left, right):
        left_keys = self.left_key(left)
        right_keys = self.right_key(right)
        return self.relate(
            left=mg.keyed(left, by=left_keys),
            right=mg.keyed(right, by=right_keys),
        )


def test_native_pipeline_relate_duplicate_keys_cartesian_and_unmatched():
    result = mg.Executor(RelatePipeline()).run(
        [(1, "l0"), (1, "l1"), (2, "unmatched-left")],
        [(1, "r0"), (1, "r1"), (9, "unmatched-right")],
    )
    assert set(result.get()) == {
        "l0+r0",
        "l0+r1",
        "l1+r0",
        "l1+r1",
    }


class PoisonExpand:
    def run(self, parents):
        return [
            (
                ["good:0", "good:1"]
                if parent == "good"
                else ["bad:0", "poison"]
            )
            for parent in parents
        ]


class PoisonMap:
    def run(self, children):
        for index, child in enumerate(children):
            if child == "poison":
                raise mg.BadRecordError("poison child", index=index)
        return [f"mapped({child})" for child in children]


class AssembleHealthy:
    def run(self, anchors, members):
        return [
            f"{anchor}=>{','.join(group)}"
            for anchor, group in zip(anchors, members)
        ]


class FailureContainmentPipeline(mg.Pipeline):
    def __init__(self):
        self.expand = mg.Expand(PoisonExpand).ray_options(batch_size=2)
        self.map = mg.Map(PoisonMap).ray_options(
            batch_size=4,
            error_policy="isolate",
        )
        self.reduce = mg.Reduce(AssembleHealthy).ray_options(batch_size=2)

    def forward(self, parents):
        children = self.expand(parents)
        mapped = self.map(children)
        return self.reduce(anchor=parents, members=mapped)


def test_native_pipeline_bad_child_suppresses_only_its_parent_fiber():
    result = mg.Executor(FailureContainmentPipeline()).run(["good", "bad"])
    assert result.get() == ("good=>mapped(good:0),mapped(good:1)",)
    assert len(result.failures) == 1
    assert result.failures[0].failure.kind == "bad_record"
    assert result.failures[0].failure.message == "poison child"


class Timed:
    def __init__(self, stage: str, delay: float):
        self.stage = stage
        self.delay = delay

    def run(self, rows):
        start = time.monotonic()
        time.sleep(self.delay)
        stop = time.monotonic()
        outputs = []
        for row in rows:
            if isinstance(row, dict):
                value = row["value"]
                history = list(row["history"])
            else:
                value = row
                history = []
            history.append(
                {
                    "stage": self.stage,
                    "start": start,
                    "stop": stop,
                }
            )
            outputs.append({"value": value, "history": history})
        return outputs


class OverlapPipeline(mg.Pipeline):
    def __init__(self):
        self.first = (
            mg.Map(Timed)
            .pre_init("first", 0.35)
            .ray_options(batch_size=1)
        )
        self.second = (
            mg.Map(Timed)
            .pre_init("second", 0.35)
            .ray_options(batch_size=1)
        )

    def forward(self, rows):
        return self.second(self.first(rows))


def test_native_pipeline_sleeping_stages_overlap():
    first, second = mg.Executor(OverlapPipeline()).run(["a", "b"]).get()
    first_b = second["history"][0]
    second_a = first["history"][1]
    assert first_b["stage"] == "first"
    assert second_a["stage"] == "second"
    assert max(first_b["start"], second_a["start"]) < min(
        first_b["stop"],
        second_a["stop"],
    )
