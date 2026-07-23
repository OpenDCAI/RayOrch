from __future__ import annotations

import pytest

from rayorch.experimental import multigrain as mg
from rayorch.experimental.multigrain.ir import GraphValidationError


class OneChild:
    def run(self, rows):
        return [[f"{row}/child"] for row in rows]


class LeftTag:
    def run(self, rows):
        return [f"left:{row}" for row in rows]


class RightTag:
    def run(self, rows):
        return [f"right:{row}" for row in rows]


class MergeTags:
    def run(self, left, right):
        return [f"{a}|{b}" for a, b in zip(left, right)]


class PairRoots:
    def run(self, left, right):
        return list(zip(left, right))


class CountGroups:
    def run(self, anchors, groups):
        return [len(group) for group in groups]


class DirectRoleBoundaryPipe(mg.Pipeline):
    def __init__(self) -> None:
        super().__init__()
        self.pair = mg.Relate(
            PairRoots,
            roles=("left", "right"),
            relation_adapter=(
                "test.experimental.multigrain.relate_adapters:"
                "pair_left_right_by_index"
            ),
        )
        self.expand = mg.Expand(OneChild, child_label="pair_item")
        self.reduce = mg.Reduce(CountGroups)

    def forward(self, left, right):
        children = self.expand(self.pair(left, right))
        return self.reduce(mg.group_by(left, mg.via(children, role="left")))


def test_by_role_is_direct_evidence_and_does_not_cross_expand() -> None:
    with pytest.raises(
        GraphValidationError,
        match="role 'left' is not available",
    ):
        DirectRoleBoundaryPipe().compile()


def test_same_grain_independent_roots_keep_distinct_domains() -> None:
    left = mg.source(["a"], name="item", identity_domain="left-root")
    right = mg.source(["a"], name="item", identity_domain="right-root")
    pairs = mg.Relate(
        PairRoots,
        roles=("left", "right"),
        relation_fn=lambda values: [
            (values[0], {"left": 0, "right": 0})
        ],
        output_grain="pair",
    )(left, right)

    assert left.identity_domain != right.identity_domain
    assert pairs.ancestors[0][left.identity_domain] == left.record_ids[0]
    assert pairs.ancestors[0][right.identity_domain] == right.record_ids[0]
    assert {
        ref.identity_domain for ref in pairs.relations[0]
    } == {left.identity_domain, right.identity_domain}


def test_source_name_defines_default_admission_domain() -> None:
    first_batch = mg.source(["a"], name="rows")
    next_batch = mg.source(["b"], name="rows")
    independent = mg.source(
        ["c"],
        name="rows",
        identity_domain=mg.IdentityDomain.fresh("rows"),
    )

    assert first_batch.identity_domain == next_batch.identity_domain
    assert independent.identity_domain != first_batch.identity_domain


def test_repeated_grain_nested_expand_does_not_overwrite_ancestry() -> None:
    roots = mg.source(["root"], name="unit")
    children = mg.Expand(OneChild, child_label="unit")(roots)
    grandchildren = mg.Expand(OneChild, child_label="unit")(children)

    assert roots.grain == children.grain == grandchildren.grain == "unit"
    assert len(
        {
            roots.identity_domain,
            children.identity_domain,
            grandchildren.identity_domain,
        }
    ) == 3
    assert grandchildren.ancestors[0][roots.identity_domain] == roots.record_ids[0]
    assert (
        grandchildren.ancestors[0][children.identity_domain]
        == children.record_ids[0]
    )


def test_shared_root_diamond_merges_metadata_without_duplicate_domain() -> None:
    roots = mg.source(["a", "b"], name="rows")
    left = mg.Map(LeftTag, name="left")(roots)
    right = mg.Map(RightTag, name="right")(roots)
    merged = mg.Map(MergeTags, name="merge")(left, right)

    assert merged.identity_domain == roots.identity_domain
    assert merged.lineage == [
        ("left", "right", "merge"),
        ("left", "right", "merge"),
    ]
    assert all(
        ancestry == {roots.identity_domain: record_id}
        for ancestry, record_id in zip(merged.ancestors, roots.record_ids)
    )
