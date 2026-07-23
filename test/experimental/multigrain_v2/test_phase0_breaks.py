from __future__ import annotations

import pytest

from rayorch.experimental.multigrain_v2.graph import (
    CompiledInputGroup,
    CompiledPort,
    ExpandSpec,
    FilterSpec,
    MapSpec,
    RelateRoleSpec,
    RelateSpec,
)
from rayorch.experimental.multigrain_v2.identity import (
    DomainId,
    EntityId,
    NodeId,
    PortId,
    stable_hash,
)


def pid(name: str) -> PortId:
    return PortId(stable_hash("break-port", name))


def test_invalid_filter_contracts_fail_before_runtime() -> None:
    target = pid("target")
    control = pid("control")
    with pytest.raises(ValueError):
        FilterSpec("predicate", ("x",), (), (target,))
    with pytest.raises(ValueError):
        FilterSpec("mask", (), (control,), (), 0)
    with pytest.raises(ValueError):
        FilterSpec(
            "select",
            ("x",),
            (control,),
            (target,),
            0,
        )


def test_map_and_expand_identity_authority_must_be_first_role() -> None:
    first = pid("first")
    second = pid("second")
    with pytest.raises(ValueError, match="first input"):
        MapSpec(second, ("first", "second"), (first, second))
    with pytest.raises(ValueError, match="first input"):
        ExpandSpec(second, ("first", "second"), (first, second))
    with pytest.raises(ValueError, match="unique"):
        MapSpec(first, ("same", "same"), (first, second))


def test_mixed_or_duplicate_relate_roles_fail_before_runtime() -> None:
    value = pid("value")
    key = pid("key")
    with pytest.raises(ValueError, match="key port"):
        RelateSpec(
            "key",
            (RelateRoleSpec("rows", value, None),),
        )
    with pytest.raises(ValueError, match="cannot have"):
        RelateSpec(
            "custom",
            (RelateRoleSpec("rows", value, key),),
        )
    with pytest.raises(ValueError, match="unique"):
        RelateSpec(
            "custom",
            (
                RelateRoleSpec("rows", value, None),
                RelateRoleSpec("rows", pid("other"), None),
            ),
        )


def test_names_and_graph_indexes_require_exact_types() -> None:
    value = pid("typed-value")
    domain = DomainId(stable_hash("break-domain", "typed"))
    with pytest.raises(ValueError, match="string"):
        RelateRoleSpec(1, value, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="string"):
        CompiledInputGroup(1, 0, domain, value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative int"):
        CompiledInputGroup("rows", True, domain, value)
    with pytest.raises(ValueError, match="non-negative int"):
        CompiledPort(value, None, True, domain)
    with pytest.raises(ValueError, match="non-negative int"):
        FilterSpec("select", ("rows",), (value,), (value,), True)


@pytest.mark.parametrize(
    "wrapper",
    [NodeId, PortId, DomainId, EntityId],
)
def test_stable_identity_wrappers_reject_truncated_hashes(wrapper) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        wrapper(b"short")
