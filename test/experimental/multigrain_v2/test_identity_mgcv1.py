from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from rayorch.experimental.multigrain_v2.identity import (
    canonical_encode,
    derive_expand_domain,
    derive_expand_entity,
    derive_input_salt,
    derive_node_id,
    derive_relate_domain,
    derive_relate_entity,
    derive_source_domain,
    derive_source_key_entity,
    derive_source_position_entity,
)
from test.experimental.multigrain_v2.reference_semantics.identity_oracle import (
    vectors as oracle_vectors,
)


GOLDEN_PATH = Path(__file__).parent / "golden" / "identity_mgcv1.json"


def production_vectors() -> dict[str, str]:
    input_salt = derive_input_salt(
        "example.pipe", "Pipe", "documents", 0
    )
    source_domain = derive_source_domain(input_salt)
    node = derive_node_id("example.pipe", "Pipe", "expand", "Expand")
    parent = derive_source_position_entity(source_domain, 3)
    return {
        "none": canonical_encode(None).hex(),
        "false": canonical_encode(False).hex(),
        "true": canonical_encode(True).hex(),
        "int_negative": canonical_encode(-12345678901234567890).hex(),
        "float": canonical_encode(1.5).hex(),
        "negative_zero": canonical_encode(-0.0).hex(),
        "nfc": canonical_encode("e\u0301").hex(),
        "bytes": canonical_encode(b"\x00\xff").hex(),
        "tuple": canonical_encode((None, True, 7, "x")).hex(),
        "node": node.value.hex(),
        "input_salt": input_salt.hex(),
        "source_domain": source_domain.value.hex(),
        "source_position": parent.value.hex(),
        "source_key": derive_source_key_entity(
            source_domain, ("doc", 7)
        ).value.hex(),
        "expand_domain": derive_expand_domain(node).value.hex(),
        "expand_entity": derive_expand_entity(
            node, source_domain, parent, 2
        ).value.hex(),
        "relate_domain": derive_relate_domain(node).value.hex(),
        "relate_entity": derive_relate_entity(
            node,
            (
                (source_domain, parent),
                (source_domain, parent),
            ),
        ).value.hex(),
    }


def test_production_matches_hand_frozen_and_independent_oracle() -> None:
    golden = json.loads(GOLDEN_PATH.read_text())
    assert oracle_vectors() == golden
    assert production_vectors() == golden


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        set(),
        object(),
        ([],),
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_canonical_encode_rejects_unsupported_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_encode(value)  # type: ignore[arg-type]


def test_encoding_is_type_sensitive_and_normalized() -> None:
    assert canonical_encode(1) != canonical_encode(1.0)
    assert canonical_encode(False) != canonical_encode(0)
    assert canonical_encode(-0.0) == canonical_encode(0.0)
    assert canonical_encode("e\u0301") == canonical_encode("\u00e9")


def test_builtin_scalar_and_tuple_subclasses_are_rejected() -> None:
    class CustomInt(int):
        pass

    class CustomFloat(float):
        pass

    class CustomStr(str):
        pass

    class CustomBytes(bytes):
        pass

    class CustomTuple(tuple):
        pass

    for value in (
        CustomInt(1),
        CustomFloat(1.0),
        CustomStr("x"),
        CustomBytes(b"x"),
        CustomTuple((1,)),
    ):
        with pytest.raises(TypeError):
            canonical_encode(value)  # type: ignore[arg-type]


def test_vectors_are_process_independent() -> None:
    code = (
        "import json;"
        "from test.experimental.multigrain_v2.test_identity_mgcv1 "
        "import production_vectors;"
        "print(json.dumps(production_vectors(),sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == json.loads(GOLDEN_PATH.read_text())
