"""Independent MGCV1 implementation; never import production identity code."""
from __future__ import annotations

import hashlib
import math
import struct
import unicodedata


def uleb128(value: int) -> bytes:
    if value < 0:
        raise ValueError("negative ULEB128")
    output = bytearray()
    while True:
        part = value & 127
        value //= 128
        output.append(part | (128 if value else 0))
        if not value:
            return bytes(output)


def frame(payload: bytes) -> bytes:
    return uleb128(len(payload)) + payload


def encode(value):
    if value is None:
        code, payload = b"N", b""
    elif type(value) is bool:
        code, payload = b"B", bytes((int(value),))
    elif type(value) is int:
        code, payload = b"I", ("%d" % value).encode("ascii")
    elif type(value) is float:
        if math.isnan(value) or math.isinf(value):
            raise ValueError("non-finite")
        code = b"F"
        payload = struct.pack(">d", 0.0 if value == 0 else value)
    elif type(value) is str:
        code = b"S"
        payload = unicodedata.normalize("NFC", value).encode("utf8")
    elif type(value) is bytes:
        code, payload = b"Y", value
    elif type(value) is tuple:
        code = b"T"
        payload = b"".join(frame(encode(child)) for child in value)
    else:
        raise TypeError(type(value).__name__)
    return code + frame(payload)


def digest(tag, *fields) -> bytes:
    message = bytearray(b"MGID/v1")
    for value in (tag,) + fields:
        message.extend(frame(encode(value)))
    return hashlib.sha256(message).digest()


def vectors() -> dict[str, str]:
    input_salt = digest("input", "example.pipe", "Pipe", "documents", 0)
    source_domain = digest("source-domain", input_salt)
    node = digest("node", "example.pipe", "Pipe", "expand", "Expand")
    parent = digest("source-position", source_domain, 3)
    return {
        "none": encode(None).hex(),
        "false": encode(False).hex(),
        "true": encode(True).hex(),
        "int_negative": encode(-12345678901234567890).hex(),
        "float": encode(1.5).hex(),
        "negative_zero": encode(-0.0).hex(),
        "nfc": encode("e\u0301").hex(),
        "bytes": encode(b"\x00\xff").hex(),
        "tuple": encode((None, True, 7, "x")).hex(),
        "node": node.hex(),
        "input_salt": input_salt.hex(),
        "source_domain": source_domain.hex(),
        "source_position": parent.hex(),
        "source_key": digest(
            "source-key", source_domain, encode(("doc", 7))
        ).hex(),
        "expand_domain": digest("expand-domain", node).hex(),
        "expand_entity": digest(node, source_domain, parent, 2).hex(),
        "relate_domain": digest("relate-domain", node).hex(),
        "relate_entity": digest(
            node, ((source_domain, parent), (source_domain, parent))
        ).hex(),
    }
