"""MinerU full matrix 命令生成器测试。"""

from __future__ import annotations

from rayorch.experimental.multigrain_v3.benchmark.mineru_matrix import (
    build_commands,
    build_parser,
)


def test_matrix_generates_four_modes_per_repeat(tmp_path) -> None:
    """每次 repeat 必须包含 elastic、parent、Ray Data 和 native。"""

    args = build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--limit",
            "48",
            "--repeats",
            "2",
        ]
    )
    commands = build_commands(args)

    assert len(commands) == 8
    assert [item["engine"] for item in commands[:4]] == [
        "v3_elastic",
        "v3_parent_bound",
        "ray_data",
        "native",
    ]
    assert [item["engine"] for item in commands[4:]] == [
        "native",
        "ray_data",
        "v3_parent_bound",
        "v3_elastic",
    ]
    assert all("--limit 48" in item["command"] for item in commands)
