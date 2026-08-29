"""Ray Data frontend for the shared TaiJi MinerU orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from experiments.taiji_multinode_mineru import orchestrate as shared


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"plan", "submit"}:
        if "--engine" not in arguments:
            arguments[1:1] = ["--engine", "raydata"]
        if "--state-file" not in arguments:
            run_index = arguments.index("--run-id") + 1
            arguments.extend(
                [
                    "--state-file",
                    str(HERE / "state" / f"{arguments[run_index]}.json"),
                ]
            )
    return shared.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
