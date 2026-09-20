"""Internal Ray Jobs entrypoint."""

from __future__ import annotations

import argparse
import base64
from importlib import import_module
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-class", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("0", "1"), required=True)
    parser.add_argument("--profile-interval-s", type=float, required=True)
    args = parser.parse_args(argv)

    module_name, separator, attribute = args.benchmark_class.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("benchmark class must use the 'module:attribute' form")
    benchmark_type = getattr(import_module(module_name), attribute)
    if not isinstance(benchmark_type, type):
        raise TypeError(f"{args.benchmark_class} is not a class")

    config = json.loads(base64.urlsafe_b64decode(args.config).decode())
    report = benchmark_type(**config).run(
        ray_address="auto",
        run_id=args.run_id,
        profile=args.profile == "1",
        profile_interval_s=args.profile_interval_s,
    )
    report.print_summary()
    return 0


if __name__ == "__main__":  # pragma: no cover - Ray Jobs entrypoint
    raise SystemExit(main())
