"""Compare one completed Ray Data state against the frozen RayOrch result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def compare(state_path: Path) -> dict:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    result = state.get("full_result")
    if not isinstance(result, dict):
        raise RuntimeError("Ray Data state has no business full_result")
    benchmark = result.get("benchmark")
    if not isinstance(benchmark, dict):
        raise RuntimeError("Ray Data full_result has no benchmark summary")
    rayorch = json.loads(
        (HERE / "rayorch_reference.json").read_text(encoding="utf-8")
    )
    raydata_rate = float(benchmark["pages_per_s"])
    rayorch_rate = float(rayorch["pages_per_s"])
    return {
        "raydata": benchmark,
        "rayorch": rayorch,
        "comparison": {
            "pages_per_s_ratio_raydata_over_rayorch": round(
                raydata_rate / rayorch_rate, 6
            ),
            "throughput_delta_percent": round(
                (raydata_rate / rayorch_rate - 1.0) * 100.0, 3
            ),
            "measured_wall_delta_s": round(
                float(benchmark["measured_wall_s"])
                - float(rayorch["measured_wall_s"]),
                3,
            ),
        },
        "raydata_validation": result.get("validation"),
        "raydata_gpu_sampling": result.get("gpu_sampling"),
        "raydata_output_mode": result.get("output_mode"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raydata-state", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.raydata_state), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
