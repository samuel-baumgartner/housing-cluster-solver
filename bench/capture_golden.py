#!/usr/bin/env python3
"""Capture golden solve result for regression / perf testing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from housing.grid import WorldGrid
from housing.solver import solve

BENCH_DIR = Path(__file__).resolve().parent
GOLDEN_PATHS = {
    "deep": BENCH_DIR / "golden_deep_15.json",
    "large": BENCH_DIR / "golden_large_zone.json",
}


def grid_for_scenario(scenario: str) -> WorldGrid:
    if scenario == "large":
        return WorldGrid.demo_deep_zone(
            zone_min=(8, 10),
            zone_max=(47, 39),
            path_y=40,
        )
    return WorldGrid.demo_deep_zone()


def result_to_dict(result) -> dict:
    return {
        "houses": [
            {
                "origin": list(h.origin),
                "line": int(h.line),
                "quarter_turns": h.quarter_turns,
                "flipped": h.flipped,
                "path_cell": list(h.path_cell) if h.path_cell else None,
            }
            for h in result.houses
        ],
        "planned_paths": sorted([list(c) for c in result.planned_paths]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture housing solver golden JSON")
    parser.add_argument(
        "--scenario",
        choices=sorted(GOLDEN_PATHS.keys()),
        default="deep",
        help="Zone fixture (default: deep)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing golden file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path",
    )
    args = parser.parse_args()

    grid = grid_for_scenario(args.scenario)
    result = solve(grid, record_steps=False)
    data = result_to_dict(result)
    text = json.dumps(data, indent=2) + "\n"

    if args.stdout:
        sys.stdout.write(text)
        return

    out_path = args.output or GOLDEN_PATHS[args.scenario]
    out_path.write_text(text)
    print(f"Wrote {len(result.houses)} houses to {out_path}")


if __name__ == "__main__":
    main()
