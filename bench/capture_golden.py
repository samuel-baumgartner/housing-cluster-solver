#!/usr/bin/env python3
"""Capture golden solve result for regression / perf testing."""

from __future__ import annotations

import json
from pathlib import Path

from housing.grid import WorldGrid
from housing.solver import solve

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_deep_15.json"


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
    grid = WorldGrid.demo_deep_zone()
    result = solve(grid, record_steps=False)
    data = result_to_dict(result)
    GOLDEN_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {len(result.houses)} houses to {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
