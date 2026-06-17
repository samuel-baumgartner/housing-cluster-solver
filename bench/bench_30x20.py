#!/usr/bin/env python3
"""Time headless solve on the 30×20 reference zone."""

from __future__ import annotations

import statistics
import time

from housing.grid import WorldGrid
from housing.solver import solve

RUNS = 5


def grid_30x20() -> WorldGrid:
    return WorldGrid.demo_deep_zone(
        zone_min=(8, 14),
        zone_max=(37, 33),
        path_y=34,
    )


def main() -> None:
    solve(grid_30x20(), record_steps=False)

    times: list[float] = []
    homes = 0
    paths = 0
    for _ in range(RUNS):
        grid = grid_30x20()
        t0 = time.perf_counter()
        result = solve(grid, record_steps=False)
        times.append(time.perf_counter() - t0)
        homes = len(result.houses)
        paths = len(result.planned_paths)

    print(f"Zone: 30×20")
    print(f"Houses: {homes}")
    print(f"Path tiles: {paths}")
    print(f"Runs: {RUNS}")
    print(f"Min:    {min(times):.3f}s")
    print(f"Median: {statistics.median(times):.3f}s")
    print(f"Max:    {max(times):.3f}s")


if __name__ == "__main__":
    main()
