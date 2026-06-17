#!/usr/bin/env python3
"""Time solve on a large zone where district pre-layout is enabled."""

from __future__ import annotations

import statistics
import time

from housing.grid import WorldGrid
from housing.layout import layout_enabled
from housing.solver import solve

RUNS = 5


def large_zone() -> WorldGrid:
    return WorldGrid.demo_deep_zone(
        zone_min=(8, 10),
        zone_max=(47, 39),
        path_y=40,
    )


def main() -> None:
    grid = large_zone()
    assert layout_enabled(len(grid.zone)), "bench zone must trigger layout"

    solve(grid, record_steps=False)

    times: list[float] = []
    homes = 0
    paths = 0
    for _ in range(RUNS):
        g = large_zone()
        t0 = time.perf_counter()
        result = solve(g, record_steps=False)
        times.append(time.perf_counter() - t0)
        homes = len(result.houses)
        paths = len(result.planned_paths)

    print(f"Zone cells: {len(grid.zone)}")
    print(f"Houses: {homes}")
    print(f"Planned paths: {paths}")
    print(f"Runs: {RUNS}")
    print(f"Min:    {min(times):.3f}s")
    print(f"Median: {statistics.median(times):.3f}s")
    print(f"Max:    {max(times):.3f}s")


if __name__ == "__main__":
    main()
