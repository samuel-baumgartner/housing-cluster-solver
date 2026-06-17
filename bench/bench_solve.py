#!/usr/bin/env python3
"""Time headless solve on demo_deep_zone."""

from __future__ import annotations

import statistics
import time

from housing.grid import WorldGrid
from housing.solver import solve

RUNS = 5


def main() -> None:
    # Warm numba JIT before timed runs.
    solve(WorldGrid.demo_deep_zone(), record_steps=False)

    times: list[float] = []
    homes = 0
    for _ in range(RUNS):
        grid = WorldGrid.demo_deep_zone()
        t0 = time.perf_counter()
        result = solve(grid, record_steps=False)
        times.append(time.perf_counter() - t0)
        homes = len(result.houses)

    print(f"Houses: {homes}")
    print(f"Runs: {RUNS}")
    print(f"Min:    {min(times):.3f}s")
    print(f"Median: {statistics.median(times):.3f}s")
    print(f"Max:    {max(times):.3f}s")


if __name__ == "__main__":
    main()
