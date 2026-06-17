#!/usr/bin/env python3
"""cProfile headless solve."""

from __future__ import annotations

import cProfile
import pstats
from io import StringIO

from housing.grid import WorldGrid
from housing.solver import solve


def main() -> None:
    grid = WorldGrid.demo_deep_zone()
    profiler = cProfile.Profile()
    profiler.enable()
    solve(grid, record_steps=False)
    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
