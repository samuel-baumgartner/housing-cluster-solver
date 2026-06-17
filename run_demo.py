#!/usr/bin/env python3
"""Quick CLI run without GUI — prints placement log."""

from housing import WorldGrid, solve


def main() -> None:
    grid = WorldGrid.demo_deep_zone()
    print(f"Zone cells: {len(grid.zone)}")
    result = solve(grid, record_steps=False)
    print(f"Houses placed: {len(result.houses)}")
    print(f"Planned paths: {len(result.planned_paths)}")
    for i, h in enumerate(result.houses, 1):
        print(
            f"  {i:2d}. {h.label:5s} origin={h.origin} "
            f"qt={h.quarter_turns} flip={h.flipped} door={h.path_cell}"
        )


if __name__ == "__main__":
    main()
