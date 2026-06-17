#!/usr/bin/env python3
"""CLI: diagnose a manual placement at a solver step (no GUI)."""

from __future__ import annotations

import argparse

from housing.diagnose import diagnose_placement
from housing.grid import WorldGrid
from housing.solver import solve
from housing.types import HousingLine


def _format_diagnosis(diag) -> str:
    lines = [
        "TRY PLACEMENT",
        diag.label,
        "",
        f"Score: {diag.score} (base {diag.score_base} + mix {diag.score_mix})",
        f"Door: {diag.door_cell}  Path: {diag.path_cell}",
        "",
    ]
    if diag.ok:
        lines.append("OK — would commit")
    else:
        lines.append("REJECTED")
    for reason in diag.reasons:
        lines.append(f"  • {reason}")
    for note in diag.warnings:
        if note != "Would commit successfully.":
            lines.append(f"  · {note}")
    return "\n".join(lines)


LINE_MAP = {
    "small": HousingLine.SMALL,
    "big": HousingLine.BIG,
    "l": HousingLine.L,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a housing placement at a solver step")
    parser.add_argument("--zone", default="deep", choices=["deep", "pocket", "shallow"])
    parser.add_argument("--step", type=int, required=True, metavar="N", help="Solver step (1-based)")
    parser.add_argument("--origin", required=True, help="Origin x,y e.g. 17,14")
    parser.add_argument("--line", default="small", choices=LINE_MAP.keys())
    parser.add_argument("--qt", type=int, default=0, help="Quarter turns 0-3")
    parser.add_argument("--flip", action="store_true")
    args = parser.parse_args()

    if args.zone == "deep":
        grid = WorldGrid.demo_deep_zone()
    elif args.zone == "pocket":
        grid = WorldGrid(width=48, height=36)
        for y in range(0, 31):
            grid.paths.add((10, y))
        for c in [(x, y) for y in range(16, 20) for x in range(11, 16)]:
            grid.zone.add(c)
        for c in [(x, y) for y in range(14, 20) for x in range(16, 24)]:
            grid.zone.add(c)
        grid.th_paths = {c for c in grid.paths if grid._path_connected_to_th(c)}
    else:
        grid = WorldGrid.demo_deep_zone(zone_min=(8, 18), zone_max=(35, 23), path_y=24)

    result = solve(grid, record_steps=True)
    idx = max(0, min(len(result.steps) - 1, args.step - 1))
    step = result.steps[idx]

    ox, oy = (int(p.strip()) for p in args.origin.split(","))
    grid.sync_solver_snapshot(step.reserved)
    diag = diagnose_placement(
        (ox, oy),
        LINE_MAP[args.line],
        args.qt % 4,
        args.flip,
        grid,
        step.houses,
        set(step.planned_paths),
        set(grid.th_paths),
    )
    print(_format_diagnosis(diag))


if __name__ == "__main__":
    main()
