from housing.grid import WorldGrid
from housing.scoring import collect_frontier, origin_can_place, trim_frontier
from housing.solver import find_best_candidate, solve
from housing.types import HousingLine

import pytest


@pytest.mark.slow
def test_trim_frontier_keeps_all_placeable_origins() -> None:
    """Placeable cells must survive trim even when far from path/building anchors."""
    result = solve(WorldGrid.demo_deep_zone(), record_steps=True, landscape=False)
    checked = False
    for step in result.steps:
        if step.kind != "evaluate":
            continue
        grid = WorldGrid.demo_deep_zone()
        grid.reserved = set(step.reserved)
        planned = set(step.planned_paths)
        frontier = collect_frontier(grid, step.houses, planned, set(grid.th_paths))
        if len(frontier) <= 64:
            continue
        placeable = {o for o in frontier if origin_can_place(grid, o, planned)}
        if not placeable:
            continue
        trimmed = trim_frontier(frontier, grid, planned, 64)
        assert placeable <= trimmed
        checked = True
    assert checked


@pytest.mark.slow
def test_trim_frontier_finds_interior_pocket_before_false_stop() -> None:
    """Regression: interior top-row pocket must be searchable before saturation."""
    result = solve(WorldGrid.demo_deep_zone(), record_steps=True, landscape=False)
    step = next(
        s
        for s in result.steps
        if s.kind == "evaluate" and len(s.houses) == 13
    )
    grid = WorldGrid.demo_deep_zone()
    grid.reserved = set(step.reserved)
    planned = set(step.planned_paths)

    best, _, _ = find_best_candidate(
        grid, step.houses, planned, set(grid.th_paths), set(), True
    )
    assert best is not None
    assert best.origin[1] == 14
    assert best.line in (HousingLine.SMALL, HousingLine.L)


def test_solve_places_past_false_saturation_pocket() -> None:
    result = solve(WorldGrid.demo_deep_zone(), record_steps=False, landscape=False)
    assert len(result.houses) >= 14


def test_collect_frontier_includes_reserved_origin_corners() -> None:
    """Origins may sit on reserved cells when the footprint extends into free space."""
    grid = WorldGrid.demo_deep_zone(zone_min=(8, 14), zone_max=(37, 33), path_y=34)
    result = solve(grid, record_steps=True, landscape=False)
    step = next(s for s in result.steps if s.kind == "evaluate" and len(s.houses) == 4)
    g = WorldGrid.demo_deep_zone(zone_min=(8, 14), zone_max=(37, 33), path_y=34)
    g.sync_solver_snapshot(step.reserved)
    planned = set(step.planned_paths)

    frontier = collect_frontier(g, step.houses, planned, set(g.th_paths))
    assert (11, 26) in frontier

    best, _, _ = find_best_candidate(
        g, step.houses, planned, set(g.th_paths), set(), True
    )
    assert best is not None
    assert best.origin == (11, 26)
    assert best.quarter_turns == 3
    assert best.flipped is True
    assert best.score >= 326
