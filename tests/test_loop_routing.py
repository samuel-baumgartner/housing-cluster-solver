from __future__ import annotations

from housing.grid import (
    LOOP_TILE_VALUE,
    MIN_LOOP_INTERIOR_AREA,
    MIN_LOOP_PERIMETER,
    WorldGrid,
)


def test_is_building_loop_requires_area_and_perimeter() -> None:
    reserved = {(3, 1), (4, 1), (3, 2), (4, 2)}  # 2x2 — too small
    border = WorldGrid._ring_border(3, 1, 2, 2)
    network = set(border)
    assert not WorldGrid.is_building_loop(network, reserved, 3, 1, 2, 2)

    reserved_big = {(3, 1), (4, 1), (5, 1), (3, 2), (4, 2), (5, 2)}
    border_big = WorldGrid._ring_border(3, 1, 3, 2)
    network_big = set(border_big)
    assert len(WorldGrid._interior_cells(3, 1, 3, 2)) >= MIN_LOOP_INTERIOR_AREA
    assert len(border_big) >= MIN_LOOP_PERIMETER
    assert WorldGrid.is_building_loop(network_big, reserved_big, 3, 1, 3, 2)


def test_is_building_loop_requires_building_enclosure() -> None:
    border = WorldGrid._ring_border(3, 1, 3, 2)
    network = set(border)
    assert not WorldGrid.is_building_loop(network, set(), 3, 1, 3, 2)


def test_new_loops_when_adding_counts_closure() -> None:
    reserved = {(3, 1), (4, 1), (5, 1), (3, 2), (4, 2), (5, 2)}
    border = WorldGrid._ring_border(3, 1, 3, 2)
    missing = (2, 0)
    before = border - {missing}
    after = border
    assert WorldGrid.new_loops_when_adding(missing, before, after, reserved) == 1


def test_route_prefers_loop_when_effective_cost_wins() -> None:
    """Complete a nearly-closed ring when the loop credit beats a few extra tiles."""
    grid = WorldGrid(width=8, height=7)
    for x in range(8):
        for y in range(5):
            grid.zone.add((x, y))
        grid.paths.add((x, 5))
    grid.th_paths = set(grid.paths)

    reserved = {(3, 1), (4, 1), (5, 1), (3, 2), (4, 2), (5, 2)}
    border = WorldGrid._ring_border(3, 1, 3, 2)
    planned = border - {(2, 0)}

    start = (1, 1)
    route = grid.route_to_network(start, reserved, planned)

    assert (2, 0) in route
    network = grid.path_network(planned) | set(route)
    assert WorldGrid.is_building_loop(network, reserved, 3, 1, 3, 2)

    # Direct drop to the road is shorter in raw tiles but worse effective cost.
    direct = grid.route_to_network(start, reserved, planned | {(2, 0)})
    assert len(direct) < len(route)
    assert len(route) - LOOP_TILE_VALUE <= len(direct)
