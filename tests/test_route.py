from __future__ import annotations

from typing import Dict, List, Set, Tuple

from housing.grid import WorldGrid, neighbors4

Cell = Tuple[int, int]


def _route_hug_score(route: List[Cell], reserved: Set[Cell]) -> int:
    return sum(WorldGrid._building_touch_score(c, reserved) for c in route)


def _min_goal_distance(
    grid: WorldGrid,
    start: Cell,
    reserved: Set[Cell],
    planned: Set[Cell],
) -> int:
    walkable = grid._walkable_for_route(reserved, planned)
    dist: Dict[Cell, int] = {start: 0}
    q = [start]
    head = 0
    while head < len(q):
        c = q[head]
        head += 1
        for n in neighbors4(c):
            if n in dist or n not in walkable:
                continue
            dist[n] = dist[c] + 1
            q.append(n)
    goal_dists = [dist[g] for g in dist if g in grid.path_network(planned)]
    return min(goal_dists) if goal_dists else -1


def _best_hug_among_shortest(
    grid: WorldGrid,
    start: Cell,
    reserved: Set[Cell],
    planned: Set[Cell],
) -> int:
    walkable = grid._walkable_for_route(reserved, planned)
    dist: Dict[Cell, int] = {start: 0}
    q = [start]
    head = 0
    while head < len(q):
        c = q[head]
        head += 1
        for n in neighbors4(c):
            if n in dist or n not in walkable:
                continue
            dist[n] = dist[c] + 1
            q.append(n)

    goal_dists = [dist[g] for g in dist if g in grid.path_network(planned)]
    if not goal_dists:
        return -1
    min_d = min(goal_dists)
    best_goals = [g for g in dist if g in grid.path_network(planned) and dist[g] == min_d]

    touch = lambda c: WorldGrid._building_touch_score(c, reserved)
    hug: Dict[Cell, int] = {start: touch(start)}
    parent: Dict[Cell, Cell | None] = {start: None}

    for d in range(1, min_d + 1):
        for c in sorted(cell for cell, dd in dist.items() if dd == d):
            best_parent = None
            best_hug = -1
            for p in neighbors4(c):
                if dist.get(p) != d - 1 or p not in hug:
                    continue
                score = hug[p] + touch(c)
                if score > best_hug:
                    best_hug = score
                    best_parent = p
            if best_parent is not None:
                hug[c] = best_hug
                parent[c] = best_parent

    return max(hug.get(g, -1) for g in best_goals)


def test_route_minimizes_length() -> None:
    grid = WorldGrid(width=5, height=5)
    for x in range(5):
        for y in range(4):
            grid.zone.add((x, y))
        grid.paths.add((x, 4))
    grid.th_paths = set(grid.paths)
    grid.reserved = {(2, 1), (2, 2)}

    start = (0, 2)
    route = grid.route_to_network(start, grid.reserved, set())
    assert route
    assert len(route) == _min_goal_distance(grid, start, grid.reserved, set())


def test_route_hugs_buildings_on_tie() -> None:
    reserved = {(0, 2), (1, 2), (2, 1), (2, 2)}
    start = (0, 1)
    grid = WorldGrid(width=6, height=6)
    for x in range(6):
        for y in range(5):
            grid.zone.add((x, y))
        grid.paths.add((x, 5))
    grid.th_paths = set(grid.paths)

    route = grid.route_to_network(start, reserved, set())
    assert len(route) == _min_goal_distance(grid, start, reserved, set())
    assert _route_hug_score(route, reserved) == _best_hug_among_shortest(
        grid, start, reserved, set()
    )
    assert (1, 1) in route
    assert (0, 0) not in route


def test_route_taps_existing_planned_path() -> None:
    """Prefer joining an existing planned spur over paving to the main road."""
    grid = WorldGrid(width=8, height=7)
    for x in range(8):
        for y in range(5):
            grid.zone.add((x, y))
        grid.paths.add((x, 5))
    grid.th_paths = set(grid.paths)

    planned = {(4, 2), (4, 3), (4, 4)}
    start = (1, 2)
    route = grid.route_to_network(start, grid.reserved, planned)

    assert len(route) == 3
    assert route == [(1, 2), (2, 2), (3, 2)]
    assert (4, 2) not in route
