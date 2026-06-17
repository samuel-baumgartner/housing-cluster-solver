"""Footprints must not overlap planned street tiles."""

from housing.footprints import footprint_set
from housing.grid import WorldGrid
from housing.grid import neighbors4
from housing.scoring import would_commit
from housing.solver import solve
from housing.types import HousingLine


def test_can_place_footprint_rejects_planned_street() -> None:
    grid = WorldGrid.demo_deep_zone()
    planned = {(30, 17), (31, 17), (32, 17), (33, 17)}
    fp = footprint_set((30, 14), HousingLine.SMALL, 1, False)
    assert not grid.can_place_footprint(fp, planned)
    assert grid.can_place_footprint(fp)


def test_would_commit_rejects_footprint_on_planned_street() -> None:
    grid = WorldGrid.demo_deep_zone()
    planned = {(30, 17), (31, 17), (32, 17), (33, 17)}
    ok, _, _ = would_commit(
        (30, 14),
        HousingLine.SMALL,
        1,
        False,
        grid,
        grid.reserved,
        planned,
        set(grid.th_paths),
        True,
    )
    assert not ok


def test_solve_keeps_all_doors_connected() -> None:
    grid = WorldGrid.demo_deep_zone()
    result = solve(grid, record_steps=False)

    def door_connected(path_cell):
        walk = set(grid.zone) | grid.paths | result.planned_paths
        walk -= grid.reserved_from_houses(result.houses)
        seen = {path_cell}
        q = [path_cell]
        head = 0
        while head < len(q):
            c = q[head]
            head += 1
            if c in grid.th_paths or (
                c in grid.paths and grid._path_connected_to_th(c)
            ):
                return True
            for n in neighbors4(c):
                if n in seen or n not in walk:
                    continue
                seen.add(n)
                q.append(n)
        return False

    for i, h in enumerate(result.houses, 1):
        assert h.path_cell is not None
        assert door_connected(h.path_cell), f"house {i} door {h.path_cell} disconnected"
