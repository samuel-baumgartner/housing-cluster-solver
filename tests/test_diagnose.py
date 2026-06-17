from housing.diagnose import diagnose_placement
from housing.grid import WorldGrid
from housing.types import HousingLine


def test_diagnose_rejects_planned_street() -> None:
    grid = WorldGrid.demo_deep_zone()
    planned = {(30, 17), (31, 17), (32, 17), (33, 17)}
    diag = diagnose_placement(
        (30, 14),
        HousingLine.SMALL,
        1,
        False,
        grid,
        [],
        planned,
        set(grid.th_paths),
    )
    assert not diag.ok
    assert any("planned street" in r for r in diag.reasons)


def test_diagnose_door_unreachable() -> None:
    grid = WorldGrid(width=20, height=20)
    for y in range(14, 20):
        for x in range(10, 16):
            grid.zone.add((x, y))
    diag = diagnose_placement(
        (12, 15),
        HousingLine.SMALL,
        0,
        False,
        grid,
        [],
        set(),
        set(),
    )
    assert not diag.ok
    assert any("path network" in r for r in diag.reasons)
