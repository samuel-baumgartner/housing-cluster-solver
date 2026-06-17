from housing.grid import WorldGrid
from housing.landscape import DEFAULT_CIRCLE_CELLS, place_natural_circles
from housing.solver import solve


def test_place_natural_circles_carves_zone() -> None:
    grid = WorldGrid.demo_deep_zone()
    total = len(grid.zone)
    carved = place_natural_circles(grid, cells_per_circle=300, seed=1)
    assert 80 <= len(carved) <= 160
    assert carved == grid.landscape
    assert not carved & grid.zone
    assert len(grid.zone) == total - len(carved)


def test_solve_places_landscape_before_buildings() -> None:
    grid = WorldGrid.demo_deep_zone()
    result = solve(grid, record_steps=True, landscape=True)
    assert result.steps[0].kind == "init"
    assert result.steps[1].kind == "landscape"
    assert len(result.steps[1].landscape) >= 80
    assert result.steps[2].kind == "evaluate"
    assert not result.steps[2].planned_paths


def test_solve_without_landscape() -> None:
    grid = WorldGrid.demo_deep_zone()
    result = solve(grid, record_steps=True, landscape=False)
    assert result.steps[0].kind == "init"
    assert result.steps[1].kind == "evaluate"
    assert not grid.landscape


def test_solve_default_skips_landscape() -> None:
    grid = WorldGrid.demo_deep_zone()
    result = solve(grid, record_steps=True)
    assert result.steps[1].kind == "evaluate"
