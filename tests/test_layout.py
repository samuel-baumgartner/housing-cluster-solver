from housing.grid import WorldGrid
from housing.layout import (
    LAYOUT_MIN_ZONE,
    district_loop_closed,
    layout_enabled,
    plan_district_layout,
    ring_has_no_path_gaps,
)
from housing.solver import solve


def test_district_loops_are_closed() -> None:
    grid = _large_zone()
    result = plan_district_layout(grid)
    assert result is not None
    districts, planned = result
    zone = set(grid.zone)
    assert ring_has_no_path_gaps(planned, grid, zone)
    for d in districts:
        assert district_loop_closed(d, planned, grid, zone)


def test_no_full_perimeter_parallel_to_paths() -> None:
    grid = _large_zone()
    result = plan_district_layout(grid)
    assert result is not None
    _, planned = result
    zone = grid.zone
    xmin = min(c[0] for c in zone)
    ymax = max(c[1] for c in zone)
    zone_w = max(c[0] for c in zone) - xmin + 1
    zone_h = ymax - min(c[1] for c in zone) + 1
    left_col = {c for c in planned if c[0] == xmin}
    bottom_row = {c for c in planned if c[1] == ymax}
    assert len(left_col) < zone_h // 2
    assert len(bottom_row) < zone_w // 2


def _large_zone() -> WorldGrid:
    return WorldGrid.demo_deep_zone(
        zone_min=(8, 10),
        zone_max=(47, 39),
        path_y=40,
    )


def test_layout_skipped_on_small_zone() -> None:
    grid = WorldGrid.demo_deep_zone()
    assert len(grid.zone) < LAYOUT_MIN_ZONE
    assert not layout_enabled(len(grid.zone))
    assert plan_district_layout(grid) is None


def test_partition_sizing_on_large_zone() -> None:
    grid = _large_zone()
    assert len(grid.zone) >= 1200
    result = plan_district_layout(grid)
    assert result is not None
    districts, planned = result
    assert len(districts) >= 3
    assert len(planned) > 0
    for d in districts:
        assert 150 <= len(d.cells) <= 450
        assert len(d.interior) >= 100
        assert d.ring


def test_district_interiors_reachable() -> None:
    grid = _large_zone()
    result = plan_district_layout(grid)
    assert result is not None
    districts, planned = result
    grid.bump_planned_version()
    grid._ensure_accel().sync_planned(planned)
    accel = grid._ensure_accel()
    reach = accel.build_reach_zone_free(planned)
    for d in districts:
        assert any(reach[c[1], c[0]] for c in d.interior)


def test_large_zone_solve_uses_layout() -> None:
    grid = _large_zone()
    result = solve(grid, record_steps=True)
    kinds = [s.kind for s in result.steps]
    assert "layout" in kinds
    assert "district" in kinds
    assert len(result.houses) >= 4


def test_golden_unchanged_on_default_zone() -> None:
    from tests.test_golden_solve import _load_golden, _result_payload

    golden = _load_golden()
    grid = WorldGrid.demo_deep_zone()
    result = solve(grid, record_steps=False)
    assert _result_payload(result) == golden
    assert not any(s.kind == "layout" for s in result.steps)
