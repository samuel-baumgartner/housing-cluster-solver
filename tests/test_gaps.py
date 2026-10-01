"""Gap analysis after solve — informs which housing shape fills leftover pockets."""

from __future__ import annotations

from housing.gaps import (
    GAP_PROTOTYPES,
    MICRO_2X3,
    ROW_4X2,
    SLIM_2X4,
    free_after_placement,
    gap_report,
    placements_for_line,
    placements_for_shape,
)
from housing.grid import WorldGrid
from housing.solver import solve
from housing.types import HousingLine


def _large_zone() -> WorldGrid:
    return WorldGrid.demo_deep_zone(
        zone_min=(8, 10),
        zone_max=(47, 39),
        path_y=40,
    )


def test_existing_lines_do_not_fit_inside_gap_pockets() -> None:
    """Current SMALL/BIG/L are too wide for the 1–2 cell slivers left after solve."""
    for factory in (WorldGrid.demo_deep_zone, _large_zone):
        grid = factory()
        result = solve(grid, record_steps=False)
        free = free_after_placement(grid, result.houses, result.planned_paths)
        for line in HousingLine:
            assert placements_for_line(line, free, grid, result.planned_paths) == set()


def test_gap_pockets_are_narrower_than_small_housing() -> None:
    grid = _large_zone()
    result = solve(grid, record_steps=False)
    report = gap_report(grid, result.houses, result.planned_paths, prototype_shapes=GAP_PROTOTYPES)
    hist = report["row_width_histogram"]
    assert hist.get(1, 0) > 0
    assert sum(c for w, c in hist.items() if w <= 2) > 0
    top_rects = report["top_inscribed_rects"]
    assert top_rects
    # No gap component contains a 4×4 block for current SMALL housing.
    assert all(max(w, h) < 4 or min(w, h) < 4 for (w, h), _ in top_rects[:5])


def test_slim_prototypes_cover_more_gap_than_current_lines() -> None:
    grid = _large_zone()
    result = solve(grid, record_steps=False)
    report = gap_report(grid, result.houses, result.planned_paths, prototype_shapes=GAP_PROTOTYPES)
    line_total = sum(report["line_placements"].values())
    slim_cover = report["shape_coverage"][SLIM_2X4.name]
    micro_cover = report["shape_coverage"][MICRO_2X3.name]
    assert line_total == 0
    assert slim_cover >= 100
    assert micro_cover >= slim_cover


def test_slim_2x4_beats_row_4x2_on_vertical_strips() -> None:
    """Default zone leaves 1×N column slivers; 2×4 fills them, 4×2 does not."""
    grid = WorldGrid.demo_deep_zone()
    result = solve(grid, record_steps=False)
    free = free_after_placement(grid, result.houses, result.planned_paths)
    slim = placements_for_shape(SLIM_2X4, free, grid, result.planned_paths)
    row = placements_for_shape(ROW_4X2, free, grid, result.planned_paths)
    assert len(slim) >= len(row)
    slim_cells = len(set().union(*slim)) if slim else 0
    row_cells = len(set().union(*row)) if row else 0
    assert slim_cells >= row_cells


def test_gap_report_tracks_free_mass_after_default_solve() -> None:
    grid = WorldGrid.demo_deep_zone()
    result = solve(grid, record_steps=False)
    report = gap_report(grid, result.houses, result.planned_paths, prototype_shapes=GAP_PROTOTYPES)
    assert report["free_cells"] >= 40
    assert report["largest_component"] >= 15
    assert report["shape_placements"][SLIM_2X4.name] > 0
