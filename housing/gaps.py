from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

from .footprints import footprint_set, orientation_variants
from .grid import WorldGrid
from .scoring import connected_components
from .types import Cell, House, HousingLine

CellShape = Tuple[Cell, ...]


@dataclass(frozen=True)
class PrototypeShape:
    """Axis-aligned footprint in local coordinates (origin = min corner)."""

    name: str
    cells: CellShape

    @property
    def width(self) -> int:
        return max(x for x, _ in self.cells) + 1

    @property
    def height(self) -> int:
        return max(y for _, y in self.cells) + 1

    @property
    def cell_count(self) -> int:
        return len(self.cells)


def _rotate_local(cell: Cell, quarter_turns: int, bbox: Tuple[int, int]) -> Cell:
    x, y = cell
    bw, bh = bbox
    qt = quarter_turns % 4
    if qt == 0:
        return x, y
    if qt == 1:
        return y, bw - 1 - x
    if qt == 2:
        return bw - 1 - x, bh - 1 - y
    return bh - 1 - y, x


def oriented_cells(
    shape: PrototypeShape, quarter_turns: int, flipped: bool
) -> CellShape:
    bw, bh = shape.width, shape.height
    out: List[Cell] = []
    for x, y in shape.cells:
        lx, ly = x, y
        if flipped:
            lx = bw - 1 - lx
        out.append(_rotate_local((lx, ly), quarter_turns, (bw, bh)))
    return tuple(out)


def footprint_from_shape(
    origin: Cell, shape: PrototypeShape, quarter_turns: int, flipped: bool
) -> Set[Cell]:
    ox, oy = origin
    return {(ox + x, oy + y) for x, y in oriented_cells(shape, quarter_turns, flipped)}


def free_after_placement(
    grid: WorldGrid, houses: Iterable[House], planned_paths: Set[Cell]
) -> Set[Cell]:
    reserved = set(grid.reserved)
    for house in houses:
        reserved.update(
            footprint_set(
                house.origin, house.line, house.quarter_turns, house.flipped
            )
        )
    blocked = reserved | planned_paths
    return {c for c in grid.zone if c not in blocked}


def placements_for_shape(
    shape: PrototypeShape,
    free: Set[Cell],
    grid: WorldGrid,
    planned_paths: Set[Cell],
) -> Set[FrozenSet[Cell]]:
    placements: Set[FrozenSet[Cell]] = set()
    for origin in free:
        for qt in range(4):
            for flipped in (False, True):
                fp = footprint_from_shape(origin, shape, qt, flipped)
                if fp <= free and grid.can_place_footprint(fp, planned_paths):
                    placements.add(frozenset(fp))
    return placements


def placements_for_line(
    line: HousingLine,
    free: Set[Cell],
    grid: WorldGrid,
    planned_paths: Set[Cell],
) -> Set[FrozenSet[Cell]]:
    placements: Set[FrozenSet[Cell]] = set()
    for origin in free:
        for qt, flipped in orientation_variants(line):
            fp = footprint_set(origin, line, qt, flipped)
            if fp and fp <= free and grid.can_place_footprint(fp, planned_paths):
                placements.add(frozenset(fp))
    return placements


def max_inscribed_rectangles(comp: Set[Cell]) -> List[Tuple[int, int]]:
    """All (w, h) sizes of axis-aligned rectangles fully contained in comp."""
    sizes: Set[Tuple[int, int]] = set()
    if len(comp) < 2:
        return []
    cells = comp
    xs = sorted({x for x, _ in cells})
    ys = sorted({y for _, y in cells})
    for x in xs:
        for y in ys:
            if (x, y) not in cells:
                continue
            for w in range(1, 13):
                for h in range(1, 13):
                    if all((x + dx, y + dy) in cells for dx in range(w) for dy in range(h)):
                        sizes.add((w, h))
    return sorted(sizes, key=lambda wh: wh[0] * wh[1], reverse=True)


def row_width_histogram(free: Set[Cell]) -> Dict[int, int]:
    """How many cells sit in rows of each width (contiguous x-run per y)."""
    by_y: Dict[int, List[int]] = {}
    for x, y in free:
        by_y.setdefault(y, []).append(x)
    hist: Dict[int, int] = {}
    for y, xs in by_y.items():
        xs.sort()
        run = 1
        for i in range(1, len(xs)):
            if xs[i] == xs[i - 1] + 1:
                run += 1
            else:
                hist[run] = hist.get(run, 0) + run
                run = 1
        hist[run] = hist.get(run, 0) + run
    return hist


def gap_report(
    grid: WorldGrid,
    houses: List[House],
    planned_paths: Set[Cell],
    *,
    prototype_shapes: Iterable[PrototypeShape],
) -> Dict[str, object]:
    free = free_after_placement(grid, houses, planned_paths)
    comps = connected_components(free)
    comps.sort(key=len, reverse=True)

    line_placements = {
        line.name: len(placements_for_line(line, free, grid, planned_paths))
        for line in HousingLine
    }
    shape_placements = {
        shape.name: len(placements_for_shape(shape, free, grid, planned_paths))
        for shape in prototype_shapes
    }
    shape_coverage = {
        shape.name: len(
            set().union(
                *placements_for_shape(shape, free, grid, planned_paths)
            )
            if placements_for_shape(shape, free, grid, planned_paths)
            else set()
        )
        for shape in prototype_shapes
    }

    rect_sizes: Dict[Tuple[int, int], int] = {}
    for comp in comps:
        for wh in max_inscribed_rectangles(comp):
            rect_sizes[wh] = rect_sizes.get(wh, 0) + 1

    return {
        "free_cells": len(free),
        "components": len(comps),
        "largest_component": len(comps[0]) if comps else 0,
        "line_placements": line_placements,
        "shape_placements": shape_placements,
        "shape_coverage": shape_coverage,
        "row_width_histogram": row_width_histogram(free),
        "top_inscribed_rects": sorted(
            rect_sizes.items(), key=lambda kv: kv[0][0] * kv[0][1], reverse=True
        )[:10],
    }


# Reference prototypes for gap-fill analysis (not yet in HousingLine).
SLIM_2X4 = PrototypeShape("slim_2x4", tuple((x, y) for y in range(4) for x in range(2)))
MICRO_2X3 = PrototypeShape("micro_2x3", tuple((x, y) for y in range(3) for x in range(2)))
ROW_4X2 = PrototypeShape("row_4x2", tuple((x, y) for y in range(2) for x in range(4)))
NARROW_3X4 = PrototypeShape("narrow_3x4", tuple((x, y) for y in range(4) for x in range(3)))

GAP_PROTOTYPES = (SLIM_2X4, MICRO_2X3, ROW_4X2, NARROW_3X4)
