from __future__ import annotations

from functools import lru_cache
from typing import FrozenSet, List, Set, Tuple

from .types import Cell, HousingLine

# Stage-1 footprints — mirrors my-colony-sim `data/buildings.csv` +
# `data/building_footprints.csv` via BuildingDimensionLog.world_footprint_cells().
#
# Small / big: full axis-aligned rectangle (ceili mesh footprint).
# L: custom `housing_l` profile (39 cells, top rows full width, lower rows 3 wide).

_SMALL: List[Cell] = [(x, y) for y in range(4) for x in range(4)]
_BIG: List[Cell] = [(x, y) for y in range(4) for x in range(5)]
_L: List[Cell] = []
for _y in range(8):
    if _y < 3:
        _L.extend((x, _y) for x in range(8))
    else:
        _L.extend((x, _y) for x in range(3))

BASE_FOOTPRINTS = {
    HousingLine.SMALL: _SMALL,
    HousingLine.BIG: _BIG,
    HousingLine.L: _L,
}

BBOX = {
    HousingLine.SMALL: (4, 4),
    HousingLine.BIG: (5, 4),
    HousingLine.L: (8, 8),
}

# Entrance offsets from `data/buildings.csv` (stage-1 housing kinds).
_ENTRANCE_DOOR_LOCAL: dict[HousingLine, Cell] = {
    HousingLine.SMALL: (1, 3),
    HousingLine.BIG: (2, 3),
    HousingLine.L: (1, 7),
}
_ENTRANCE_PATH_STEP_LOCAL: dict[HousingLine, Cell] = {
    HousingLine.SMALL: (0, 1),
    HousingLine.BIG: (0, 1),
    HousingLine.L: (0, 1),
}


def _flip_footprint_offset(offset: Cell, bbox: Tuple[int, int]) -> Cell:
    x, y = offset
    bw, _ = bbox
    return bw - 1 - x, y


def _rotate_footprint_offset(offset: Cell, bbox: Tuple[int, int], quarter_turns: int) -> Cell:
    x, y = offset
    bw, bh = bbox
    qt = quarter_turns % 4
    if qt == 0:
        return x, y
    if qt == 1:
        return y, bw - 1 - x
    if qt == 2:
        return bw - 1 - x, bh - 1 - y
    return bh - 1 - y, x


def _rotate_footprint_vector(vec: Cell, quarter_turns: int) -> Cell:
    x, y = vec
    qt = quarter_turns % 4
    if qt == 0:
        return x, y
    if qt == 1:
        return y, -x
    if qt == 2:
        return -x, -y
    return -y, x


def _rotate_cell(cell: Cell, quarter_turns: int, bbox: Tuple[int, int]) -> Cell:
    return _rotate_footprint_offset(cell, bbox, quarter_turns)


def footprint_cells(
    origin: Cell,
    line: HousingLine,
    quarter_turns: int = 0,
    flipped: bool = False,
) -> List[Cell]:
    ox, oy = origin
    base = BASE_FOOTPRINTS[line]
    bw, bh = BBOX[line]
    cells: List[Cell] = []
    for cx, cy in base:
        lx, ly = cx, cy
        if flipped:
            lx = bw - 1 - lx
        rx, ry = _rotate_cell((lx, ly), quarter_turns, (bw, bh))
        cells.append((ox + rx, oy + ry))
    return cells


def orientation_variants(line: HousingLine) -> List[Tuple[int, bool]]:
    out: List[Tuple[int, bool]] = []
    custom = line == HousingLine.L
    for qt in range(4):
        out.append((qt, False))
        if custom or qt in (0, 2):
            out.append((qt, True))
    return out


def _build_orient_offset_tables() -> dict[tuple[int, int, bool], Tuple[Cell, ...]]:
    table: dict[tuple[int, int, bool], Tuple[Cell, ...]] = {}
    for line in HousingLine:
        for qt, flip in orientation_variants(line):
            cells = tuple(footprint_cells((0, 0), line, qt, flip))
            table[(int(line), qt, flip)] = cells
    return table


_ORIENT_OFFSETS = _build_orient_offset_tables()


@lru_cache(maxsize=131072)
def _footprint_frozen(
    origin: Cell,
    line: int,
    quarter_turns: int,
    flipped: bool,
) -> FrozenSet[Cell]:
    ox, oy = origin
    offsets = _ORIENT_OFFSETS.get((line, quarter_turns, flipped))
    if offsets is None:
        return frozenset(
            footprint_cells(origin, HousingLine(line), quarter_turns, flipped)
        )
    return frozenset((ox + dx, oy + dy) for dx, dy in offsets)


def footprint_set(
    origin: Cell,
    line: HousingLine,
    quarter_turns: int = 0,
    flipped: bool = False,
) -> FrozenSet[Cell]:
    return _footprint_frozen(origin, int(line), quarter_turns, flipped)


def entrance_door_offset(
    line: HousingLine,
    quarter_turns: int = 0,
    flipped: bool = False,
) -> Cell:
    """Local door cell offset — mirrors BuildingDimensionLog.entrance_door_offset."""
    door = _ENTRANCE_DOOR_LOCAL[line]
    bbox = BBOX[line]
    if flipped:
        door = _flip_footprint_offset(door, bbox)
    return _rotate_footprint_offset(door, bbox, quarter_turns)


def entrance_path_offset(
    line: HousingLine,
    quarter_turns: int = 0,
    flipped: bool = False,
) -> Cell:
    """Step from door cell to the path tile — mirrors BuildingDimensionLog.entrance_path_offset."""
    pdx, pdz = _ENTRANCE_PATH_STEP_LOCAL[line]
    if flipped:
        pdx = -pdx
    return _rotate_footprint_vector((pdx, pdz), quarter_turns)


def entrance_door_cell(
    origin: Cell,
    line: HousingLine,
    quarter_turns: int = 0,
    flipped: bool = False,
) -> Cell:
    """Occupied footprint cell where the door sits."""
    ox, oy = origin
    dx, dy = entrance_door_offset(line, quarter_turns, flipped)
    return ox + dx, oy + dy


def entrance_path_cell(
    origin: Cell,
    line: HousingLine,
    quarter_turns: int = 0,
    flipped: bool = False,
) -> Cell:
    """Street tile in front of the door — mirrors BuildingDimensionLog.entrance_path_cell."""
    ox, oy = origin
    dx, dy = _DOOR_PATH_OFFSETS[(int(line), quarter_turns, flipped)]
    return ox + dx, oy + dy


def door_path_cell(
    origin: Cell,
    line: HousingLine,
    quarter_turns: int = 0,
    flipped: bool = False,
) -> Cell:
    """Alias kept for solver code; returns the entrance path tile."""
    return entrance_path_cell(origin, line, quarter_turns, flipped)


def _build_door_path_offsets() -> dict[tuple[int, int, bool], Cell]:
    out: dict[tuple[int, int, bool], Cell] = {}
    for line in HousingLine:
        for qt, flip in orientation_variants(line):
            door = entrance_door_offset(line, qt, flip)
            pdx, pdy = entrance_path_offset(line, qt, flip)
            out[(int(line), qt, flip)] = (door[0] + pdx, door[1] + pdy)
    return out


_DOOR_PATH_OFFSETS = _build_door_path_offsets()
