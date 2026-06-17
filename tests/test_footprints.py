"""Footprint shapes must match my-colony-sim building_footprints.csv + buildings.csv."""

from housing.footprints import (
    BBOX,
    BASE_FOOTPRINTS,
    entrance_door_cell,
    footprint_set,
)
from housing.types import HousingLine

# housing_l occupied cells from data/building_footprints.csv
_GAME_L_CELLS = {
    (x, y)
    for y in range(8)
    for x in (range(8) if y < 3 else range(3))
}


def test_bbox_matches_game_stage1() -> None:
    assert BBOX[HousingLine.SMALL] == (4, 4)
    assert BBOX[HousingLine.BIG] == (5, 4)
    assert BBOX[HousingLine.L] == (8, 8)


def test_cell_counts() -> None:
    assert len(BASE_FOOTPRINTS[HousingLine.SMALL]) == 16
    assert len(BASE_FOOTPRINTS[HousingLine.BIG]) == 20
    assert len(BASE_FOOTPRINTS[HousingLine.L]) == 39


def test_l_shape_matches_csv() -> None:
    assert set(BASE_FOOTPRINTS[HousingLine.L]) == _GAME_L_CELLS
    # Old wrong prototype kept x=3 on rows y>=3; game only uses x=0..2 there.
    assert (3, 3) not in BASE_FOOTPRINTS[HousingLine.L]
    assert (3, 7) not in BASE_FOOTPRINTS[HousingLine.L]
    assert (7, 0) in BASE_FOOTPRINTS[HousingLine.L]


def test_small_big_are_full_rectangles() -> None:
    w, h = BBOX[HousingLine.SMALL]
    assert set(BASE_FOOTPRINTS[HousingLine.SMALL]) == {
        (x, y) for y in range(h) for x in range(w)
    }
    w, h = BBOX[HousingLine.BIG]
    assert set(BASE_FOOTPRINTS[HousingLine.BIG]) == {
        (x, y) for y in range(h) for x in range(w)
    }


def test_entrance_doors_on_footprint() -> None:
    origin = (10, 10)
    for line in HousingLine:
        for qt in range(4):
            for flipped in (False, True):
                fp = footprint_set(origin, line, qt, flipped)
                door = entrance_door_cell(origin, line, qt, flipped)
                assert door in fp, f"{line} qt={qt} flip={flipped} door {door} not in footprint"
