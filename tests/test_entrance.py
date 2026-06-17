"""Entrance cells must match my-colony-sim BuildingDimensionLog (stage-1 housing)."""

from housing.footprints import entrance_door_cell, entrance_path_cell
from housing.types import HousingLine


def test_small_stage1_qt0() -> None:
    origin = (8, 14)
    assert entrance_door_cell(origin, HousingLine.SMALL, 0, False) == (9, 17)
    assert entrance_path_cell(origin, HousingLine.SMALL, 0, False) == (9, 18)


def test_small_stage1_qt1() -> None:
    origin = (8, 26)
    assert entrance_door_cell(origin, HousingLine.SMALL, 1, False) == (11, 28)
    assert entrance_path_cell(origin, HousingLine.SMALL, 1, False) == (12, 28)


def test_big_stage1_qt0() -> None:
    origin = (12, 14)
    assert entrance_door_cell(origin, HousingLine.BIG, 0, False) == (14, 17)
    assert entrance_path_cell(origin, HousingLine.BIG, 0, False) == (14, 18)


def test_l_stage1_qt0() -> None:
    origin = (8, 14)
    assert entrance_door_cell(origin, HousingLine.L, 0, False) == (9, 21)
    assert entrance_path_cell(origin, HousingLine.L, 0, False) == (9, 22)
