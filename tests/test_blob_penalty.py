from housing.footprints import footprint_set
from housing.grid import WorldGrid
from housing.layout import LAYOUT_MIN_ZONE
from housing.scoring import (
    blob_cell_cap,
    blob_size_penalty,
    reserved_component_size,
)
from housing.types import HousingLine


def test_reserved_component_size_counts_orthogonally_connected() -> None:
    reserved = {(0, 0), (1, 0), (0, 1), (5, 5)}
    assert reserved_component_size(reserved, (0, 0)) == 3
    assert reserved_component_size(reserved, (5, 5)) == 1


def test_blob_penalty_off_when_zone_mostly_full() -> None:
    fp = footprint_set((10, 10), HousingLine.SMALL, 0, False)
    reserved = set(fp)
    assert blob_size_penalty(fp, reserved, [None] * 6, 100, 0.30) == 0  # type: ignore[list-item]


def test_blob_penalty_on_oversized_cluster_with_space_left() -> None:
    reserved = {(x, 10) for x in range(70)}
    fp = {(70, 10)}
    penalty = blob_size_penalty(fp, reserved, [None] * 6, 200, 0.50)  # type: ignore[list-item]
    cap = blob_cell_cap(200)
    assert cap == 64
    assert penalty == (71 - cap) * 2


def test_blob_penalty_skipped_on_layout_zones() -> None:
    reserved = {(x, 10) for x in range(70)}
    fp = {(70, 10)}
    assert (
        blob_size_penalty(
            fp,
            reserved,
            [None] * 6,  # type: ignore[list-item]
            LAYOUT_MIN_ZONE,
            0.50,
        )
        == 0
    )
