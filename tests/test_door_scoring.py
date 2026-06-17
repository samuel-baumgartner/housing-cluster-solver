from housing.footprints import door_path_cell, footprint_set
from housing.grid import WorldGrid
from housing.types import HousingLine


def _door_score_component(origin, line, qt, flipped, grid, houses, planned) -> int:
    fp = footprint_set(origin, line, qt, flipped)
    path_cell = door_path_cell(origin, line, qt, flipped)
    reserved_with_fp = set(grid.reserved)
    reserved_with_fp.update(fp)
    route_len = grid.connection_path_length(path_cell, reserved_with_fp, planned)
    door_inv = 1000 if route_len == 0 else 1000 // (1 + route_len)
    return door_inv


def test_door_score_uses_connection_route_not_manhattan() -> None:
    """Same route length → same door score even when Manhattan to network differs."""
    grid = WorldGrid.demo_deep_zone()
    planned: set = set()
    houses = []
    origin = (9, 16)
    qt, flip = 1, False

    small_pc = door_path_cell(origin, HousingLine.SMALL, qt, flip)
    l_pc = door_path_cell(origin, HousingLine.L, qt, flip)
    network = grid.path_network(planned)

    small_manhattan = min(
        abs(small_pc[0] - n[0]) + abs(small_pc[1] - n[1]) for n in network
    )
    l_manhattan = min(abs(l_pc[0] - n[0]) + abs(l_pc[1] - n[1]) for n in network)
    assert small_manhattan != l_manhattan

    small_fp = footprint_set(origin, HousingLine.SMALL, qt, flip)
    l_fp = footprint_set(origin, HousingLine.L, qt, flip)
    small_route = grid.connection_path_length(
        small_pc, grid.reserved | small_fp, planned
    )
    l_route = grid.connection_path_length(l_pc, grid.reserved | l_fp, planned)
    assert small_route == l_route

    small_door = _door_score_component(
        origin, HousingLine.SMALL, qt, flip, grid, houses, planned
    )
    l_door = _door_score_component(
        origin, HousingLine.L, qt, flip, grid, houses, planned
    )
    assert small_door == l_door

    # Old Manhattan metric would have favored small (shorter manhattan).
    old_small_inv = 1000 // (1 + small_manhattan)
    old_l_inv = 1000 // (1 + l_manhattan)
    assert old_small_inv > old_l_inv
