from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .grid import WorldGrid, neighbors4
from .types import Cell, District

LAYOUT_MIN_ZONE = 800
DISTRICT_TARGET_CELLS = 300
LLOYD_ITERATIONS = 3
MIN_INTERIOR_CELLS = 200


def layout_enabled(zone_cell_count: int) -> bool:
    return zone_cell_count >= LAYOUT_MIN_ZONE


def _sq_dist(a: Cell, b: Cell) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _farthest_point_seeds(zone: Set[Cell], k: int) -> List[Cell]:
    cells = sorted(zone)
    if not cells:
        return []
    k = min(k, len(cells))
    cx = sum(c[0] for c in cells) / len(cells)
    cy = sum(c[1] for c in cells) / len(cells)
    seeds = [min(cells, key=lambda c: _sq_dist(c, (round(cx), round(cy))))]
    while len(seeds) < k:
        best_cell = cells[0]
        best_min = -1
        for c in cells:
            d = min(_sq_dist(c, s) for s in seeds)
            if d > best_min:
                best_min = d
                best_cell = c
        seeds.append(best_cell)
    return seeds


def _assign_voronoi(zone: Set[Cell], seeds: List[Cell]) -> Dict[int, Set[Cell]]:
    districts: Dict[int, Set[Cell]] = {i: set() for i in range(len(seeds))}
    for c in zone:
        best_i = min(range(len(seeds)), key=lambda i: _sq_dist(c, seeds[i]))
        districts[best_i].add(c)
    return districts


def _nearest_zone_cell(target: Tuple[float, float], cells: Set[Cell]) -> Cell:
    tx, ty = target
    return min(cells, key=lambda c: (c[0] - tx) ** 2 + (c[1] - ty) ** 2)


def _lloyd_relax(zone: Set[Cell], seeds: List[Cell]) -> List[Cell]:
    districts = _assign_voronoi(zone, seeds)
    new_seeds: List[Cell] = []
    for i in range(len(seeds)):
        cells = districts.get(i, set())
        if not cells:
            new_seeds.append(seeds[i])
            continue
        cx = sum(c[0] for c in cells) / len(cells)
        cy = sum(c[1] for c in cells) / len(cells)
        new_seeds.append(_nearest_zone_cell((cx, cy), cells))
    return new_seeds


def _district_owners(partition: Dict[int, Set[Cell]]) -> Dict[Cell, int]:
    owners: Dict[Cell, int] = {}
    for did, cells in partition.items():
        for c in cells:
            owners[c] = did
    return owners


def _adjacent_to_existing_path(cell: Cell, grid: WorldGrid) -> bool:
    return any(n in grid.paths for n in neighbors4(cell))


def _single_width_streets(
    zone: Set[Cell], partition: Dict[int, Set[Cell]], grid: WorldGrid
) -> Set[Cell]:
    """One-cell-wide streets on zone edges and district borders."""
    owners = _district_owners(partition)
    streets: Set[Cell] = set()
    for c in zone:
        if any(n not in zone for n in neighbors4(c)):
            if _adjacent_to_existing_path(c, grid):
                continue
            streets.add(c)
    adj = _district_neighbors(partition)
    for did_a, neighbors in adj.items():
        for did_b in neighbors:
            if did_a >= did_b:
                continue
            low, high = did_a, did_b
            for c in partition[low]:
                for n in neighbors4(c):
                    if owners.get(n) != high:
                        continue
                    if any(nn not in zone for nn in neighbors4(n)):
                        continue
                    streets.add(c)
    return streets


def _district_neighbors(
    districts: Dict[int, Set[Cell]],
) -> Dict[int, Set[int]]:
    owners: Dict[Cell, int] = {}
    for did, cells in districts.items():
        for c in cells:
            owners[c] = did
    adj: Dict[int, Set[int]] = {i: set() for i in districts}
    for did, cells in districts.items():
        for c in cells:
            for n in neighbors4(c):
                other = owners.get(n)
                if other is not None and other != did:
                    adj[did].add(other)
    return adj


def _merge_districts(
    districts: Dict[int, Set[Cell]], a: int, b: int
) -> Dict[int, Set[Cell]]:
    merged = {i: set(cells) for i, cells in districts.items()}
    merged[a] |= merged.pop(b)
    reindexed: Dict[int, Set[Cell]] = {}
    for new_id, (_, cells) in enumerate(sorted(merged.items())):
        reindexed[new_id] = cells
    return reindexed


def _build_district_objects(
    districts: Dict[int, Set[Cell]],
    streets: Set[Cell],
) -> List[District]:
    result: List[District] = []
    for did in sorted(districts):
        cells = districts[did]
        if not cells:
            continue
        ring = cells & streets
        interior = cells - ring
        result.append(District(id=did, cells=cells, interior=interior, ring=ring))
    return result


def _close_ring_connectors(
    grid: WorldGrid, streets: Set[Cell], zone: Set[Cell]
) -> Set[Cell]:
    """Extend dead-end ring cells one step into zone cells that meet existing paths."""
    result = set(streets)
    changed = True
    while changed:
        changed = False
        street_graph = result | grid.paths
        for c in list(result):
            if len([n for n in neighbors4(c) if n in street_graph]) != 1:
                continue
            for n in neighbors4(c):
                if n in result or n in grid.paths or n not in zone:
                    continue
                if any(p in grid.paths for p in neighbors4(n)):
                    result.add(n)
                    changed = True
                    break
    return result


def _barrier_for_district(
    district: District, streets: Set[Cell], grid: WorldGrid
) -> Set[Cell]:
    barrier: Set[Cell] = set()
    for c in district.cells:
        for n in neighbors4(c):
            if n in streets or n in grid.paths:
                barrier.add(n)
    return barrier


def district_loop_closed(
    district: District, streets: Set[Cell], grid: WorldGrid, zone: Set[Cell]
) -> bool:
    """True when every interior cell is separated from the outside by streets or paths."""
    if len(district.interior) < MIN_INTERIOR_CELLS:
        return False
    barrier = _barrier_for_district(district, streets, grid)
    for c in district.interior:
        for n in neighbors4(c):
            if n in district.interior:
                continue
            if n in barrier:
                continue
            return False
    return True


def ring_has_no_path_gaps(
    streets: Set[Cell], grid: WorldGrid, zone: Set[Cell]
) -> bool:
    """No planned dead-end should still be one step from closing onto a path."""
    street_graph = streets | grid.paths
    for c in streets:
        if len([n for n in neighbors4(c) if n in street_graph]) != 1:
            continue
        for n in neighbors4(c):
            if n in zone and n not in streets and n not in grid.paths:
                if any(p in grid.paths for p in neighbors4(n)):
                    return False
    return True


def validate_layout_loops(
    districts: List[District],
    streets: Set[Cell],
    grid: WorldGrid,
    zone: Set[Cell],
) -> None:
    if not ring_has_no_path_gaps(streets, grid, zone):
        raise ValueError("layout ring has dead-ends that should connect to paths")
    for d in districts:
        if not district_loop_closed(d, streets, grid, zone):
            raise ValueError(f"district {d.id} ring is not closed")


def _manhattan_to_set(cell: Cell, targets: Set[Cell]) -> int:
    return min(abs(cell[0] - t[0]) + abs(cell[1] - t[1]) for t in targets)


def _append_route_gap(
    grid: WorldGrid, streets: Set[Cell], route: List[Cell]
) -> None:
    """Add only route cells not already on the path network."""
    network = grid.path_network(streets)
    for c in route:
        if c in network or c in grid.paths:
            continue
        streets.add(c)
        network.add(c)


def _connect_district_rings(
    grid: WorldGrid,
    districts: List[District],
    planned: Set[Cell],
) -> Set[Cell]:
    th = set(grid.th_paths)
    streets = set(planned)
    for d in districts:
        ring_candidates = sorted(
            d.ring,
            key=lambda c: _manhattan_to_set(c, th),
        )
        connected = False
        for anchor in ring_candidates[:8]:
            if anchor in grid.path_network(streets):
                connected = True
                break
            route = grid.route_to_network(
                anchor, grid.reserved, streets, allow_loops=False
            )
            if route:
                _append_route_gap(grid, streets, route)
                connected = True
                break
        if not connected:
            for c in d.ring:
                if c not in grid.paths and not _adjacent_to_existing_path(c, grid):
                    streets.add(c)
    return streets


def _interior_reachable(
    grid: WorldGrid, interior: Set[Cell], planned: Set[Cell]
) -> bool:
    if not interior:
        return False
    accel = grid._ensure_accel()
    reach = accel.build_reach_zone_free(planned)
    return any(reach[c[1], c[0]] for c in interior)


def _planned_from_partition(
    grid: WorldGrid,
    partition: Dict[int, Set[Cell]],
) -> Tuple[List[District], Set[Cell]]:
    zone = set(grid.zone)
    streets = _single_width_streets(zone, partition, grid)
    streets = _close_ring_connectors(grid, streets, zone)
    districts = _build_district_objects(partition, streets)
    planned = {c for c in streets if c not in grid.paths}
    planned = _connect_district_rings(grid, districts, planned)
    streets = set(planned) | {c for c in streets if c in grid.paths}
    streets = _close_ring_connectors(grid, streets, zone)
    planned = {c for c in streets if c not in grid.paths}
    districts = _build_district_objects(partition, streets)
    validate_layout_loops(districts, streets, grid, zone)
    return districts, planned


def _validate_and_repair(
    grid: WorldGrid,
    partition: Dict[int, Set[Cell]],
) -> Dict[int, Set[Cell]]:
    current = partition
    guard = 0
    while guard < len(partition) + 4:
        guard += 1
        districts, planned = _planned_from_partition(grid, current)
        bad: Optional[District] = None
        for d in districts:
            if len(d.interior) < MIN_INTERIOR_CELLS or not _interior_reachable(
                grid, d.interior, planned
            ):
                bad = d
                break
        if bad is None:
            return current
        adj = _district_neighbors(current)
        neighbors = adj.get(bad.id, set())
        if not neighbors:
            return current
        merge_into = min(
            neighbors,
            key=lambda n: len(current.get(n, set())),
        )
        current = _merge_districts(current, merge_into, bad.id)
    return current


def plan_district_layout(
    grid: WorldGrid,
) -> Optional[Tuple[List[District], Set[Cell]]]:
    """Partition a large zone and lay street loops. Returns None for small zones."""
    zone = set(grid.zone)
    if not layout_enabled(len(zone)):
        return None

    k = max(2, round(len(zone) / DISTRICT_TARGET_CELLS))
    seeds = _farthest_point_seeds(zone, k)
    for _ in range(LLOYD_ITERATIONS):
        seeds = _lloyd_relax(zone, seeds)
    partition = _assign_voronoi(zone, seeds)
    partition = {i: cells for i, cells in partition.items() if cells}

    partition = _validate_and_repair(grid, partition)
    districts, planned = _planned_from_partition(grid, partition)

    return districts, planned
