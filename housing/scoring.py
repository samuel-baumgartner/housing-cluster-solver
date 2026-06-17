from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .footprints import BBOX, door_path_cell, footprint_cells, footprint_set, orientation_variants
from .grid import WorldGrid, neighbors4
from .types import Candidate, Cell, House, HousingLine

STRANDED_POCKET_MIN = 16
BLOB_PENALTY_FREE_RATIO = 0.35
BLOB_PENALTY_MIN_HOUSES = 4
BLOB_ZONE_FRACTION = 0.20
BLOB_MIN_CELLS = 64
BLOB_PENALTY_PER_CELL = 2


def is_compact_wide_zone(zone_size: Tuple[int, int]) -> bool:
    """Wide shallow zones like 30×20 where edge packing dominates search."""
    zw, zh = zone_size
    return zw >= 28 and zh >= 18


def frontier_trim_limit(
    grid: WorldGrid,
    zone_size: Tuple[int, int],
    free_ratio: float,
    n_houses: int = 0,
) -> int:
    if is_compact_wide_zone(zone_size):
        return 200 if n_houses < 5 else 34
    if zone_size[1] >= 12 and free_ratio > 0.35:
        return 200
    return 64


def blob_cell_cap(zone_cell_count: int) -> int:
    return max(BLOB_MIN_CELLS, int(BLOB_ZONE_FRACTION * zone_cell_count))


def reserved_component_size(reserved: Set[Cell], seed: Cell) -> int:
    """Orthogonally connected reserved/building cells containing seed."""
    if seed not in reserved:
        return 0
    q = [seed]
    seen = {seed}
    head = 0
    while head < len(q):
        c = q[head]
        head += 1
        for n in neighbors4(c):
            if n in seen or n not in reserved:
                continue
            seen.add(n)
            q.append(n)
    return len(seen)


def blob_size_penalty(
    footprint: Set[Cell],
    reserved: Set[Cell],
    houses: List[House],
    zone_cell_count: int,
    free_ratio: float,
) -> int:
    """Penalize growing one mega-cluster while the district still has open space."""
    if len(houses) < BLOB_PENALTY_MIN_HOUSES or free_ratio <= BLOB_PENALTY_FREE_RATIO:
        return 0
    if not footprint:
        return 0
    reserved_with_fp = set(reserved)
    reserved_with_fp.update(footprint)
    blob = reserved_component_size(reserved_with_fp, next(iter(footprint)))
    cap = blob_cell_cap(zone_cell_count)
    if blob <= cap:
        return 0
    return (blob - cap) * BLOB_PENALTY_PER_CELL


def line_order_for_span(span_w: int, span_h: int) -> List[HousingLine]:
    fitting: List[HousingLine] = []
    for line in HousingLine:
        bw, bh = BBOX[line]
        if span_w >= bw and span_h >= bh:
            fitting.append(line)
    return fitting or list(HousingLine)


def line_order_smallest_first(span_w: int, span_h: int) -> List[HousingLine]:
    lines = line_order_for_span(span_w, span_h)
    return sorted(lines, key=lambda ln: BBOX[ln][0] * BBOX[ln][1])


def line_mix_bonus(line: HousingLine, houses: List[House], zone_size: Tuple[int, int]) -> int:
    zw, zh = zone_size
    if not houses:
        if zh >= 12:
            return 60 if line == HousingLine.SMALL else 0
        if line == HousingLine.L and zw >= 8 and zh >= 8:
            return 80
        if line == HousingLine.BIG and zw >= 5 and zh >= 4:
            return 40
        return 0
    counts = {ln: 0 for ln in HousingLine}
    for h in houses:
        counts[h.line] += 1
    if counts[line] == 0:
        if line == HousingLine.L:
            return 90
        if line == HousingLine.BIG:
            return 45
        return 25
    min_used = min(counts.values())
    if counts[line] <= min_used:
        bonus = 28
        if line == HousingLine.L:
            bonus += 55
        elif line == HousingLine.BIG:
            bonus += 22
        return bonus
    return 0


def score_candidate(
    origin: Cell,
    line: HousingLine,
    qt: int,
    flipped: bool,
    grid: WorldGrid,
    houses: List[House],
    th_paths: Set[Cell],
    planned: Set[Cell],
    *,
    network: Optional[Set[Cell]] = None,
    route_len: Optional[int] = None,
    fp: Optional[FrozenSet[Cell]] = None,
    house_fps: Optional[Tuple[FrozenSet[Cell], ...]] = None,
    free_ratio: Optional[float] = None,
    south_edge: Optional[int] = None,
) -> int:
    if fp is None:
        fp = footprint_set(origin, line, qt, flipped)
    if not fp:
        return -1_000_000
    if network is None:
        network = grid.path_network(planned)
    touch = 0
    exposed = 0
    for c in fp:
        for n in neighbors4(c):
            if n in grid.reserved:
                touch += 1
            elif n not in fp:
                if (
                    n in grid.zone
                    and n not in grid.reserved
                    and n not in network
                    and n not in grid.paths
                    and n not in planned
                ):
                    exposed += 1
    path_cell = door_path_cell(origin, line, qt, flipped)
    if route_len is None:
        reserved_with_fp = set(grid.reserved)
        reserved_with_fp.update(fp)
        route_len = grid.connection_path_length(path_cell, reserved_with_fp, planned)
    door_inv = 1000 if route_len == 0 else 1000 // (1 + route_len)
    near_houses = 0
    near: Set[Cell] = set()
    for c in fp:
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if max(abs(dx), abs(dy)) > 2:
                    continue
                near.add((c[0] + dx, c[1] + dy))
    if house_fps is None:
        house_fps = tuple(
            footprint_set(h.origin, h.line, h.quarter_turns, h.flipped) for h in houses
        )
    for hcells in house_fps:
        if hcells & near:
            near_houses += 1
    depth_bonus = 0
    if len(houses) >= 3 and south_edge is not None:
        _, zone_size = grid.bbox_of(grid.zone)
        if zone_size[1] >= 10:
            depth_bonus = max(0, south_edge - origin[1]) * 4
    if free_ratio is None:
        free_ratio = len(grid.unreserved_zone()) / max(1, len(grid.zone))
    blob_penalty = blob_size_penalty(
        fp, grid.reserved, houses, len(grid.zone), free_ratio
    )
    return (
        4 * touch
        + door_inv
        + near_houses
        + depth_bonus
        - 3 * exposed
        - blob_penalty
    )


def solver_placement_score(
    origin: Cell,
    line: HousingLine,
    qt: int,
    flipped: bool,
    grid: WorldGrid,
    houses: List[House],
    planned: Set[Cell],
) -> Tuple[int, int, int, int]:
    """Score a placement the same way find_best_candidate does."""
    from .fast_grid import SearchPass
    from .footprints import footprint_set

    free = grid.unreserved_zone()
    free_ratio = len(free) / max(1, len(grid.zone))
    zpos, zone_size = grid.bbox_of(grid.zone)
    south_edge = zpos[1] + zone_size[1] - 1 if zone_size[1] >= 10 else None
    house_fps = tuple(
        footprint_set(h.origin, h.line, h.quarter_turns, h.flipped) for h in houses
    )
    search = SearchPass(
        grid, planned, houses, house_fps, free_ratio, south_edge, zone_size
    )
    _, _, route_len, ok = search.evaluate(origin, int(line), qt, flipped)
    fp = footprint_set(origin, line, qt, flipped)
    if not ok or not fp:
        base = -1_000_000
    else:
        base = search.score(origin, route_len, fp, len(houses))
    mix = line_mix_bonus(line, houses, zone_size)
    return base, mix, base + mix, route_len


def collect_frontier(
    grid: WorldGrid,
    houses: List[House],
    planned: Set[Cell],
    th_paths: Set[Cell],
) -> Set[Cell]:
    network = grid.path_network(planned)
    _, zone_size = grid.bbox_of(grid.zone)
    free_count = len(grid.unreserved_zone())
    free_ratio = free_count / max(1, len(grid.zone))
    seeds: Set[Cell] = set()
    if len(houses) >= 2 and zone_size[1] >= 12 and free_ratio > 0.35:
        seeds = grid.unreserved_zone().copy()
    elif not houses:
        for c in grid.zone:
            if c in grid.reserved:
                continue
            if any(n in network for n in neighbors4(c)):
                seeds.add(c)
    else:
        for c in grid.zone:
            if c in grid.reserved:
                continue
            if any(
                (n in grid.reserved or n in network) for n in neighbors4(c)
            ):
                seeds.add(c)
        expanded = set(seeds)
        q = list(seeds)
        head = 0
        while head < len(q):
            c = q[head]
            head += 1
            for n in neighbors4(c):
                if n in expanded or n not in grid.zone or n in grid.reserved:
                    continue
                expanded.add(n)
                q.append(n)
        seeds = expanded
    seeds |= reserved_ne_corner_origins(grid, houses, planned)
    return seeds


def origin_can_place(
    grid: WorldGrid,
    origin: Cell,
    planned: Set[Cell],
) -> bool:
    """True if any housing orientation fits at origin without overlapping blocked cells."""
    for line in HousingLine:
        for qt, flip in orientation_variants(line):
            fp = footprint_set(origin, line, qt, flip)
            if fp and grid.can_place_footprint(fp, planned):
                return True
    return False


def reserved_ne_corner_origins(
    grid: WorldGrid,
    houses: List[House],
    planned: Set[Cell],
) -> Set[Cell]:
    """Reserved NE building corners that can still anchor a valid footprint."""
    extras: Set[Cell] = set()
    for c in grid.zone:
        if c not in grid.reserved:
            continue
        owner_fp: Optional[FrozenSet[Cell]] = None
        for h in houses:
            hfp = footprint_set(h.origin, h.line, h.quarter_turns, h.flipped)
            if c in hfp:
                owner_fp = hfp
                break
        if owner_fp is None:
            continue
        x, y = c
        if (x, y - 1) in owner_fp or (x + 1, y) in owner_fp:
            continue
        if origin_can_place(grid, c, planned):
            extras.add(c)
    return extras


def trim_frontier(
    frontier: Set[Cell],
    grid: WorldGrid,
    planned: Set[Cell],
    limit: int,
    *,
    network: Optional[Set[Cell]] = None,
    placeable_cache: Optional[Dict[Cell, bool]] = None,
    zone_size: Optional[Tuple[int, int]] = None,
    n_houses: int = 0,
    houses: Optional[List[House]] = None,
) -> Set[Cell]:
    if len(frontier) <= limit:
        return frontier
    if network is None:
        network = grid.path_network(planned)
    anchors = set(grid.reserved)
    anchors.update(network)
    anchor_arr = np.empty((len(anchors), 2), dtype=np.int32)
    for i, (ax, ay) in enumerate(anchors):
        anchor_arr[i, 0] = ax
        anchor_arr[i, 1] = ay
    n_anchors = anchor_arr.shape[0]
    from .fast_bfs import min_manhattan_to_anchors

    scored = []
    for origin in frontier:
        best_d = min_manhattan_to_anchors(origin[0], origin[1], anchor_arr, n_anchors)
        scored.append((best_d, origin))
    scored.sort(key=lambda t: t[0])
    nearest = {o for _, o in scored[:limit]}
    if zone_size is None:
        _, zone_size = grid.bbox_of(grid.zone)
    if is_compact_wide_zone(zone_size) and houses is not None:
        return nearest | reserved_ne_corner_origins(grid, houses, planned)
    if placeable_cache is None:
        placeable = {o for o in frontier if origin_can_place(grid, o, planned)}
    else:
        placeable = set()
        for o in frontier:
            if o not in placeable_cache:
                placeable_cache[o] = origin_can_place(grid, o, planned)
            if placeable_cache[o]:
                placeable.add(o)
    return nearest | placeable


def connected_components(free: Set[Cell]) -> List[Set[Cell]]:
    remaining = set(free)
    comps: List[Set[Cell]] = []
    while remaining:
        start = next(iter(remaining))
        comp = {start}
        q = [start]
        remaining.remove(start)
        head = 0
        while head < len(q):
            c = q[head]
            head += 1
            for n in neighbors4(c):
                if n not in remaining:
                    continue
                remaining.remove(n)
                comp.add(n)
                q.append(n)
        comps.append(comp)
    return comps


def developable_from_parent(
    grid: WorldGrid,
    reserved: Set[Cell],
    parent_seeds: Set[Cell],
    planned: Set[Cell],
) -> Set[Cell]:
    walkable = set(grid.zone)
    walkable.update(grid.paths)
    walkable.update(planned)
    for c in reserved:
        walkable.discard(c)
    developable: Set[Cell] = set()
    for seed in parent_seeds:
        if seed not in walkable:
            continue
        q = [seed]
        seen = {seed}
        head = 0
        while head < len(q):
            c = q[head]
            head += 1
            if c in grid.zone and c not in reserved:
                developable.add(c)
            for n in neighbors4(c):
                if n in seen or n not in walkable:
                    continue
                seen.add(n)
                q.append(n)
    return developable


def build_route_parent_seeds(
    grid: WorldGrid,
    reserved: Set[Cell],
    planned: Set[Cell],
    th_paths: Set[Cell],
) -> Set[Cell]:
    seeds = set(th_paths)
    seeds.update(planned)
    for c in grid.zone:
        if c in grid.paths and grid._path_connected_to_th(c):
            seeds.add(c)
    walkable = set(grid.zone)
    walkable.update(grid.paths)
    walkable.update(planned)
    for c in reserved:
        walkable.discard(c)
    reachable = set(seeds)
    q = list(seeds)
    head = 0
    while head < len(q):
        c = q[head]
        head += 1
        for n in neighbors4(c):
            if n in reachable or n not in walkable:
                continue
            reachable.add(n)
            q.append(n)
    return reachable


def strands_large_pocket(
    grid: WorldGrid,
    reserved: Set[Cell],
    planned: Set[Cell],
    reach_before: Set[Cell],
    reach_after: Set[Cell],
) -> bool:
    free = {c for c in grid.zone if c not in reserved}
    for comp in connected_components(free):
        if len(comp) < STRANDED_POCKET_MIN:
            continue
        had = any(c in reach_before for c in comp)
        if not had:
            continue
        still = any(c in reach_after for c in comp)
        if not still:
            return True
    return False


def would_commit(
    origin: Cell,
    line: HousingLine,
    qt: int,
    flipped: bool,
    grid: WorldGrid,
    reserved: Set[Cell],
    planned: Set[Cell],
    th_paths: Set[Cell],
    check_strand: bool,
    *,
    sim_planned: Optional[Set[Cell]] = None,
    reach_before: Optional[np.ndarray] = None,
) -> Tuple[bool, Set[Cell], Set[Cell]]:
    fp = footprint_set(origin, line, qt, flipped)
    if not grid.can_place_footprint(fp, planned):
        return False, planned, set()
    reserved_copy = set(reserved)
    reserved_copy.update(fp)
    path_cell = door_path_cell(origin, line, qt, flipped)
    if sim_planned is None:
        ok, sim_planned = grid.probe_path_connection(path_cell, reserved_copy, planned)
        if not ok:
            return False, planned, set()
    elif path_cell not in grid.path_network(sim_planned):
        ok, sim_planned = grid.probe_path_connection(path_cell, reserved_copy, sim_planned)
        if not ok:
            return False, planned, set()
    if check_strand:
        accel = grid._ensure_accel()
        if reach_before is None:
            reach_before = accel.build_reach_zone_free(planned)
        reach_after = accel.reach_with_extra_reserved(planned, fp)
        if accel.strands_large_pocket(
            reserved_copy, reach_before, reach_after, STRANDED_POCKET_MIN
        ):
            return False, planned, set()
    return True, sim_planned, fp
