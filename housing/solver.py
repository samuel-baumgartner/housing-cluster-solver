from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .footprints import BBOX, door_path_cell, footprint_set, orientation_variants
from .fast_grid import SearchPass
from .grid import WorldGrid, neighbors4
from .scoring import (
    collect_frontier,
    frontier_trim_limit,
    is_compact_wide_zone,
    line_mix_bonus,
    line_order_for_span,
    line_order_smallest_first,
    trim_frontier,
    would_commit,
)
from .types import Candidate, Cell, House, HousingLine, SolveResult, SolveStep


def find_best_candidate(
    grid: WorldGrid,
    houses: List[House],
    planned: Set[Cell],
    th_paths: Set[Cell],
    skip_keys: Set[str],
    require_strand: bool,
) -> Tuple[Optional[Candidate], Set[Cell], List[Candidate]]:
    free = grid.unreserved_zone()
    if not free:
        return None, set(), []
    _, free_size = grid.bbox_of(free)
    if free_size == (0, 0):
        return None, set(), []
    network = grid.path_network(planned)
    frontier = collect_frontier(grid, houses, planned, th_paths)
    _, zone_size = grid.bbox_of(grid.zone)
    free_ratio = len(free) / max(1, len(grid.zone))
    trim_limit = frontier_trim_limit(grid, zone_size, free_ratio, len(houses))
    placeable_cache: Dict[Cell, bool] = {}
    frontier = trim_frontier(
        frontier,
        grid,
        planned,
        trim_limit,
        network=network,
        placeable_cache=placeable_cache,
        zone_size=zone_size,
        n_houses=len(houses),
        houses=houses,
    )
    if not frontier:
        return None, set(), []
    line_order = line_order_for_span(free_size[0], free_size[1])
    if zone_size[1] >= 14 and free_ratio > 0.5 and len(houses) >= 4:
        line_order = line_order_smallest_first(free_size[0], free_size[1])

    accel = grid._ensure_accel()
    reach_before = accel.build_reach_zone_free(planned)
    house_fps = tuple(
        footprint_set(h.origin, h.line, h.quarter_turns, h.flipped) for h in houses
    )
    zpos, zone_size = grid.bbox_of(grid.zone)
    south_edge = zpos[1] + zone_size[1] - 1 if zone_size[1] >= 10 else None
    search = SearchPass(
        grid, planned, houses, house_fps, free_ratio, south_edge, zone_size
    )

    mix_bonus = {line: line_mix_bonus(line, houses, zone_size) for line in HousingLine}
    use_batch = is_compact_wide_zone(zone_size)
    scored: List[Candidate] = []
    for origin in sorted(frontier):
        for line in line_order:
            orients = orientation_variants(line)
            if use_batch:
                orient_data = search.batch_line_orients(origin, int(line), orients)
            else:
                orient_data = []
                for qt, flip in orients:
                    _, _, route_len, ok = search.evaluate(
                        origin, int(line), qt, flip
                    )
                    orient_data.append((route_len, qt, flip, None, route_len, ok))
            orient_data.sort(key=lambda t: t[0])
            orient_limit = (
                len(orient_data)
                if require_strand or len(houses) < 3
                else min(2 if is_compact_wide_zone(zone_size) and len(houses) >= 5 else 3, len(orient_data))
            )
            for route_len, qt, flipped, fp, _, ok in orient_data[:orient_limit]:
                if not ok:
                    continue
                cand = Candidate(origin, line, qt, flipped, 0)
                if cand.key() in skip_keys:
                    continue
                if not fp:
                    fp = footprint_set(origin, line, qt, flipped)
                score = search.score(origin, route_len, fp, len(houses))
                score += mix_bonus[line]
                scored.append(Candidate(origin, line, qt, flipped, score))
    if not scored:
        return None, frontier, []
    scored.sort(key=lambda c: (-c.score, c.line.value))
    limits = [len(scored)] if require_strand or not houses else [48, len(scored)]
    for limit in limits:
        for entry in scored[:limit]:
            fp = footprint_set(
                entry.origin, entry.line, entry.quarter_turns, entry.flipped
            )
            path_cell = door_path_cell(
                entry.origin, entry.line, entry.quarter_turns, entry.flipped
            )
            ok, sim_planned, _ = search.full_connection(path_cell, fp)
            if not ok:
                continue
            ok_commit, _, _ = would_commit(
                entry.origin,
                entry.line,
                entry.quarter_turns,
                entry.flipped,
                grid,
                grid.reserved,
                planned,
                th_paths,
                require_strand,
                sim_planned=sim_planned,
                reach_before=search.reach_before,
            )
            if ok_commit:
                return entry, frontier, scored[:8]
        if limit >= len(scored):
            break
    return None, frontier, scored[:8]


def try_open_street_at_row(
    grid: WorldGrid,
    street_y: int,
    planned: Dict[Cell, bool],
) -> bool:
    street_cells = {
        c
        for c in grid.zone
        if c[1] == street_y and c not in grid.reserved
    }
    if not street_cells:
        return False
    for c in street_cells:
        planned[c] = True
    for anchor in street_cells:
        route = grid.route_to_network(
            anchor, grid.reserved, set(planned.keys()), allow_loops=False
        )
        if route:
            for c in route:
                if c not in grid.paths:
                    planned[c] = True
            return True
    for c in street_cells:
        planned.pop(c, None)
    return False


def try_open_interior_street(
    grid: WorldGrid,
    planned: Dict[Cell, bool],
) -> bool:
    row_set = set()
    for c in grid.zone:
        if c in grid.reserved:
            continue
        south = (c[0], c[1] + 1)
        if south in grid.reserved:
            row_set.add(c[1])
    if not row_set:
        return _try_open_boundary_spine(grid, planned)
    for street_y in sorted(row_set, reverse=True):
        if try_open_street_at_row(grid, street_y, planned):
            return True
    return _try_open_boundary_spine(grid, planned)


def _try_open_boundary_spine(grid: WorldGrid, planned: Dict[Cell, bool]) -> bool:
    pos, size = grid.bbox_of(grid.zone)
    if size == (0, 0):
        return False
    for spine_x in (pos[0], pos[0] + size[0] - 1):
        spine_cells = {
            (spine_x, y)
            for y in range(pos[1], pos[1] + size[1])
            if (spine_x, y) in grid.zone and (spine_x, y) not in grid.reserved
        }
        if not spine_cells:
            continue
        for c in spine_cells:
            planned[c] = True
        for anchor in spine_cells:
            route = grid.route_to_network(
            anchor, grid.reserved, set(planned.keys()), allow_loops=False
        )
            if route:
                for c in route:
                    if c not in grid.paths:
                        planned[c] = True
                return True
        for c in spine_cells:
            planned.pop(c, None)
    return False


def commit_house(
    grid: WorldGrid,
    cand: Candidate,
    houses: List[House],
    planned: Set[Cell],
    th_paths: Set[Cell],
) -> Tuple[bool, List[Cell]]:
    ok, sim_planned, fp = would_commit(
        cand.origin,
        cand.line,
        cand.quarter_turns,
        cand.flipped,
        grid,
        grid.reserved,
        planned,
        th_paths,
        True,
    )
    if not ok:
        return False, []
    path_cell = door_path_cell(
        cand.origin, cand.line, cand.quarter_turns, cand.flipped
    )
    route = grid.route_to_network(
        path_cell, grid.reserved | fp, sim_planned, allow_loops=False
    )
    planned.update(sim_planned)
    grid.bump_planned_version()
    grid._ensure_accel().sync_planned(planned)
    for c in fp:
        if c in planned:
            planned.discard(c)
    grid.reserved.update(fp)
    grid._bump_reserved_version()
    grid._ensure_accel()._dist_net_key = None
    grid._ensure_accel()._reach_key = None
    houses.append(
        House(
            origin=cand.origin,
            line=cand.line,
            quarter_turns=cand.quarter_turns,
            flipped=cand.flipped,
            path_cell=path_cell,
        )
    )
    return True, route


def solve(grid: WorldGrid, record_steps: bool = True, *, animation: bool = False) -> SolveResult:
    houses: List[House] = []
    planned: Set[Cell] = set()
    steps: List[SolveStep] = []
    th_paths = set(grid.th_paths)
    skip_keys: Set[str] = set()
    street_opens = 0
    frontier_snap_limit = 64 if animation else 0

    def _snap_frontier(frontier: Optional[Set[Cell]]) -> Set[Cell]:
        if not frontier:
            return set()
        if not frontier_snap_limit or len(frontier) <= frontier_snap_limit:
            return set(frontier)
        items = sorted(frontier)
        stride = len(items) / frontier_snap_limit
        return {items[int(i * stride)] for i in range(frontier_snap_limit)}

    def snap(
        kind: str,
        title: str,
        detail: str = "",
        frontier: Optional[Set[Cell]] = None,
        top: Optional[List[Candidate]] = None,
        selected: Optional[Candidate] = None,
        highlight_fp: Optional[Set[Cell]] = None,
        route: Optional[List[Cell]] = None,
    ) -> None:
        if not record_steps:
            return
        steps.append(
            SolveStep(
                kind=kind,
                title=title,
                detail=detail,
                houses=list(houses),
                planned_paths=set(planned),
                reserved=set(grid.reserved),
                frontier=frontier or set(),
                highlight_footprint=highlight_fp or set(),
                highlight_path_route=route or [],
                top_candidates=top or [],
                selected=selected,
            )
        )

    snap("init", "Green zone ready", f"{len(grid.zone)} buildable cells")
    guard = 0
    while guard < 256:
        guard += 1
        placed = False
        while True:
            best, frontier, top = find_best_candidate(
                grid, houses, planned, th_paths, skip_keys, True
            )
            if best is None:
                break
            snap(
                "evaluate",
                f"Best candidate score {best.score}",
                f"{best.label} at {best.origin} qt={best.quarter_turns}",
                frontier=_snap_frontier(frontier),
                top=top,
                selected=best,
                highlight_fp=footprint_set(
                    best.origin, best.line, best.quarter_turns, best.flipped
                ),
            )
            committed, route = commit_house(grid, best, houses, planned, th_paths)
            if committed:
                skip_keys.clear()
                placed = True
                snap(
                    "place",
                    f"Placed {LINE_NAME(best.line)} #{len(houses)}",
                    f"origin={best.origin} score={best.score}",
                    highlight_fp=footprint_set(
                        best.origin, best.line, best.quarter_turns, best.flipped
                    ),
                    route=route,
                )
                break
            skip_keys.add(best.key())
            snap(
                "reject",
                "Commit failed (strand/path)",
                best.key(),
                selected=best,
            )
        if placed:
            continue
        before = set(planned)
        planned_dict: Dict[Cell, bool] = {c: True for c in planned}
        if try_open_interior_street(grid, planned_dict):
            planned = set(planned_dict.keys())
            grid.bump_planned_version()
            grid._ensure_accel().sync_planned(planned)
            new_cells = planned - before
            street_opens += 1
            snap(
                "street",
                "Opened interior street",
                f"+{len(new_cells)} path cells",
                highlight_fp=new_cells,
            )
            if street_opens > 48:
                break
            skip_keys.clear()
            continue
        snap("done", "Saturated", f"{len(houses)} homes, {len(planned)} path tiles")
        break
    return SolveResult(houses=houses, planned_paths=planned, steps=steps)


def LINE_NAME(line: HousingLine) -> str:
    from .types import LINE_NAMES

    return LINE_NAMES[line]
