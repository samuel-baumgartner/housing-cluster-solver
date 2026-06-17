from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Set, Tuple

from .footprints import footprint_cells, footprint_set
from .fast_grid import GridAccel
from .types import Cell, House, HousingLine

NEIGHBORS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

# Long building-enclosing loop routing (local check near new tile).
MIN_LOOP_INTERIOR_AREA = 6
MIN_LOOP_PERIMETER = 10
MIN_INTERIOR_DIM = 2
LOOP_TILE_VALUE = 3  # effective cost = length - LOOP_TILE_VALUE * loops
MAX_ROUTE_LOOPS = 4


def neighbors4(cell: Cell) -> List[Cell]:
    x, y = cell
    return [(x + dx, y + dy) for dx, dy in NEIGHBORS4]


@dataclass
class WorldGrid:
    width: int
    height: int
    zone: Set[Cell] = field(default_factory=set)
    landscape: Set[Cell] = field(default_factory=set)
    paths: Set[Cell] = field(default_factory=set)  # existing TH-connected paths
    reserved: Set[Cell] = field(default_factory=set)
    planned_paths: Set[Cell] = field(default_factory=set)
    th_paths: Set[Cell] = field(default_factory=set)
    _accel: Optional[GridAccel] = field(default=None, repr=False, compare=False)
    _reserved_version: int = field(default=0, repr=False, compare=False)
    _accel_reserved_version: int = field(default=-1, repr=False, compare=False)
    _loop_routing_cache_key: Optional[Tuple[int, FrozenSet[Cell]]] = field(
        default=None, repr=False, compare=False
    )
    _loop_routing_cache_val: bool = field(default=False, repr=False, compare=False)
    _network_cache_key: Optional[int] = field(default=None, repr=False, compare=False)
    _network_cache: Optional[Set[Cell]] = field(default=None, repr=False, compare=False)
    _planned_version: int = field(default=0, repr=False, compare=False)
    _dist_net_key: Optional[Tuple[int, int]] = field(default=None, repr=False, compare=False)

    def _ensure_accel(self) -> GridAccel:
        if self._accel is None:
            zone_th = {
                c for c in self.zone if c in self.paths and self._path_connected_to_th(c)
            }
            accel = GridAccel(self.width, self.height)
            accel.load_static(self.zone, self.paths, self.th_paths, zone_th)
            self._accel = accel
        if self._accel_reserved_version != self._reserved_version:
            self._accel.sync_reserved(self.reserved)
            self._accel_reserved_version = self._reserved_version
        return self._accel

    def _bump_reserved_version(self) -> None:
        self._reserved_version += 1

    @classmethod
    def demo_deep_zone(
        cls,
        zone_min: Cell = (8, 14),
        zone_max: Cell = (35, 29),
        path_y: int = 30,
        spine_x: int = 7,
    ) -> "WorldGrid":
        xmin, ymin = zone_min
        xmax, ymax = zone_max
        w = max(48, xmax + 4)
        h = max(36, path_y + 4)
        g = cls(width=w, height=h)
        for x in range(w):
            g.paths.add((x, path_y))
        for y in range(path_y + 1):
            g.paths.add((spine_x, y))
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                g.zone.add((x, y))
        g.th_paths = {c for c in g.paths if g._path_connected_to_th(c)}
        return g

    def _path_connected_to_th(self, start: Cell) -> bool:
        if start not in self.paths:
            return False
        q = [start]
        seen = {start}
        head = 0
        while head < len(q):
            c = q[head]
            head += 1
            if c == (0, 0):
                return True
            for n in neighbors4(c):
                if n in seen or n not in self.paths:
                    continue
                seen.add(n)
                q.append(n)
        return start[1] == 30 or start[0] == 7  # spine touch

    def path_network(
        self,
        planned: Optional[Set[Cell]] = None,
    ) -> Set[Cell]:
        planned = planned if planned is not None else self.planned_paths
        key = id(planned)
        if self._network_cache is not None and self._network_cache_key == key:
            return self._network_cache
        net = set(self.th_paths)
        net.update(planned)
        for c in self.zone:
            if c in self.paths and self._path_connected_to_th(c):
                net.add(c)
        self._network_cache_key = key
        self._network_cache = net
        return net

    def invalidate_network_cache(self) -> None:
        self._network_cache_key = None
        self._network_cache = None
        self._loop_routing_cache_key = None
        self._dist_net_key = None

    def sync_solver_snapshot(self, reserved: Set[Cell]) -> None:
        """Apply in-progress solver reserved state; refresh acceleration masks."""
        self.reserved = set(reserved)
        self._accel = None
        self._accel_reserved_version = -1
        self._bump_reserved_version()
        self.invalidate_network_cache()

    def bump_planned_version(self) -> None:
        self._planned_version += 1
        self.invalidate_network_cache()

    def can_place_footprint(
        self,
        cells: Iterable[Cell],
        planned: Optional[Set[Cell]] = None,
    ) -> bool:
        blocked_planned = self.planned_paths if planned is None else planned
        for c in cells:
            if c not in self.zone:
                return False
            if c in self.reserved:
                return False
            if c in blocked_planned:
                return False
            if c in self.paths:
                return False
        return True

    def connection_path_length(
        self,
        start: Cell,
        reserved: Set[Cell],
        planned: Set[Cell],
    ) -> int:
        """New street tiles needed to connect start to the TH path network."""
        if start in self.path_network(planned):
            return 0
        accel = self._ensure_accel()
        extra = reserved.difference(self.reserved)
        saved = self._push_extra(extra)
        try:
            if not self._loop_routing_worthwhile(start, reserved, planned):
                route = accel.route_shortest_hug(start, planned)
            else:
                route = self.route_to_network(start, reserved, planned)
        finally:
            self._pop_extra(saved)
        if not route:
            return 999
        return len(route)

    def _push_extra(self, cells: Set[Cell]) -> List[Cell]:
        accel = self._ensure_accel()
        saved: List[Cell] = []
        for x, y in cells:
            if accel._in_bounds(self.width, self.height, x, y) and not accel.reserved[y, x]:
                saved.append((x, y))
                accel.reserved[y, x] = True
        return saved

    def _pop_extra(self, saved: List[Cell]) -> None:
        accel = self._ensure_accel()
        for x, y in saved:
            accel.reserved[y, x] = False

    @staticmethod
    def _building_touch_score(cell: Cell, reserved: Set[Cell]) -> int:
        """Count orthogonal neighbors occupied by buildings (path-hug tie-break)."""
        return sum(1 for n in neighbors4(cell) if n in reserved)

    def _walkable_for_route(
        self,
        reserved: Set[Cell],
        planned: Set[Cell],
    ) -> Set[Cell]:
        walkable = set(self.zone)
        walkable.update(self.paths)
        walkable.update(planned)
        for c in reserved:
            walkable.discard(c)
        return walkable

    def _is_th_path_goal(self, cell: Cell) -> bool:
        if cell in self.th_paths:
            return True
        return cell in self.paths and self._path_connected_to_th(cell)

    @staticmethod
    def _ring_border(ix: int, iy: int, iw: int, ih: int) -> Set[Cell]:
        border: Set[Cell] = set()
        for x in range(ix - 1, ix + iw + 1):
            border.add((x, iy - 1))
            border.add((x, iy + ih))
        for y in range(iy, iy + ih):
            border.add((ix - 1, y))
            border.add((ix + iw, y))
        return border

    @staticmethod
    def _interior_cells(ix: int, iy: int, iw: int, ih: int) -> Set[Cell]:
        return {(ix + dx, iy + dy) for dx in range(iw) for dy in range(ih)}

    @staticmethod
    def _border_contains(ix: int, iy: int, iw: int, ih: int, cell: Cell) -> bool:
        return cell in WorldGrid._ring_border(ix, iy, iw, ih)

    @staticmethod
    def _iter_candidate_rings_near(cell: Cell) -> Iterator[Tuple[int, int, int, int]]:
        bx, by = cell
        for iw in range(MIN_INTERIOR_DIM, 6):
            for ih in range(MIN_INTERIOR_DIM, 6):
                if iw * ih < MIN_LOOP_INTERIOR_AREA:
                    continue
                if 2 * (iw + ih + 2) < MIN_LOOP_PERIMETER:
                    continue
                for iy in range(by - ih + 1, by + 2):
                    for ix in range(bx - iw + 1, bx + 2):
                        if WorldGrid._border_contains(ix, iy, iw, ih, cell):
                            yield (ix, iy, iw, ih)

    @staticmethod
    def _interior_encloses_buildings(interior: Set[Cell], reserved: Set[Cell]) -> bool:
        if not interior:
            return False
        reserved_count = sum(1 for c in interior if c in reserved)
        if reserved_count >= len(interior) * 0.5:
            return True
        for c in interior:
            if c in reserved:
                return True
            for n in neighbors4(c):
                if n in reserved:
                    return True
        return False

    @staticmethod
    def is_building_loop(
        network: Set[Cell],
        reserved: Set[Cell],
        ix: int,
        iy: int,
        iw: int,
        ih: int,
    ) -> bool:
        interior = WorldGrid._interior_cells(ix, iy, iw, ih)
        border = WorldGrid._ring_border(ix, iy, iw, ih)
        if len(interior) < MIN_LOOP_INTERIOR_AREA:
            return False
        if len(border) < MIN_LOOP_PERIMETER:
            return False
        if not all(b in network for b in border):
            return False
        if any(i in network for i in interior):
            return False
        return WorldGrid._interior_encloses_buildings(interior, reserved)

    @staticmethod
    def new_loops_when_adding(
        cell: Cell,
        network_before: Set[Cell],
        network_after: Set[Cell],
        reserved: Set[Cell],
    ) -> int:
        if cell in network_before:
            return 0
        count = 0
        seen: Set[Tuple[int, int, int, int]] = set()
        for ring in WorldGrid._iter_candidate_rings_near(cell):
            if ring in seen:
                continue
            if WorldGrid.is_building_loop(network_after, reserved, *ring) and not WorldGrid.is_building_loop(
                network_before, reserved, *ring
            ):
                seen.add(ring)
                count += 1
        return count

    @staticmethod
    def _cell_near_reserved(cell: Cell, reserved: Set[Cell], margin: int = 5) -> bool:
        if not reserved:
            return False
        cx, cy = cell
        min_rx = min(r[0] for r in reserved) - margin
        max_rx = max(r[0] for r in reserved) + margin
        min_ry = min(r[1] for r in reserved) - margin
        max_ry = max(r[1] for r in reserved) + margin
        if cx < min_rx or cx > max_rx or cy < min_ry or cy > max_ry:
            return False
        for rx, ry in reserved:
            if abs(rx - cx) + abs(ry - cy) <= margin:
                return True
        return False

    def _loop_routing_worthwhile(
        self,
        start: Cell,
        reserved: Set[Cell],
        planned: Set[Cell],
    ) -> bool:
        """True when a long building loop may be closable near start or planned paths."""
        if not reserved or not planned:
            return False
        seeds: Set[Cell] = set()
        if self._cell_near_reserved(start, reserved, margin=8):
            seeds.add(start)
        for p in planned:
            if self._cell_near_reserved(p, reserved, margin=4):
                seeds.add(p)
        if not seeds:
            return False
        partial = set(planned)
        seen_rings: Set[Tuple[int, int, int, int]] = set()
        for seed in seeds:
            for ring in self._iter_candidate_rings_near(seed):
                if ring in seen_rings:
                    continue
                seen_rings.add(ring)
                interior = self._interior_cells(*ring)
                border = self._ring_border(*ring)
                if len(interior) < MIN_LOOP_INTERIOR_AREA or len(border) < MIN_LOOP_PERIMETER:
                    continue
                if not self._interior_encloses_buildings(interior, reserved):
                    continue
                filled = sum(1 for b in border if b in partial)
                if filled >= len(border) - 2:
                    return True
        return False

    def _bfs_distances(self, start: Cell, walkable: Set[Cell]) -> Dict[Cell, int]:
        dist: Dict[Cell, int] = {start: 0}
        q = [start]
        head = 0
        while head < len(q):
            c = q[head]
            head += 1
            for n in neighbors4(c):
                if n in dist or n not in walkable:
                    continue
                dist[n] = dist[c] + 1
                q.append(n)
        return dist

    def _reconstruct_route(
        self,
        start: Cell,
        goal: Cell,
        parent: Dict[Cell, Optional[Cell]],
        network: Set[Cell],
    ) -> List[Cell]:
        chain: List[Cell] = []
        c: Optional[Cell] = goal
        while c is not None and c != start:
            if c not in self.paths:
                chain.append(c)
            c = parent.get(c)
        chain.reverse()
        if start not in self.paths:
            route = [start] + chain
        else:
            route = chain
        return [cell for cell in route if cell not in network]

    def _route_shortest_hug(
        self,
        start: Cell,
        reserved: Set[Cell],
        planned: Set[Cell],
        network: Set[Cell],
        walkable: Set[Cell],
    ) -> List[Cell]:
        dist = self._bfs_distances(start, walkable)
        goal_dists = [dist[g] for g in dist if g in network]
        if not goal_dists:
            return []
        min_d = min(goal_dists)
        best_goals = [g for g in dist if g in network and dist[g] == min_d]
        planned_goals = [g for g in best_goals if g in planned]
        pick_from = planned_goals or best_goals

        touch = lambda c: self._building_touch_score(c, reserved)
        hug: Dict[Cell, int] = {start: touch(start)}
        parent: Dict[Cell, Optional[Cell]] = {start: None}

        for d in range(1, min_d + 1):
            for c in sorted(cell for cell, dd in dist.items() if dd == d):
                best_parent: Optional[Cell] = None
                best_hug = -1
                for p in neighbors4(c):
                    if dist.get(p) != d - 1 or p not in hug:
                        continue
                    score = hug[p] + touch(c)
                    if score > best_hug:
                        best_hug = score
                        best_parent = p
                if best_parent is None:
                    continue
                hug[c] = best_hug
                parent[c] = best_parent

        goal = max(pick_from, key=lambda g: hug.get(g, -1))
        if goal not in parent:
            return []
        return self._reconstruct_route(start, goal, parent, network)

    def _route_with_loops(
        self,
        start: Cell,
        reserved: Set[Cell],
        planned: Set[Cell],
        network: Set[Cell],
        walkable: Set[Cell],
        dist: Dict[Cell, int],
        min_d: int,
    ) -> List[Cell]:
        max_length = min_d + LOOP_TILE_VALUE * MAX_ROUTE_LOOPS + 2
        touch = lambda c: self._building_touch_score(c, reserved)
        start_state = (start, 0)
        start_new: Set[Cell] = {start}

        heap: List[Tuple[int, int, int, Cell, int]] = [
            (0, -touch(start), 0, start, 0),
        ]
        best: Dict[Tuple[Cell, int], Tuple[int, int, int, Set[Cell]]] = {
            start_state: (0, touch(start), 0, start_new),
        }
        parent: Dict[Tuple[Cell, int], Tuple[Cell, int]] = {}
        best_goal: Optional[Tuple[int, int, int, int, Cell, int]] = None

        while heap:
            eff, nhug, length, c, loops = heapq.heappop(heap)
            state = (c, loops)
            stored = best.get(state)
            if stored is None:
                continue
            p_len, p_hug, p_eff, _ = stored
            if (eff, nhug, length) != (p_eff, -p_hug, p_len):
                continue

            hug = p_hug
            new_tiles = stored[3]

            if best_goal is not None and eff > best_goal[0] + LOOP_TILE_VALUE:
                continue

            if c in network:
                goal = (eff, nhug, 0 if c in planned else 1, length, c, loops)
                if best_goal is None or goal < best_goal:
                    best_goal = goal
                continue

            if loops >= MAX_ROUTE_LOOPS or length >= max_length:
                continue

            for n in neighbors4(c):
                if n not in walkable:
                    continue
                n_length = length + 1
                if n_length > max_length:
                    continue
                n_new = new_tiles
                delta_loops = 0
                if (
                    n not in network
                    and n not in new_tiles
                    and self._cell_near_reserved(n, reserved)
                ):
                    net_before = network | new_tiles
                    net_after = net_before | {n}
                    delta_loops = self.new_loops_when_adding(n, net_before, net_after, reserved)
                    n_new = new_tiles | {n}
                elif n not in network and n not in new_tiles:
                    n_new = new_tiles | {n}
                n_loops = loops + delta_loops
                if n_loops > MAX_ROUTE_LOOPS:
                    continue
                n_state = (n, n_loops)
                n_eff = n_length - LOOP_TILE_VALUE * n_loops
                n_hug = hug + touch(n)

                prev = best.get(n_state)
                if prev is not None:
                    p_len, p_hug, p_eff, _ = prev
                    if (n_eff, -n_hug, n_length) >= (p_eff, -p_hug, p_len):
                        continue

                best[n_state] = (n_length, n_hug, n_eff, n_new)
                parent[n_state] = state
                heapq.heappush(heap, (n_eff, -n_hug, n_length, n, n_loops))

        if best_goal is None:
            return []

        _, _, _, _, goal, goal_loops = best_goal
        goal_state = (goal, goal_loops)

        chain: List[Cell] = []
        cur: Optional[Tuple[Cell, int]] = goal_state
        while cur is not None and cur[0] != start:
            cell = cur[0]
            if cell not in self.paths:
                chain.append(cell)
            cur = parent.get(cur)
        chain.reverse()
        if start not in self.paths:
            route = [start] + chain
        else:
            route = chain
        return [cell for cell in route if cell not in network]

    def search_loop_routing_enabled(self, reserved: Set[Cell], planned: Set[Cell]) -> bool:
        """Once per candidate search: may any route benefit from loop closure?"""
        key = (len(reserved), frozenset(planned))
        if self._loop_routing_cache_key == key:
            return self._loop_routing_cache_val
        result = self._search_loop_routing_enabled_impl(reserved, planned)
        self._loop_routing_cache_key = key
        self._loop_routing_cache_val = result
        return result

    def _search_loop_routing_enabled_impl(self, reserved: Set[Cell], planned: Set[Cell]) -> bool:
        if not reserved or not planned:
            return False
        partial = set(planned)
        seen_rings: Set[Tuple[int, int, int, int]] = set()
        for p in planned:
            if not self._cell_near_reserved(p, reserved, margin=4):
                continue
            for ring in self._iter_candidate_rings_near(p):
                if ring in seen_rings:
                    continue
                seen_rings.add(ring)
                interior = self._interior_cells(*ring)
                border = self._ring_border(*ring)
                if len(interior) < MIN_LOOP_INTERIOR_AREA or len(border) < MIN_LOOP_PERIMETER:
                    continue
                if not self._interior_encloses_buildings(interior, reserved):
                    continue
                filled = sum(1 for b in border if b in partial)
                if filled >= len(border) - 2:
                    return True
        return False

    def route_to_network(
        self,
        start: Cell,
        reserved: Set[Cell],
        planned: Set[Cell],
        *,
        allow_loops: Optional[bool] = None,
    ) -> List[Cell]:
        """Route to path network minimizing length - K*long_building_loops; tie-break hugs."""
        if start in self.path_network(planned):
            return []

        walkable = self._walkable_for_route(reserved, planned)
        if start not in walkable:
            return []

        use_loops = allow_loops
        if use_loops is None:
            use_loops = self._loop_routing_worthwhile(start, reserved, planned)
        elif use_loops and not self._loop_routing_worthwhile(start, reserved, planned):
            use_loops = False

        if not use_loops:
            accel = self._ensure_accel()
            extra = reserved.difference(self.reserved)
            saved = self._push_extra(extra)
            try:
                return accel.route_shortest_hug(start, planned)
            finally:
                self._pop_extra(saved)

        network = self.path_network(planned)
        dist = self._bfs_distances(start, walkable)
        goal_dists = [dist[g] for g in dist if g in network]
        if not goal_dists:
            return []
        min_d = min(goal_dists)
        return self._route_with_loops(start, reserved, planned, network, walkable, dist, min_d)

    def probe_path_connection(
        self,
        path_cell: Cell,
        reserved: Set[Cell],
        planned: Dict[Cell, bool] | Set[Cell],
    ) -> Tuple[bool, Set[Cell]]:
        ok, sim, _ = self.probe_path_connection_route(path_cell, reserved, planned)
        return ok, sim

    def probe_path_connection_route(
        self,
        path_cell: Cell,
        reserved: Set[Cell],
        planned: Dict[Cell, bool] | Set[Cell],
        *,
        allow_loops: Optional[bool] = None,
    ) -> Tuple[bool, Set[Cell], List[Cell]]:
        sim = set(planned) if isinstance(planned, set) else set(planned.keys())
        if path_cell in self.path_network(sim):
            return True, sim, []
        route = self.route_to_network(path_cell, reserved, sim, allow_loops=allow_loops)
        if not route:
            return False, sim, []
        for c in route:
            if c in reserved:
                return False, sim, route
            if c not in self.zone and c not in self.paths:
                return False, sim, route
            sim.add(c)
        return True, sim, route

    def bbox_of(self, cells: Set[Cell]) -> Tuple[Cell, Tuple[int, int]]:
        if not cells:
            return (0, 0), (0, 0)
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return (min_x, min_y), (max_x - min_x + 1, max_y - min_y + 1)

    def unreserved_zone(self) -> Set[Cell]:
        return {c for c in self.zone if c not in self.reserved}

    def reserved_from_houses(self, houses: List[House]) -> Set[Cell]:
        out: Set[Cell] = set()
        for h in houses:
            out.update(
                footprint_set(h.origin, h.line, h.quarter_turns, h.flipped)
            )
        return out
