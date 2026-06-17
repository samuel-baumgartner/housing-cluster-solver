"""Numpy-backed grid masks and BFS for hot solve paths."""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .fast_bfs import (
    build_reach_zone_free_flat,
    can_place_fp_flat,
    cells_from_route_array,
    door_dist_block_fp_flat,
    batch_orient_dists,
    door_dist_to_network_flat,
    multi_source_bfs_dist_flat,
    route_shortest_hug_flat,
    score_footprint_flat,
)

from .types import Cell

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


class GridAccel:
    """Flat-array acceleration for a fixed-size WorldGrid."""

    __slots__ = (
        "width",
        "height",
        "zone",
        "paths",
        "reserved",
        "th_paths",
        "zone_th_paths",
        "_dist",
        "_seen_gen",
        "_gen",
        "_hug",
        "_parent",
        "_planned_flat",
        "_route_buf",
        "_fp_buf",
        "_dist_from_net",
        "_fp_backup",
        "_reach_buf",
        "_base_walk",
        "_walkable_live",
        "_network_live",
        "_planned_cells",
        "_dist_net_key",
        "_reach_key",
    )

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.zone = np.zeros((height, width), dtype=np.bool_)
        self.paths = np.zeros((height, width), dtype=np.bool_)
        self.reserved = np.zeros((height, width), dtype=np.bool_)
        self.th_paths = np.zeros((height, width), dtype=np.bool_)
        self.zone_th_paths = np.zeros((height, width), dtype=np.bool_)
        n = height * width
        self._dist = np.full(n, -1, dtype=np.int32)
        self._seen_gen = np.zeros(n, dtype=np.int32)
        self._gen = 1
        self._hug = np.full(n, -1, dtype=np.int16)
        self._parent = np.full(n, -1, dtype=np.int32)
        self._planned_flat = np.zeros(n, dtype=np.bool_)
        self._route_buf = np.zeros((n, 2), dtype=np.int32)
        self._fp_buf = np.zeros((64, 2), dtype=np.int32)
        self._dist_from_net = np.full(n, -1, dtype=np.int32)
        self._fp_backup = np.zeros(64, dtype=np.int8)
        self._reach_buf = np.zeros((height, width), dtype=np.bool_)
        self._base_walk = np.zeros((height, width), dtype=np.bool_)
        self._walkable_live = np.zeros((height, width), dtype=np.bool_)
        self._network_live = np.zeros((height, width), dtype=np.bool_)
        self._planned_cells: Set[Cell] = set()
        self._dist_net_key: Optional[Tuple[int, int]] = None
        self._reach_key: Optional[Tuple[int, int]] = None

    def _bump_gen(self) -> int:
        self._gen += 1
        if self._gen >= 2_000_000_000:
            self._seen_gen.fill(0)
            self._gen = 1
        return self._gen

    @staticmethod
    def _in_bounds(width: int, height: int, x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    def load_static(
        self,
        zone: Set[Cell],
        paths: Set[Cell],
        th_paths: Set[Cell],
        zone_th_paths: Set[Cell],
    ) -> None:
        self.zone.fill(False)
        self.paths.fill(False)
        self.th_paths.fill(False)
        self.zone_th_paths.fill(False)
        for x, y in zone:
            if self._in_bounds(self.width, self.height, x, y):
                self.zone[y, x] = True
        for x, y in paths:
            if self._in_bounds(self.width, self.height, x, y):
                self.paths[y, x] = True
        for x, y in th_paths:
            if self._in_bounds(self.width, self.height, x, y):
                self.th_paths[y, x] = True
        for x, y in zone_th_paths:
            if self._in_bounds(self.width, self.height, x, y):
                self.zone_th_paths[y, x] = True
        self._base_walk = (self.zone | self.paths).copy()
        self._network_live = self.th_paths | self.zone_th_paths
        self._rebuild_walkable_live()

    def _rebuild_walkable_live(self) -> None:
        self._walkable_live = self._base_walk.copy()
        self._walkable_live &= ~self.reserved

    def sync_planned(self, planned: Set[Cell]) -> None:
        if planned == self._planned_cells:
            return
        if self._planned_cells and planned >= self._planned_cells:
            for x, y in planned - self._planned_cells:
                if self._in_bounds(self.width, self.height, x, y):
                    if not self.reserved[y, x]:
                        self._walkable_live[y, x] = True
                    self._network_live[y, x] = True
            self._planned_cells = set(planned)
            return
        self._planned_cells = set(planned)
        self._rebuild_walkable_live()
        for x, y in self._planned_cells:
            if self._in_bounds(self.width, self.height, x, y):
                if not self.reserved[y, x]:
                    self._walkable_live[y, x] = True
                self._network_live[y, x] = True

    def sync_reserved(self, reserved: Set[Cell]) -> None:
        self.reserved.fill(False)
        for x, y in reserved:
            if self._in_bounds(self.width, self.height, x, y):
                self.reserved[y, x] = True
        self._rebuild_walkable_live()
        for x, y in self._planned_cells:
            if self._in_bounds(self.width, self.height, x, y):
                if not self.reserved[y, x]:
                    self._walkable_live[y, x] = True

    def walkable_mask(self, planned: Set[Cell]) -> np.ndarray:
        self.sync_planned(planned)
        return self._walkable_live

    def network_mask(self, planned: Set[Cell]) -> np.ndarray:
        self.sync_planned(planned)
        return self._network_live

    def _touch_score(self, x: int, y: int) -> int:
        score = 0
        w, h = self.width, self.height
        for dx, dy in NEIGHBORS4:
            nx, ny = x + dx, y + dy
            if self._in_bounds(w, h, nx, ny) and self.reserved[ny, nx]:
                score += 1
        return score

    def bfs_distances(self, start: Cell, walkable: np.ndarray) -> np.ndarray:
        sx, sy = start
        if not self._in_bounds(self.width, self.height, sx, sy):
            return self._dist
        gen = self._bump_gen()
        dist = self._dist
        seen = self._seen_gen
        dist.fill(-1)
        if not walkable[sy, sx]:
            return dist
        head = 0
        q: List[Tuple[int, int]] = [(sx, sy)]
        idx = sy * self.width + sx
        dist[idx] = 0
        seen[idx] = gen
        w, h = self.width, self.height
        while head < len(q):
            x, y = q[head]
            head += 1
            d = dist[y * w + x]
            for dx, dy in NEIGHBORS4:
                nx, ny = x + dx, y + dy
                if not self._in_bounds(w, h, nx, ny):
                    continue
                nidx = ny * w + nx
                if seen[nidx] == gen or not walkable[ny, nx]:
                    continue
                seen[nidx] = gen
                dist[nidx] = d + 1
                q.append((nx, ny))
        return dist

    def route_shortest_hug(
        self,
        start: Cell,
        planned: Set[Cell],
    ) -> List[Cell]:
        network = self.network_mask(planned)
        sx, sy = start
        w, h = self.width, self.height
        if self._in_bounds(w, h, sx, sy) and network[sy, sx]:
            return []

        walkable = self.walkable_mask(planned)
        self._planned_flat.fill(False)
        for x, y in planned:
            if self._in_bounds(w, h, x, y):
                self._planned_flat[y * w + x] = True

        gen = self._bump_gen()
        route_arr = route_shortest_hug_flat(
            walkable,
            network,
            self.paths,
            self.reserved,
            w,
            h,
            sx,
            sy,
            self._dist,
            self._seen_gen,
            gen,
            self._hug,
            self._parent,
            self._planned_flat,
        )
        return cells_from_route_array(route_arr)

    def build_reach_zone_free(
        self,
        planned: Set[Cell],
    ) -> np.ndarray:
        """Reachable zone cells (not reserved) from path network."""
        walkable = self.walkable_mask(planned)
        network = self.network_mask(planned)
        gen = self._bump_gen()
        build_reach_zone_free_flat(
            walkable,
            network,
            self.zone,
            self.reserved,
            self.width,
            self.height,
            self._reach_buf,
            self._seen_gen,
            gen,
        )
        return self._reach_buf.copy()

    def strands_large_pocket(
        self,
        reserved_with_fp: Set[Cell],
        reach_before: np.ndarray,
        reach_after: np.ndarray,
        min_pocket: int,
    ) -> bool:
        free = self.zone.copy()
        for x, y in reserved_with_fp:
            if self._in_bounds(self.width, self.height, x, y):
                free[y, x] = False
        gen = self._bump_gen()
        seen = self._seen_gen
        w, h = self.width, self.height
        for y in range(h):
            for x in range(w):
                if not free[y, x]:
                    continue
                idx = y * w + x
                if seen[idx] == gen:
                    continue
                # BFS component
                size = 0
                had_before = False
                still_after = False
                q = [(x, y)]
                seen[idx] = gen
                head = 0
                while head < len(q):
                    cx, cy = q[head]
                    head += 1
                    size += 1
                    if reach_before[cy, cx]:
                        had_before = True
                    if reach_after[cy, cx]:
                        still_after = True
                    for dx, dy in NEIGHBORS4:
                        nx, ny = cx + dx, cy + dy
                        if not self._in_bounds(w, h, nx, ny) or not free[ny, nx]:
                            continue
                        nidx = ny * w + nx
                        if seen[nidx] == gen:
                            continue
                        seen[nidx] = gen
                        q.append((nx, ny))
                if size >= min_pocket and had_before and not still_after:
                    return True
        return False

    def reach_with_extra_reserved(
        self,
        planned: Set[Cell],
        extra_reserved: Set[Cell],
    ) -> np.ndarray:
        """Reach zone-free mask with temporary extra reserved cells."""
        saved: List[Tuple[int, int]] = []
        for x, y in extra_reserved:
            if self._in_bounds(self.width, self.height, x, y):
                if not self.reserved[y, x]:
                    saved.append((x, y))
                self.reserved[y, x] = True
        reach = self.build_reach_zone_free(planned)
        for x, y in saved:
            self.reserved[y, x] = False
        return reach

    def _fp_array(self, fp: FrozenSet[Cell]) -> np.ndarray:
        n = len(fp)
        if n > self._fp_buf.shape[0]:
            self._fp_buf = np.zeros((n, 2), dtype=np.int32)
            self._fp_backup = np.zeros(n, dtype=np.int8)
        arr = self._fp_buf[:n]
        for i, (x, y) in enumerate(fp):
            arr[i, 0] = x
            arr[i, 1] = y
        return arr

    def _orient_offsets(self, okey: tuple[int, int, bool]) -> Tuple[Cell, ...]:
        from .footprints import _ORIENT_OFFSETS

        return _ORIENT_OFFSETS[okey]

    def _door_offsets(self, okey: tuple[int, int, bool]) -> Cell:
        from .footprints import _DOOR_PATH_OFFSETS

        return _DOOR_PATH_OFFSETS[okey]

    def ensure_reach_before(
        self,
        reserved_version: int,
        planned_version: int,
        planned: Set[Cell],
    ) -> np.ndarray:
        key = (reserved_version, planned_version)
        if self._reach_key == key:
            return self._reach_buf
        self._reach_key = key
        walkable = self.walkable_mask(planned)
        network = self.network_mask(planned)
        gen = self._bump_gen()
        build_reach_zone_free_flat(
            walkable,
            network,
            self.zone,
            self.reserved,
            self.width,
            self.height,
            self._reach_buf,
            self._seen_gen,
            gen,
        )
        return self._reach_buf

    def ensure_dist_from_network(
        self,
        reserved_version: int,
        planned_version: int,
        walk_scratch: np.ndarray,
        network: np.ndarray,
    ) -> None:
        key = (reserved_version, planned_version)
        if self._dist_net_key == key:
            return
        self._dist_net_key = key
        w, h = self.width, self.height
        sources = np.empty((w * h, 2), dtype=np.int32)
        count = 0
        for y in range(h):
            for x in range(w):
                if network[y, x]:
                    sources[count, 0] = x
                    sources[count, 1] = y
                    count += 1
        if count == 0:
            self._dist_from_net.fill(-1)
            return
        gen = self._bump_gen()
        multi_source_bfs_dist_flat(
            walk_scratch,
            sources[:count],
            w,
            h,
            self._dist_from_net,
            self._seen_gen,
            gen,
        )

    def precompute_dist_from_network(
        self, walk_scratch: np.ndarray, network: np.ndarray
    ) -> None:
        """Legacy entry point — prefer ensure_dist_from_network with version keys."""
        w, h = self.width, self.height
        sources = np.empty((w * h, 2), dtype=np.int32)
        count = 0
        for y in range(h):
            for x in range(w):
                if network[y, x]:
                    sources[count, 0] = x
                    sources[count, 1] = y
                    count += 1
        if count == 0:
            self._dist_from_net.fill(-1)
            return
        gen = self._bump_gen()
        multi_source_bfs_dist_flat(
            walk_scratch,
            sources[:count],
            w,
            h,
            self._dist_from_net,
            self._seen_gen,
            gen,
        )

    def door_dist_with_fp(
        self,
        path_cell: Cell,
        fp: FrozenSet[Cell],
        walk_scratch: np.ndarray,
        network: np.ndarray,
    ) -> int:
        w, h = self.width, self.height
        sx, sy = path_cell
        if self._in_bounds(w, h, sx, sy) and network[sy, sx]:
            return 0
        fp_arr = self._fp_array(fp)
        n = len(fp)
        if not self._in_bounds(w, h, sx, sy):
            return -1
        gen = self._bump_gen()
        return int(
            door_dist_block_fp_flat(
                walk_scratch,
                network,
                fp_arr,
                w,
                h,
                sx,
                sy,
                self._dist,
                self._seen_gen,
                gen,
                self._fp_backup[:n],
            )
        )

    def can_place_offsets(
        self,
        origin: Cell,
        offsets: Tuple[Cell, ...],
        planned_flat: np.ndarray,
    ) -> bool:
        ox, oy = origin
        w, h = self.width, self.height
        for dx, dy in offsets:
            x = ox + dx
            y = oy + dy
            if not self._in_bounds(w, h, x, y):
                return False
            if not self.zone[y, x]:
                return False
            if self.reserved[y, x]:
                return False
            if self.paths[y, x]:
                return False
            if planned_flat[y * w + x]:
                return False
        return True

    def door_dist_with_offsets(
        self,
        door: Cell,
        origin: Cell,
        offsets: Tuple[Cell, ...],
        walk_scratch: np.ndarray,
        network: np.ndarray,
    ) -> int:
        w, h = self.width, self.height
        sx, sy = door
        if self._in_bounds(w, h, sx, sy) and network[sy, sx]:
            return 0
        if not self._in_bounds(w, h, sx, sy):
            return -1
        ox, oy = origin
        n = len(offsets)
        if n > self._fp_buf.shape[0]:
            self._fp_buf = np.zeros((n, 2), dtype=np.int32)
            self._fp_backup = np.zeros(n, dtype=np.int8)
        fp_arr = self._fp_buf[:n]
        for i, (dx, dy) in enumerate(offsets):
            fp_arr[i, 0] = ox + dx
            fp_arr[i, 1] = oy + dy
        base_d = int(self._dist_from_net[sy * w + sx])
        if base_d < 0:
            return -1
        if base_d == 0:
            return 0
        min_fp = 999
        for i in range(n):
            fx = fp_arr[i, 0]
            fy = fp_arr[i, 1]
            d = abs(fx - sx) + abs(fy - sy)
            if d < min_fp:
                min_fp = d
        if min_fp > base_d:
            return base_d
        gen = self._bump_gen()
        return int(
            door_dist_block_fp_flat(
                walk_scratch,
                network,
                fp_arr,
                w,
                h,
                sx,
                sy,
                self._dist,
                self._seen_gen,
                gen,
                self._fp_backup[:n],
            )
        )

    def planned_flat_from(self, planned: Set[Cell]) -> np.ndarray:
        w, h = self.width, self.height
        flat = np.zeros(w * h, dtype=np.bool_)
        for x, y in planned:
            if self._in_bounds(w, h, x, y):
                flat[y * w + x] = True
        return flat

    def route_with_extra_fp(
        self,
        start: Cell,
        planned_flat: np.ndarray,
        walkable: np.ndarray,
        network: np.ndarray,
        extra_fp: FrozenSet[Cell],
        walk_scratch: np.ndarray,
        reserved_scratch: np.ndarray,
    ) -> List[Cell]:
        w, h = self.width, self.height
        sx, sy = start
        if self._in_bounds(w, h, sx, sy) and network[sy, sx]:
            return []

        saved_w: List[Tuple[int, int]] = []
        saved_r: List[Tuple[int, int]] = []
        for x, y in extra_fp:
            if not self._in_bounds(w, h, x, y):
                continue
            if walk_scratch[y, x]:
                saved_w.append((x, y))
                walk_scratch[y, x] = False
            if not reserved_scratch[y, x]:
                saved_r.append((x, y))
                reserved_scratch[y, x] = True

        if not walk_scratch[sy, sx]:
            for x, y in saved_w:
                walk_scratch[y, x] = True
            for x, y in saved_r:
                reserved_scratch[y, x] = False
            return []

        try:
            gen = self._bump_gen()
            route_arr = route_shortest_hug_flat(
                walk_scratch,
                network,
                self.paths,
                reserved_scratch,
                w,
                h,
                sx,
                sy,
                self._dist,
                self._seen_gen,
                gen,
                self._hug,
                self._parent,
                planned_flat,
            )
            return cells_from_route_array(route_arr)
        finally:
            for x, y in saved_w:
                walk_scratch[y, x] = True
            for x, y in saved_r:
                reserved_scratch[y, x] = False

    def score_fp(
        self,
        fp: FrozenSet[Cell],
        route_len: int,
        origin_y: int,
        south_edge: int,
        zone_cell_count: int,
        free_ratio: float,
        n_houses: int,
        house_cells: np.ndarray,
        house_offsets: np.ndarray,
        planned_flat: np.ndarray,
        network: np.ndarray,
        walkable: np.ndarray,
        score_gen: int,
    ) -> int:
        from .scoring import (
            BLOB_MIN_CELLS,
            BLOB_PENALTY_FREE_RATIO,
            BLOB_PENALTY_MIN_HOUSES,
            BLOB_PENALTY_PER_CELL,
            BLOB_ZONE_FRACTION,
        )

        fp_arr = self._fp_array(fp)
        return int(
            score_footprint_flat(
                fp_arr,
                self.reserved,
                self.zone,
                self.paths,
                network,
                planned_flat,
                house_cells,
                house_offsets,
                self.width,
                self.height,
                route_len,
                origin_y,
                south_edge if south_edge is not None else -1,
                zone_cell_count,
                free_ratio,
                n_houses,
                BLOB_PENALTY_MIN_HOUSES,
                BLOB_PENALTY_FREE_RATIO,
                BLOB_MIN_CELLS,
                BLOB_ZONE_FRACTION,
                BLOB_PENALTY_PER_CELL,
                self._seen_gen,
                score_gen,
            )
        )


class SearchPass:
    """Per find_best_candidate caches for fast evaluate + score."""

    __slots__ = (
        "grid",
        "planned",
        "network",
        "walkable",
        "network_mask",
        "planned_flat",
        "reach_before",
        "house_cells",
        "house_offsets",
        "house_fps",
        "free_ratio",
        "south_edge",
        "zone_cell_count",
        "zone_size",
        "eval_cache",
        "dist_cache",
        "route_cache",
        "accel",
        "_score_gen",
        "walk_scratch",
        "reserved_scratch",
        "_orient_doors",
        "_orient_fp_flat",
        "_orient_fp_offsets",
        "_orient_can_place",
        "_orient_lens",
        "_orient_ok",
    )

    def __init__(
        self,
        grid: "WorldGrid",
        planned: Set[Cell],
        houses: List,
        house_fps: Tuple[FrozenSet[Cell], ...],
        free_ratio: float,
        south_edge: Optional[int],
        zone_size: Tuple[int, int],
    ) -> None:
        self.grid = grid
        self.planned = planned
        self.accel = grid._ensure_accel()
        self.network = grid.path_network(planned)
        self.accel.sync_planned(planned)
        self.network_mask = self.accel._network_live
        self.walkable = self.accel._walkable_live
        self.planned_flat = self.accel.planned_flat_from(planned)
        self.reach_before = self.accel.ensure_reach_before(
            grid._reserved_version, grid._planned_version, planned
        ).copy()
        self.house_fps = house_fps
        self.house_cells, self.house_offsets = _pack_house_cells(house_fps)
        self.free_ratio = free_ratio
        self.south_edge = south_edge
        self.zone_cell_count = len(grid.zone)
        self.zone_size = zone_size
        self.eval_cache: Dict[
            Tuple[Cell, int, int, bool],
            Tuple[FrozenSet[Cell], Optional[Set[Cell]], int, bool],
        ] = {}
        self.dist_cache: Dict[Tuple[Cell, FrozenSet[Cell]], Tuple[int, bool]] = {}
        self.route_cache: Dict[
            Tuple[Cell, FrozenSet[Cell]],
            Tuple[bool, Set[Cell], List[Cell], int],
        ] = {}
        self._score_gen = 0
        self.walk_scratch = self.walkable.copy()
        self.reserved_scratch = self.accel.reserved.copy()
        self.accel.ensure_dist_from_network(
            grid._reserved_version,
            grid._planned_version,
            self.walk_scratch,
            self.network_mask,
        )
        self._orient_doors = np.zeros((16, 2), dtype=np.int32)
        self._orient_fp_flat = np.zeros((512, 2), dtype=np.int32)
        self._orient_fp_offsets = np.zeros(17, dtype=np.int32)
        self._orient_can_place = np.zeros(16, dtype=np.bool_)
        self._orient_lens = np.zeros(16, dtype=np.int32)
        self._orient_ok = np.zeros(16, dtype=np.int8)

    def batch_line_orients(
        self,
        origin: Cell,
        line: int,
        orients: List[Tuple[int, bool]],
    ) -> List[Tuple[int, int, bool, FrozenSet[Cell], int, bool]]:
        from .footprints import _DOOR_PATH_OFFSETS, _ORIENT_OFFSETS, footprint_set
        from .types import HousingLine

        hl = HousingLine(line)
        n = len(orients)
        ox, oy = origin
        if n > self._orient_doors.shape[0]:
            self._orient_doors = np.zeros((n, 2), dtype=np.int32)
            self._orient_can_place = np.zeros(n, dtype=np.bool_)
            self._orient_lens = np.zeros(n, dtype=np.int32)
            self._orient_ok = np.zeros(n, dtype=np.int8)
            self._orient_fp_offsets = np.zeros(n + 1, dtype=np.int32)

        fp_count = 0
        for i, (qt, flip) in enumerate(orients):
            okey = (line, qt, flip)
            offsets = _ORIENT_OFFSETS[okey]
            ok_place = bool(offsets) and self.accel.can_place_offsets(
                origin, offsets, self.planned_flat
            )
            self._orient_can_place[i] = ok_place
            if ok_place:
                ddx, ddy = _DOOR_PATH_OFFSETS[okey]
                self._orient_doors[i, 0] = ox + ddx
                self._orient_doors[i, 1] = oy + ddy
                self._orient_fp_offsets[i] = fp_count
                for dx, dy in offsets:
                    if fp_count >= self._orient_fp_flat.shape[0]:
                        grown = np.zeros((fp_count + 64, 2), dtype=np.int32)
                        grown[:fp_count] = self._orient_fp_flat[:fp_count]
                        self._orient_fp_flat = grown
                    self._orient_fp_flat[fp_count, 0] = ox + dx
                    self._orient_fp_flat[fp_count, 1] = oy + dy
                    fp_count += 1
            else:
                self._orient_doors[i, 0] = 0
                self._orient_doors[i, 1] = 0
                self._orient_fp_offsets[i] = fp_count
        self._orient_fp_offsets[n] = fp_count

        gen = self.accel._bump_gen()
        batch_orient_dists(
            self.walk_scratch,
            self.network_mask,
            self.accel._dist_from_net,
            self._orient_doors[:n],
            self._orient_fp_flat[:fp_count],
            self._orient_fp_offsets[: n + 1],
            self._orient_can_place[:n],
            self.accel.width,
            self.accel.height,
            self.accel._dist,
            self.accel._seen_gen,
            gen,
            self.accel._fp_backup,
            self._orient_lens[:n],
            self._orient_ok[:n],
        )

        out: List[Tuple[int, int, bool, Optional[FrozenSet[Cell]], int, bool]] = []
        for i, (qt, flip) in enumerate(orients):
            if not self._orient_can_place[i]:
                out.append((999, qt, flip, None, 999, False))
                continue
            route_len = int(self._orient_lens[i])
            ok = bool(self._orient_ok[i])
            out.append((route_len, qt, flip, None, route_len, ok))
            key = (origin, line, qt, flip)
            self.eval_cache[key] = (None, None, route_len, ok)
        return out

    def _next_score_gen(self) -> int:
        self._score_gen = self.accel._bump_gen()
        return self._score_gen

    def evaluate(
        self,
        origin: Cell,
        line: int,
        qt: int,
        flipped: bool,
    ) -> Tuple[FrozenSet[Cell], Optional[Set[Cell]], int, bool]:
        from .footprints import _DOOR_PATH_OFFSETS, _ORIENT_OFFSETS, footprint_set
        from .types import HousingLine

        key = (origin, line, qt, flipped)
        if key in self.eval_cache:
            return self.eval_cache[key]
        okey = (line, qt, flipped)
        offsets = _ORIENT_OFFSETS[okey]
        if not offsets or not self.accel.can_place_offsets(
            origin, offsets, self.planned_flat
        ):
            result: Tuple[FrozenSet[Cell], Optional[Set[Cell]], int, bool] = (
                frozenset(),
                None,
                999,
                False,
            )
            self.eval_cache[key] = result
            return result
        ox, oy = origin
        ddx, ddy = _DOOR_PATH_OFFSETS[okey]
        door = (ox + ddx, oy + ddy)
        dist = self.accel.door_dist_with_offsets(
            door, origin, offsets, self.walk_scratch, self.network_mask
        )
        if dist < 0:
            result = (frozenset(), None, 999, False)
        else:
            result = (None, None, dist, True)
        self.eval_cache[key] = result
        if result[0] is not None:
            self.dist_cache[(door, result[0])] = (result[2], result[3])
        return result

    def _dist_for(self, path_cell: Cell, fp: FrozenSet[Cell]) -> Tuple[int, bool]:
        rkey = (path_cell, fp)
        if rkey in self.dist_cache:
            return self.dist_cache[rkey]
        if path_cell in self.network:
            entry = (0, True)
            self.dist_cache[rkey] = entry
            return entry
        sx, sy = path_cell
        w = self.accel.width
        h = self.accel.height
        if not (0 <= sx < w and 0 <= sy < h):
            entry = (999, False)
            self.dist_cache[rkey] = entry
            return entry
        base_d = int(self.accel._dist_from_net[sy * w + sx])
        if base_d < 0:
            entry = (999, False)
            self.dist_cache[rkey] = entry
            return entry
        if base_d == 0:
            entry = (0, True)
            self.dist_cache[rkey] = entry
            return entry
        min_fp = 999
        for fx, fy in fp:
            d = abs(fx - sx) + abs(fy - sy)
            if d < min_fp:
                min_fp = d
        if min_fp > base_d:
            entry = (base_d, True)
            self.dist_cache[rkey] = entry
            return entry
        dist = self.accel.door_dist_with_fp(
            path_cell, fp, self.walk_scratch, self.network_mask
        )
        if dist < 0:
            entry = (999, False)
        else:
            entry = (dist, True)
        self.dist_cache[rkey] = entry
        return entry

    def full_connection(
        self,
        path_cell: Cell,
        fp: FrozenSet[Cell],
    ) -> Tuple[bool, Set[Cell], int]:
        """Full hug route + sim_planned for commit checks."""
        ok, sim, route_len = self._route_for(path_cell, fp)
        return ok, sim, route_len

    def _route_for(
        self,
        path_cell: Cell,
        fp: FrozenSet[Cell],
    ) -> Tuple[bool, Optional[Set[Cell]], int]:
        rkey = (path_cell, fp)
        if rkey in self.route_cache:
            ok, sim, route, route_len = self.route_cache[rkey]
            return ok, sim, route_len
        if path_cell in self.network:
            entry = (True, set(self.planned), [], 0)
            self.route_cache[rkey] = entry
            return True, entry[1], 0
        route = self.accel.route_with_extra_fp(
            path_cell,
            self.planned_flat,
            self.walkable,
            self.network_mask,
            fp,
            self.walk_scratch,
            self.reserved_scratch,
        )
        if not route:
            entry = (False, set(self.planned), [], 999)
            self.route_cache[rkey] = entry
            return False, entry[1], 999
        sim = set(self.planned)
        for x, y in route:
            c = (x, y)
            if c in self.grid.reserved or c in fp:
                entry = (False, sim, route, 999)
                self.route_cache[rkey] = entry
                return False, sim, 999
            if c not in self.grid.zone and c not in self.grid.paths:
                entry = (False, sim, route, 999)
                self.route_cache[rkey] = entry
                return False, sim, 999
            sim.add(c)
        route_len = 0 if path_cell in self.network else len(route)
        entry = (True, sim, route, route_len)
        self.route_cache[rkey] = entry
        return True, sim, route_len

    def score(
        self,
        origin: Cell,
        route_len: int,
        fp: FrozenSet[Cell],
        n_houses: int,
    ) -> int:
        gen = self._next_score_gen()
        return self.accel.score_fp(
            fp,
            route_len,
            origin[1],
            self.south_edge,
            self.zone_cell_count,
            self.free_ratio,
            n_houses,
            self.house_cells,
            self.house_offsets,
            self.planned_flat,
            self.network_mask,
            self.walkable,
            gen,
        )


def _pack_house_cells(
    house_fps: Tuple[FrozenSet[Cell], ...],
) -> Tuple[np.ndarray, np.ndarray]:
    total = sum(len(fp) for fp in house_fps)
    cells = np.empty((total, 2), dtype=np.int32)
    offsets = np.zeros(len(house_fps) + 1, dtype=np.int32)
    k = 0
    for i, fp in enumerate(house_fps):
        offsets[i] = k
        for x, y in fp:
            cells[k, 0] = x
            cells[k, 1] = y
            k += 1
    offsets[len(house_fps)] = k
    return cells, offsets
