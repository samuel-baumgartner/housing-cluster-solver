"""Numba-JIT BFS routing on flat grid masks."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap

Cell = Tuple[int, int]
NEIGH4 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32)


@njit(cache=True)
def _touch(reserved: np.ndarray, w: int, h: int, x: int, y: int) -> int:
    score = 0
    for i in range(4):
        nx = x + NEIGH4[i, 0]
        ny = y + NEIGH4[i, 1]
        if 0 <= nx < w and 0 <= ny < h and reserved[ny, nx]:
            score += 1
    return score


@njit(cache=True)
def bfs_dist_flat(
    walkable: np.ndarray,
    w: int,
    h: int,
    sx: int,
    sy: int,
    dist: np.ndarray,
    seen: np.ndarray,
    gen: int,
) -> None:
    n = w * h
    dist.fill(-1)
    if not (0 <= sx < w and 0 <= sy < h) or not walkable[sy, sx]:
        return
    qx = np.empty(n, dtype=np.int32)
    qy = np.empty(n, dtype=np.int32)
    head = 0
    tail = 0
    idx = sy * w + sx
    dist[idx] = 0
    seen[idx] = gen
    qx[tail] = sx
    qy[tail] = sy
    tail += 1
    while head < tail:
        x = qx[head]
        y = qy[head]
        head += 1
        d = dist[y * w + x]
        for i in range(4):
            nx = x + NEIGH4[i, 0]
            ny = y + NEIGH4[i, 1]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            nidx = ny * w + nx
            if seen[nidx] == gen or not walkable[ny, nx]:
                continue
            seen[nidx] = gen
            dist[nidx] = d + 1
            qx[tail] = nx
            qy[tail] = ny
            tail += 1


@njit(cache=True)
def route_shortest_hug_flat(
    walkable: np.ndarray,
    network: np.ndarray,
    paths: np.ndarray,
    reserved: np.ndarray,
    w: int,
    h: int,
    sx: int,
    sy: int,
    dist: np.ndarray,
    seen: np.ndarray,
    gen: int,
    hug: np.ndarray,
    parent: np.ndarray,
    planned_flat: np.ndarray,
) -> np.ndarray:
    """Return route cells as Nx2 int32 array (may be empty)."""
    n = w * h
    out = np.empty((n, 2), dtype=np.int32)
    out_len = 0

    if 0 <= sx < w and 0 <= sy < h and network[sy, sx]:
        return out[:0]

    bfs_dist_flat(walkable, w, h, sx, sy, dist, seen, gen)

    min_d = -1
    for y in range(h):
        for x in range(w):
            if network[y, x]:
                d = dist[y * w + x]
                if d >= 0 and (min_d < 0 or d < min_d):
                    min_d = d
    if min_d < 0:
        return out[:0]

    hug.fill(-1)
    parent.fill(-1)
    start_idx = sy * w + sx
    hug[start_idx] = _touch(reserved, w, h, sx, sy)

    for d in range(1, min_d + 1):
        for y in range(h):
            base = y * w
            for x in range(w):
                idx = base + x
                if dist[idx] != d:
                    continue
                best_hug = -1
                best_parent = -1
                for i in range(4):
                    nx = x + NEIGH4[i, 0]
                    ny = y + NEIGH4[i, 1]
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    pidx = ny * w + nx
                    if dist[pidx] != d - 1 or hug[pidx] < 0:
                        continue
                    score = hug[pidx] + _touch(reserved, w, h, x, y)
                    if score > best_hug:
                        best_hug = score
                        best_parent = pidx
                if best_parent >= 0:
                    hug[idx] = best_hug
                    parent[idx] = best_parent

    goal_x = -1
    goal_y = -1
    goal_hug = -1
    has_planned_goal = False
    for y in range(h):
        for x in range(w):
            if not network[y, x]:
                continue
            if dist[y * w + x] != min_d:
                continue
            hscore = hug[y * w + x]
            if hscore < 0:
                continue
            is_planned = planned_flat[y * w + x]
            if has_planned_goal and not is_planned:
                continue
            if is_planned and not has_planned_goal:
                goal_x, goal_y, goal_hug = x, y, hscore
                has_planned_goal = True
                continue
            if goal_x < 0 or hscore > goal_hug:
                goal_x, goal_y, goal_hug = x, y, hscore

    if goal_x < 0:
        return out[:0]

    goal_idx = goal_y * w + goal_x
    cur = goal_idx
    while cur != start_idx:
        x = cur % w
        y = cur // w
        if not paths[y, x]:
            out[out_len, 0] = x
            out[out_len, 1] = y
            out_len += 1
        cur = parent[cur]
        if cur < 0:
            return out[:0]

    if not paths[sy, sx]:
        out[out_len, 0] = sx
        out[out_len, 1] = sy
        out_len += 1

    # reverse prefix
    for i in range(out_len // 2):
        j = out_len - 1 - i
        tx, ty = out[i, 0], out[i, 1]
        out[i, 0], out[i, 1] = out[j, 0], out[j, 1]
        out[j, 0], out[j, 1] = tx, ty

    # filter network cells
    write = 0
    for i in range(out_len):
        x = out[i, 0]
        y = out[i, 1]
        if not network[y, x]:
            if write != i:
                out[write, 0] = x
                out[write, 1] = y
            write += 1
    return out[:write]


def cells_from_route_array(route_arr: np.ndarray) -> List[Cell]:
    return [(int(route_arr[i, 0]), int(route_arr[i, 1])) for i in range(route_arr.shape[0])]


@njit(cache=True)
def multi_source_bfs_dist_flat(
    walkable: np.ndarray,
    sources: np.ndarray,
    w: int,
    h: int,
    dist: np.ndarray,
    seen: np.ndarray,
    gen: int,
) -> None:
    n = w * h
    dist.fill(-1)
    qx = np.empty(n, dtype=np.int32)
    qy = np.empty(n, dtype=np.int32)
    head = 0
    tail = 0
    for i in range(sources.shape[0]):
        sx = sources[i, 0]
        sy = sources[i, 1]
        if sx < 0 or sx >= w or sy < 0 or sy >= h:
            continue
        if not walkable[sy, sx]:
            continue
        idx = sy * w + sx
        if seen[idx] == gen:
            continue
        seen[idx] = gen
        dist[idx] = 0
        qx[tail] = sx
        qy[tail] = sy
        tail += 1
    while head < tail:
        x = qx[head]
        y = qy[head]
        head += 1
        d = dist[y * w + x]
        for i in range(4):
            nx = x + NEIGH4[i, 0]
            ny = y + NEIGH4[i, 1]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            nidx = ny * w + nx
            if seen[nidx] == gen or not walkable[ny, nx]:
                continue
            seen[nidx] = gen
            dist[nidx] = d + 1
            qx[tail] = nx
            qy[tail] = ny
            tail += 1


@njit(cache=True)
def label_reserved_flat(
    reserved: np.ndarray,
    zone: np.ndarray,
    w: int,
    h: int,
    labels: np.ndarray,
    sizes: np.ndarray,
    seen: np.ndarray,
    gen: int,
) -> int:
    labels.fill(0)
    sizes.fill(0)
    label_id = 0
    qx = np.empty(w * h, dtype=np.int32)
    qy = np.empty(w * h, dtype=np.int32)
    for y in range(h):
        for x in range(w):
            if not reserved[y, x]:
                continue
            idx = y * w + x
            if seen[idx] == gen:
                continue
            label_id += 1
            head = 0
            tail = 0
            seen[idx] = gen
            labels[idx] = label_id
            qx[tail] = x
            qy[tail] = y
            tail += 1
            size = 0
            while head < tail:
                cx = qx[head]
                cy = qy[head]
                head += 1
                size += 1
                for i in range(4):
                    nx = cx + NEIGH4[i, 0]
                    ny = cy + NEIGH4[i, 1]
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    if not reserved[ny, nx]:
                        continue
                    nidx = ny * w + nx
                    if seen[nidx] == gen:
                        continue
                    seen[nidx] = gen
                    labels[nidx] = label_id
                    qx[tail] = nx
                    qy[tail] = ny
                    tail += 1
            sizes[label_id] = size
    return label_id


@njit(cache=True)
def min_manhattan_to_anchors(
    px: int,
    py: int,
    anchors: np.ndarray,
    n: int,
) -> int:
    best = 999
    for i in range(n):
        d = abs(px - anchors[i, 0]) + abs(py - anchors[i, 1])
        if d < best:
            best = d
    return best


@njit(cache=True)
def can_place_fp_flat(
    fp_cells: np.ndarray,
    zone: np.ndarray,
    reserved: np.ndarray,
    paths: np.ndarray,
    planned_flat: np.ndarray,
    w: int,
    h: int,
) -> bool:
    for i in range(fp_cells.shape[0]):
        x = fp_cells[i, 0]
        y = fp_cells[i, 1]
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        if not zone[y, x]:
            return False
        if reserved[y, x]:
            return False
        if paths[y, x]:
            return False
        if planned_flat[y * w + x]:
            return False
    return True


@njit(cache=True)
def door_dist_block_fp_flat(
    walkable: np.ndarray,
    network: np.ndarray,
    fp_cells: np.ndarray,
    w: int,
    h: int,
    sx: int,
    sy: int,
    dist: np.ndarray,
    seen: np.ndarray,
    gen: int,
    backup: np.ndarray,
) -> int:
    n = fp_cells.shape[0]
    for i in range(n):
        x = fp_cells[i, 0]
        y = fp_cells[i, 1]
        if 0 <= x < w and 0 <= y < h:
            backup[i] = 1 if walkable[y, x] else 0
            walkable[y, x] = False
        else:
            backup[i] = 0
    out = door_dist_to_network_flat(walkable, network, w, h, sx, sy, dist, seen, gen)
    for i in range(n):
        if backup[i]:
            x = fp_cells[i, 0]
            y = fp_cells[i, 1]
            walkable[y, x] = True
    return out


@njit(cache=True)
def batch_orient_can_place(
    fp_flat: np.ndarray,
    fp_offsets: np.ndarray,
    zone: np.ndarray,
    reserved: np.ndarray,
    paths: np.ndarray,
    planned_flat: np.ndarray,
    w: int,
    h: int,
    out: np.ndarray,
) -> None:
    n = fp_offsets.shape[0] - 1
    for oi in range(n):
        start = fp_offsets[oi]
        end = fp_offsets[oi + 1]
        if start >= end:
            out[oi] = False
            continue
        ok = True
        for fi in range(start, end):
            x = fp_flat[fi, 0]
            y = fp_flat[fi, 1]
            if x < 0 or x >= w or y < 0 or y >= h:
                ok = False
                break
            if (
                not zone[y, x]
                or reserved[y, x]
                or paths[y, x]
                or planned_flat[y * w + x]
            ):
                ok = False
                break
        out[oi] = ok


@njit(cache=True)
def batch_orient_dists(
    walkable: np.ndarray,
    network: np.ndarray,
    dist_from_net: np.ndarray,
    doors: np.ndarray,
    fp_flat: np.ndarray,
    fp_offsets: np.ndarray,
    can_place: np.ndarray,
    w: int,
    h: int,
    dist: np.ndarray,
    seen: np.ndarray,
    gen: int,
    backup: np.ndarray,
    out_len: np.ndarray,
    out_ok: np.ndarray,
) -> None:
    n = doors.shape[0]
    for oi in range(n):
        if not can_place[oi]:
            out_len[oi] = 999
            out_ok[oi] = 0
            continue
        sx = doors[oi, 0]
        sy = doors[oi, 1]
        if not (0 <= sx < w and 0 <= sy < h):
            out_len[oi] = 999
            out_ok[oi] = 0
            continue
        if network[sy, sx]:
            out_len[oi] = 0
            out_ok[oi] = 1
            continue
        base_d = dist_from_net[sy * w + sx]
        if base_d < 0:
            out_len[oi] = 999
            out_ok[oi] = 0
            continue
        start = fp_offsets[oi]
        end = fp_offsets[oi + 1]
        fp_n = end - start
        min_fp = 999
        for fi in range(start, end):
            fx = fp_flat[fi, 0]
            fy = fp_flat[fi, 1]
            d = abs(fx - sx) + abs(fy - sy)
            if d < min_fp:
                min_fp = d
        if base_d == 0 or min_fp > base_d:
            out_len[oi] = base_d
            out_ok[oi] = 1
            continue
        cur_gen = gen + oi + 1
        dval = door_dist_block_fp_flat(
            walkable,
            network,
            fp_flat[start:end],
            w,
            h,
            sx,
            sy,
            dist,
            seen,
            cur_gen,
            backup[:fp_n],
        )
        if dval < 0:
            out_len[oi] = 999
            out_ok[oi] = 0
        else:
            out_len[oi] = dval
            out_ok[oi] = 1


@njit(cache=True)
def build_reach_zone_free_flat(
    walkable: np.ndarray,
    network: np.ndarray,
    zone: np.ndarray,
    reserved: np.ndarray,
    w: int,
    h: int,
    reach: np.ndarray,
    seen: np.ndarray,
    gen: int,
) -> None:
    reach.fill(False)
    n = w * h
    qx = np.empty(n, dtype=np.int32)
    qy = np.empty(n, dtype=np.int32)
    head = 0
    tail = 0
    for y in range(h):
        for x in range(w):
            if network[y, x]:
                idx = y * w + x
                seen[idx] = gen
                qx[tail] = x
                qy[tail] = y
                tail += 1
    while head < tail:
        x = qx[head]
        y = qy[head]
        head += 1
        if zone[y, x] and not reserved[y, x]:
            reach[y, x] = True
        for i in range(4):
            nx = x + NEIGH4[i, 0]
            ny = y + NEIGH4[i, 1]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            nidx = ny * w + nx
            if seen[nidx] == gen or not walkable[ny, nx]:
                continue
            seen[nidx] = gen
            qx[tail] = nx
            qy[tail] = ny
            tail += 1


@njit(cache=True)
def door_dist_to_network_flat(
    walkable: np.ndarray,
    network: np.ndarray,
    w: int,
    h: int,
    sx: int,
    sy: int,
    dist: np.ndarray,
    seen: np.ndarray,
    gen: int,
) -> int:
    if 0 <= sx < w and 0 <= sy < h and network[sy, sx]:
        return 0
    bfs_dist_flat(walkable, w, h, sx, sy, dist, seen, gen)
    min_d = -1
    for y in range(h):
        row = y * w
        for x in range(w):
            if network[y, x]:
                d = dist[row + x]
                if d >= 0 and (min_d < 0 or d < min_d):
                    min_d = d
    return min_d


@njit(cache=True)
def _fp_contains(fp_cells: np.ndarray, x: int, y: int) -> bool:
    for i in range(fp_cells.shape[0]):
        if fp_cells[i, 0] == x and fp_cells[i, 1] == y:
            return True
    return False


@njit(cache=True)
def _reserved_blob_size(
    reserved: np.ndarray,
    fp_cells: np.ndarray,
    w: int,
    h: int,
    seed_x: int,
    seed_y: int,
    seen: np.ndarray,
    gen: int,
) -> int:
    if not (0 <= seed_x < w and 0 <= seed_y < h):
        return 0
    idx = seed_y * w + seed_x
    if not reserved[seed_y, seed_x] and not _fp_contains(fp_cells, seed_x, seed_y):
        return 0
    qx = np.empty(w * h, dtype=np.int32)
    qy = np.empty(w * h, dtype=np.int32)
    head = 0
    tail = 0
    seen[idx] = gen
    qx[tail] = seed_x
    qy[tail] = seed_y
    tail += 1
    size = 0
    while head < tail:
        x = qx[head]
        y = qy[head]
        head += 1
        size += 1
        for i in range(4):
            nx = x + NEIGH4[i, 0]
            ny = y + NEIGH4[i, 1]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            nidx = ny * w + nx
            if seen[nidx] == gen:
                continue
            if not reserved[ny, nx] and not _fp_contains(fp_cells, nx, ny):
                continue
            seen[nidx] = gen
            qx[tail] = nx
            qy[tail] = ny
            tail += 1
    return size


@njit(cache=True)
def count_near_houses_flat(
    fp_cells: np.ndarray,
    house_cells: np.ndarray,
    house_offsets: np.ndarray,
) -> int:
    near_houses = 0
    n_houses = house_offsets.shape[0] - 1
    fp_len = fp_cells.shape[0]
    for hi in range(n_houses):
        start = house_offsets[hi]
        end = house_offsets[hi + 1]
        found = False
        for ci in range(start, end):
            hx = house_cells[ci, 0]
            hy = house_cells[ci, 1]
            for fi in range(fp_len):
                fx = fp_cells[fi, 0]
                fy = fp_cells[fi, 1]
                if abs(hx - fx) <= 2 and abs(hy - fy) <= 2:
                    found = True
                    break
            if found:
                break
        if found:
            near_houses += 1
    return near_houses


@njit(cache=True)
def score_footprint_flat(
    fp_cells: np.ndarray,
    reserved: np.ndarray,
    zone: np.ndarray,
    paths: np.ndarray,
    network: np.ndarray,
    planned_flat: np.ndarray,
    house_cells: np.ndarray,
    house_offsets: np.ndarray,
    w: int,
    h: int,
    route_len: int,
    origin_y: int,
    south_edge: int,
    zone_cell_count: int,
    free_ratio: float,
    n_houses: int,
    blob_min_houses: int,
    blob_free_ratio: float,
    blob_min_cells: int,
    blob_zone_fraction: float,
    blob_penalty_per_cell: int,
    seen: np.ndarray,
    gen: int,
) -> int:
    touch = 0
    exposed = 0
    fp_len = fp_cells.shape[0]
    for i in range(fp_len):
        x = fp_cells[i, 0]
        y = fp_cells[i, 1]
        for ni in range(4):
            nx = x + NEIGH4[ni, 0]
            ny = y + NEIGH4[ni, 1]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if reserved[ny, nx]:
                touch += 1
            elif not _fp_contains(fp_cells, nx, ny):
                if (
                    zone[ny, nx]
                    and not reserved[ny, nx]
                    and not network[ny, nx]
                    and not paths[ny, nx]
                    and not planned_flat[ny * w + nx]
                ):
                    exposed += 1

    door_inv = 1000 if route_len == 0 else 1000 // (1 + route_len)
    near_houses = count_near_houses_flat(fp_cells, house_cells, house_offsets)

    depth_bonus = 0
    if n_houses >= 3 and south_edge >= 0:
        dy = south_edge - origin_y
        if dy > 0:
            depth_bonus = dy * 4

    blob_penalty = 0
    if n_houses >= blob_min_houses and free_ratio > blob_free_ratio and fp_len > 0:
        cap = blob_min_cells
        cap_calc = int(blob_zone_fraction * zone_cell_count)
        if cap_calc > cap:
            cap = cap_calc
        blob = _reserved_blob_size(
            reserved,
            fp_cells,
            w,
            h,
            fp_cells[0, 0],
            fp_cells[0, 1],
            seen,
            gen,
        )
        if blob > cap:
            blob_penalty = (blob - cap) * blob_penalty_per_cell

    return 4 * touch + door_inv + near_houses + depth_bonus - 3 * exposed - blob_penalty
