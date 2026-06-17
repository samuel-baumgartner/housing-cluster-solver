"""Organic landscape blobs carved from the buildable zone before housing placement."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Set

from .grid import WorldGrid, neighbors4
from .types import Cell

DEFAULT_CIRCLE_CELLS = 300
PATH_BUFFER_DEPTH = 4
MAX_ZONE_FRACTION = 0.18
MAX_TOTAL_FRACTION = 0.30
MAX_BLOBS = 2
MAX_BLOBS_SMALL_ZONE = 550


def _path_buffer(zone: Set[Cell], grid: WorldGrid, depth: int) -> Set[Cell]:
    """Zone cells within `depth` steps of the TH path network — kept buildable."""
    network = grid.path_network(set())
    seeds = [c for c in zone if any(n in network for n in neighbors4(c))]
    protected: Set[Cell] = set()
    q = list(seeds)
    dist = {c: 0 for c in seeds}
    head = 0
    while head < len(q):
        cell = q[head]
        head += 1
        d = dist[cell]
        if d >= depth:
            continue
        protected.add(cell)
        for n in neighbors4(cell):
            if n not in zone or n in dist:
                continue
            dist[n] = d + 1
            q.append(n)
    return protected


def _organic_blob(
    zone: Set[Cell],
    center: Cell,
    target: int,
    rng: random.Random,
) -> Set[Cell]:
    """Compact irregular circle — radius scales with target cell count."""
    cx, cy = center
    radius = math.sqrt(target / math.pi)
    rx = radius * rng.uniform(0.92, 1.08)
    ry = radius * rng.uniform(0.88, 1.05)
    rot = rng.uniform(0.0, math.pi)
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    phase_a = rng.uniform(0.0, math.tau)
    phase_b = rng.uniform(0.0, math.tau)
    scored: List[tuple[float, Cell]] = []
    for cell in zone:
        x, y = cell[0] - cx, cell[1] - cy
        xr = x * cos_r + y * sin_r
        yr = -x * sin_r + y * cos_r
        angle = math.atan2(yr, xr) if xr or yr else 0.0
        ell = math.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
        wobble = 1.0 + 0.22 * math.sin(angle * 2.3 + phase_a)
        wobble += 0.12 * math.sin(angle * 4.9 + phase_b)
        score = ell * wobble + rng.uniform(0.0, 0.12)
        scored.append((score, cell))
    scored.sort(key=lambda item: item[0])
    return {cell for _, cell in scored[: min(target, len(scored))]}


def _pick_centers(
    zone: Set[Cell],
    count: int,
    rng: random.Random,
    *,
    path_dist: Optional[Dict[Cell, int]] = None,
) -> List[Cell]:
    xs = [c[0] for c in zone]
    ys = [c[1] for c in zone]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    ranked: List[tuple[float, Cell]] = []
    for cell in zone:
        margin = min(
            cell[0] - xmin,
            xmax - cell[0],
            cell[1] - ymin,
            ymax - cell[1],
        )
        away = path_dist.get(cell, 0) if path_dist else 0
        ranked.append((margin * 0.4 + away * 0.8 + rng.uniform(0.0, 1.0), cell))
    ranked.sort(key=lambda item: item[0], reverse=True)
    centers: List[Cell] = []
    min_sep = max(10, int(math.sqrt(len(zone) / max(count, 1))))
    for _, cell in ranked:
        if len(centers) >= count:
            break
        if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) >= min_sep for other in centers):
            centers.append(cell)
    while len(centers) < count:
        centers.append(rng.choice(list(zone)))
    return centers


def place_natural_circles(
    grid: WorldGrid,
    *,
    cells_per_circle: int = DEFAULT_CIRCLE_CELLS,
    count: Optional[int] = None,
    seed: int = 42,
) -> Set[Cell]:
    """Remove compact organic blobs from the zone before buildings or paths are placed."""
    remaining = set(grid.zone)
    if not remaining:
        grid.landscape = set()
        return set()
    protected = _path_buffer(remaining, grid, PATH_BUFFER_DEPTH)
    carve_pool = remaining - protected
    if len(carve_pool) < 80:
        grid.landscape = set()
        return set()

    original_size = len(remaining)
    per_blob_cap = min(cells_per_circle, max(80, int(original_size * MAX_ZONE_FRACTION)))
    total_cap = min(len(carve_pool), int(original_size * MAX_TOTAL_FRACTION))
    carve_budget = total_cap
    if carve_budget < 80:
        grid.landscape = set()
        return set()

    max_blobs = 1 if original_size < MAX_BLOBS_SMALL_ZONE else MAX_BLOBS
    if count is None:
        count = min(max_blobs, max(1, carve_budget // per_blob_cap))
    count = min(count, max_blobs, max(1, carve_budget // 80))

    rng = random.Random(seed)
    path_dist: Dict[Cell, int] = {}
    network = grid.path_network(set())
    for cell in carve_pool:
        path_dist[cell] = min(
            abs(cell[0] - p[0]) + abs(cell[1] - p[1]) for p in network
        )
    centers = _pick_centers(carve_pool, count, rng, path_dist=path_dist)
    placed: Set[Cell] = set()
    pool = set(carve_pool)
    for index, center in enumerate(centers):
        budget_left = carve_budget - len(placed)
        if budget_left < 80 or not pool:
            break
        slots_left = len(centers) - index
        share = budget_left // slots_left + budget_left % slots_left
        target = min(per_blob_cap, share, len(pool))
        target = max(80, target)
        blob = _organic_blob(pool, center, target, rng)
        if not blob:
            continue
        placed |= blob
        pool -= blob
        grid.zone -= blob
    grid.landscape = placed
    grid._accel = None
    grid.invalidate_network_cache()
    return placed
