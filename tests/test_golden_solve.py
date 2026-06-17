"""Golden solve regression: identical 15-home placement on demo_deep_zone."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pytest

from housing.grid import WorldGrid
from housing.solver import solve

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "bench" / "golden_deep_15.json"


def _load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


def _result_payload(result) -> dict:
    return {
        "houses": [
            {
                "origin": list(h.origin),
                "line": int(h.line),
                "quarter_turns": h.quarter_turns,
                "flipped": h.flipped,
                "path_cell": list(h.path_cell) if h.path_cell else None,
            }
            for h in result.houses
        ],
        "planned_paths": sorted([list(c) for c in result.planned_paths]),
    }


def test_matches_golden_placement() -> None:
    golden = _load_golden()
    grid = WorldGrid.demo_deep_zone()
    result = solve(grid, record_steps=False)
    assert _result_payload(result) == golden


@pytest.mark.slow
def test_solve_30x20_under_one_second() -> None:
    zone_kw = dict(zone_min=(8, 14), zone_max=(37, 33), path_y=34)
    solve(WorldGrid.demo_deep_zone(**zone_kw), record_steps=False)
    times: list[float] = []
    houses = 0
    for _ in range(3):
        t0 = time.perf_counter()
        result = solve(WorldGrid.demo_deep_zone(**zone_kw), record_steps=False)
        times.append(time.perf_counter() - t0)
        houses = len(result.houses)
    elapsed = statistics.median(times)
    assert houses >= 10
    assert elapsed < 1.0, f"30×20 solve median took {elapsed:.3f}s"


@pytest.mark.slow
def test_solve_under_one_second() -> None:
    grid = WorldGrid.demo_deep_zone()
    # Warm numba JIT before timed run (matches bench_solve.py).
    solve(WorldGrid.demo_deep_zone(), record_steps=False)
    t0 = time.perf_counter()
    result = solve(grid, record_steps=False)
    elapsed = time.perf_counter() - t0
    assert len(result.houses) == len(_load_golden()["houses"])
    assert elapsed < 1.0, f"solve took {elapsed:.3f}s"
