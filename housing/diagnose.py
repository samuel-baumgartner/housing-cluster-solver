from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from .footprints import entrance_door_cell, entrance_path_cell, footprint_set
from .grid import WorldGrid
from .scoring import (
    STRANDED_POCKET_MIN,
    build_route_parent_seeds,
    solver_placement_score,
    strands_large_pocket,
)
from .types import Cell, House, HousingLine

LINE_NAMES = {HousingLine.SMALL: "small", HousingLine.BIG: "big", HousingLine.L: "L"}


@dataclass
class PlacementDiagnosis:
    origin: Cell
    line: HousingLine
    quarter_turns: int
    flipped: bool
    ok: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    footprint: Set[Cell] = field(default_factory=set)
    door_cell: Optional[Cell] = None
    path_cell: Optional[Cell] = None
    route: List[Cell] = field(default_factory=list)
    score: int = 0
    score_base: int = 0
    score_mix: int = 0

    @property
    def label(self) -> str:
        return (
            f"{LINE_NAMES[self.line]}@{self.origin} "
            f"qt={self.quarter_turns} flip={self.flipped}"
        )


def diagnose_placement(
    origin: Cell,
    line: HousingLine,
    quarter_turns: int,
    flipped: bool,
    grid: WorldGrid,
    houses: List[House],
    planned: Set[Cell],
    th_paths: Set[Cell],
    *,
    check_strand: bool = True,
) -> PlacementDiagnosis:
    """Explain why a candidate would be accepted or rejected."""
    grid.sync_solver_snapshot(set(grid.reserved))

    diag = PlacementDiagnosis(
        origin=origin,
        line=line,
        quarter_turns=quarter_turns,
        flipped=flipped,
        ok=False,
    )
    fp = footprint_set(origin, line, quarter_turns, flipped)
    diag.footprint = fp
    if not fp:
        diag.reasons.append("Empty footprint.")
        return diag

    door = entrance_door_cell(origin, line, quarter_turns, flipped)
    path_cell = entrance_path_cell(origin, line, quarter_turns, flipped)
    diag.door_cell = door
    diag.path_cell = path_cell

    reserved = set(grid.reserved)
    outside_zone = [c for c in fp if c not in grid.zone]
    if outside_zone:
        diag.reasons.append(
            f"Footprint leaves green zone ({len(outside_zone)} cells outside)."
        )

    on_reserved = sorted(fp & reserved)
    if on_reserved:
        sample = ", ".join(f"({x},{y})" for x, y in on_reserved[:4])
        extra = f" +{len(on_reserved) - 4} more" if len(on_reserved) > 4 else ""
        diag.reasons.append(
            f"Overlaps {len(on_reserved)} reserved building cell(s): {sample}{extra}."
        )

    on_planned = sorted(fp & planned)
    if on_planned:
        sample = ", ".join(f"({x},{y})" for x, y in on_planned[:4])
        extra = f" +{len(on_planned) - 4} more" if len(on_planned) > 4 else ""
        diag.reasons.append(
            f"Overlaps {len(on_planned)} planned street cell(s): {sample}{extra}."
        )

    on_paths = sorted(fp & grid.paths)
    if on_paths:
        sample = ", ".join(f"({x},{y})" for x, y in on_paths[:4])
        diag.reasons.append(f"Overlaps {len(on_paths)} built path cell(s): {sample}.")

    if diag.reasons:
        return diag

    _, zone_size = grid.bbox_of(grid.zone)
    diag.score_base, diag.score_mix, diag.score, _ = solver_placement_score(
        origin, line, quarter_turns, flipped, grid, houses, planned
    )

    reserved_copy = set(reserved)
    reserved_copy.update(fp)
    ok_path, sim_planned = grid.probe_path_connection(path_cell, reserved_copy, planned)
    if not ok_path:
        diag.reasons.append(
            f"Door path at {path_cell} cannot reach the path network "
            f"(no route through green / planned / built paths)."
        )
        return diag

    route = grid.route_to_network(path_cell, reserved_copy, sim_planned)
    diag.route = route
    if path_cell not in grid.paths and path_cell not in planned:
        if not route:
            diag.warnings.append(
                f"Door path {path_cell} is already on network; no new tiles needed."
            )
        else:
            new_tiles = [c for c in route if c not in grid.paths]
            diag.warnings.append(
                f"Would add {len(new_tiles)} planned path tile(s) for door access."
            )

    if check_strand:
        reach_before = build_route_parent_seeds(grid, reserved, planned, th_paths)
        reach_before = {
            c for c in reach_before if c in grid.zone and c not in reserved
        }
        reach_after = build_route_parent_seeds(
            grid, reserved_copy, sim_planned, th_paths
        )
        reach_after = {
            c for c in reach_after if c in grid.zone and c not in reserved_copy
        }
        if strands_large_pocket(
            grid, reserved_copy, sim_planned, reach_before, reach_after
        ):
            diag.reasons.append(
                f"Strand reject: would cut off a large green pocket "
                f"(≥{STRANDED_POCKET_MIN} cells) from path access."
            )
            return diag

    diag.ok = True
    diag.warnings.append("Would commit successfully.")
    return diag
