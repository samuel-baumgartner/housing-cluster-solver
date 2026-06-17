from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]


class HousingLine(IntEnum):
    SMALL = 1
    BIG = 2
    L = 3


LINE_NAMES = {
    HousingLine.SMALL: "small",
    HousingLine.BIG: "big",
    HousingLine.L: "L",
}

LINE_COLORS = {
    HousingLine.SMALL: "#4a90d9",
    HousingLine.BIG: "#e67e22",
    HousingLine.L: "#9b59b6",
}


@dataclass
class House:
    origin: Cell
    line: HousingLine
    quarter_turns: int = 0
    flipped: bool = False
    path_cell: Optional[Cell] = None

    @property
    def label(self) -> str:
        return LINE_NAMES[self.line]


@dataclass
class Candidate:
    origin: Cell
    line: HousingLine
    quarter_turns: int
    flipped: bool
    score: int

    def key(self) -> str:
        return f"{self.origin[0]},{self.origin[1]}|{self.line}|{self.quarter_turns}|{self.flipped}"

    @property
    def label(self) -> str:
        return LINE_NAMES[self.line]


@dataclass
class SolveStep:
    """One frame in the step-by-step animation."""

    kind: str
    title: str
    detail: str = ""
    houses: List[House] = field(default_factory=list)
    planned_paths: Set[Cell] = field(default_factory=set)
    reserved: Set[Cell] = field(default_factory=set)
    frontier: Set[Cell] = field(default_factory=set)
    highlight_footprint: Set[Cell] = field(default_factory=set)
    highlight_path_route: List[Cell] = field(default_factory=list)
    landscape: Set[Cell] = field(default_factory=set)
    top_candidates: List[Candidate] = field(default_factory=list)
    selected: Optional[Candidate] = None


@dataclass
class SolveResult:
    houses: List[House]
    planned_paths: Set[Cell]
    steps: List[SolveStep]
