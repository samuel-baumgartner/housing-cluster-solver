#!/usr/bin/env python3
"""Step-by-step matplotlib animation of the housing cluster solver."""

from __future__ import annotations

import argparse
from typing import List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

from housing.diagnose import PlacementDiagnosis, diagnose_placement
from housing.footprints import entrance_door_cell, footprint_set
from housing.grid import WorldGrid
from housing.solver import solve
from housing.types import LINE_COLORS, HousingLine, SolveStep

# Display colors
COLOR_ZONE = "#3d8b40"
COLOR_PATH = "#c4a574"
COLOR_PLANNED = "#e8c87a"
COLOR_FRONTIER = "#ffffff44"
COLOR_HIGHLIGHT = "#ffeb3b"
COLOR_ROUTE = "#ff5722"
COLOR_GRID = "#1a1a2e"
COLOR_BG = "#16213e"
COLOR_TRY_OK = "#66ffcc"
COLOR_TRY_BAD = "#ff6666"
DISTRICT_COLORS = [
    "#4a6fa533",
    "#6a4a8a33",
    "#4a8a6a33",
    "#8a6a4a33",
    "#4a7a8a33",
    "#8a4a6a33",
]


def _parse_cell(value: str) -> Tuple[int, int]:
    parts = value.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected X,Y got {value!r}")
    return int(parts[0]), int(parts[1])


def _parse_size(value: str) -> Tuple[int, int]:
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected WxH got {value!r}")
    w, h = int(parts[0]), int(parts[1])
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError("zone size must be at least 1×1")
    return w, h


def _zone_presets() -> dict:
    return {
        "deep": ((8, 14), (35, 29), 30),
        "shallow": ((8, 18), (35, 23), 24),
    }


def _build_grid(args: argparse.Namespace) -> WorldGrid:
    if args.zone == "pocket":
        grid = WorldGrid(width=48, height=36)
        for y in range(0, 31):
            grid.paths.add((10, y))
        for c in [(x, y) for y in range(16, 20) for x in range(11, 16)]:
            grid.zone.add(c)
        for c in [(x, y) for y in range(14, 20) for x in range(16, 24)]:
            grid.zone.add(c)
        grid.th_paths = {c for c in grid.paths if grid._path_connected_to_th(c)}
        return grid

    presets = _zone_presets()
    zone_min, zone_max, path_y = presets.get(args.zone, presets["deep"])
    custom = args.zone_size is not None or args.zone_max is not None or args.zone_min is not None

    if args.zone_min is not None:
        zone_min = args.zone_min
    if args.zone_max is not None:
        zone_max = args.zone_max
    elif args.zone_size is not None:
        zw, zh = args.zone_size
        zone_max = (zone_min[0] + zw - 1, zone_min[1] + zh - 1)
    if custom and args.path_y is None:
        path_y = zone_max[1] + 1
    if args.path_y is not None:
        path_y = args.path_y

    return WorldGrid.demo_deep_zone(
        zone_min=zone_min,
        zone_max=zone_max,
        path_y=path_y,
    )


def _zone_label(grid: WorldGrid) -> str:
    if not grid.zone:
        return "0×0"
    xs = [c[0] for c in grid.zone]
    ys = [c[1] for c in grid.zone]
    return f"{max(xs) - min(xs) + 1}×{max(ys) - min(ys) + 1}"


def _apply_step_state(grid: WorldGrid, step: SolveStep) -> None:
    grid.sync_solver_snapshot(step.reserved)


def _format_diagnosis(diag: PlacementDiagnosis) -> str:
    lines = [
        "TRY PLACEMENT",
        diag.label,
        "",
        f"Score: {diag.score} (base {diag.score_base} + mix {diag.score_mix})",
        f"Door: {diag.door_cell}  Path: {diag.path_cell}",
        "",
    ]
    if diag.ok:
        lines.append("OK — would commit")
    else:
        lines.append("REJECTED")
    for reason in diag.reasons:
        lines.append(f"  • {reason}")
    for note in diag.warnings:
        if note != "Would commit successfully.":
            lines.append(f"  · {note}")
    lines.extend(
        [
            "",
            "Keys: click origin",
            "  1/2/3 small/big/L",
            "  r rotate  x flip",
            "  t try mode  Esc exit try",
        ]
    )
    return "\n".join(lines)


def draw_step(
    ax,
    grid: WorldGrid,
    step: SolveStep,
    step_idx: int,
    total: int,
    *,
    try_diag: Optional[PlacementDiagnosis] = None,
    try_mode: bool = False,
) -> None:
    ax.clear()
    ax.set_facecolor(COLOR_BG)
    ax.set_aspect("equal")
    margin = 2
    ax.set_xlim(-margin, grid.width + margin)
    ax.set_ylim(grid.height + margin, -margin)  # y down
    title = f"Step {step_idx + 1}/{total}: [{step.kind}] {step.title}"
    if try_mode:
        title += "  [TRY MODE — click grid]"
    ax.set_title(
        title,
        color="white",
        fontsize=11,
        pad=8,
    )
    if step.detail:
        ax.text(
            0.01,
            0.02,
            step.detail,
            transform=ax.transAxes,
            color="#aaaacc",
            fontsize=9,
            verticalalignment="bottom",
        )

    # Existing paths (outside zone too)
    for c in grid.paths:
        ax.add_patch(
            plt.Rectangle(
                (c[0] - 0.05, c[1] - 0.05),
                0.9,
                0.9,
                facecolor=COLOR_PATH,
                edgecolor="none",
            )
        )

    # Green zone (free)
    free_zone = grid.zone - step.reserved - step.planned_paths
    for c in free_zone:
        ax.add_patch(
            plt.Rectangle(
                (c[0], c[1]),
                1,
                1,
                facecolor=COLOR_ZONE,
                edgecolor="#2a5a2e",
                linewidth=0.3,
                alpha=0.55,
            )
        )

    # District interiors (layout / per-district growth)
    if step.districts:
        for d in step.districts:
            tint = DISTRICT_COLORS[d.id % len(DISTRICT_COLORS)]
            for c in d.interior:
                if c in free_zone:
                    ax.add_patch(
                        plt.Rectangle(
                            (c[0], c[1]),
                            1,
                            1,
                            facecolor=tint,
                            edgecolor="none",
                            alpha=0.85,
                        )
                    )
            for c in d.ring:
                if c not in step.planned_paths and c not in grid.paths:
                    ax.add_patch(
                        plt.Rectangle(
                            (c[0], c[1]),
                            1,
                            1,
                            facecolor="#5a4a3a",
                            edgecolor="#887766",
                            linewidth=0.4,
                            alpha=0.45,
                        )
                    )

    # Planned paths
    for c in step.planned_paths:
        ax.add_patch(
            plt.Rectangle(
                (c[0] - 0.05, c[1] - 0.05),
                0.9,
                0.9,
                facecolor=COLOR_PLANNED,
                edgecolor="#aa8833",
                linewidth=0.5,
            )
        )

    # Frontier dots
    for c in step.frontier:
        ax.plot(c[0] + 0.5, c[1] + 0.5, "o", color="white", markersize=3, alpha=0.35)

    # Placed houses at this step
    for h in step.houses:
        fp = footprint_set(h.origin, h.line, h.quarter_turns, h.flipped)
        color = LINE_COLORS[h.line]
        for c in fp:
            ax.add_patch(
                plt.Rectangle(
                    (c[0] + 0.05, c[1] + 0.05),
                    0.9,
                    0.9,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.9,
                )
            )
        door = entrance_door_cell(h.origin, h.line, h.quarter_turns, h.flipped)
        ax.plot(
            door[0] + 0.5,
            door[1] + 0.5,
            "D",
            color="#ffeb3b",
            markersize=6,
            markeredgecolor="black",
        )
        if h.path_cell:
            ax.plot(
                h.path_cell[0] + 0.5,
                h.path_cell[1] + 0.5,
                "s",
                color="white",
                markersize=5,
                markeredgecolor="black",
            )

    # Candidate highlight
    for c in step.highlight_footprint:
        ax.add_patch(
            plt.Rectangle(
                (c[0], c[1]),
                1,
                1,
                facecolor=COLOR_HIGHLIGHT,
                edgecolor="white",
                linewidth=1.2,
                alpha=0.55,
            )
        )

    # Path route being added
    for i, c in enumerate(step.highlight_path_route):
        ax.add_patch(
            plt.Rectangle(
                (c[0] - 0.05, c[1] - 0.05),
                0.9,
                0.9,
                facecolor=COLOR_ROUTE,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.85,
            )
        )
        ax.text(
            c[0] + 0.5,
            c[1] + 0.5,
            str(i + 1),
            ha="center",
            va="center",
            color="white",
            fontsize=7,
        )

    # Interactive try-placement overlay
    if try_diag is not None:
        color = COLOR_TRY_OK if try_diag.ok else COLOR_TRY_BAD
        for c in try_diag.footprint:
            ax.add_patch(
                plt.Rectangle(
                    (c[0], c[1]),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.4,
                    alpha=0.65,
                )
            )
        if try_diag.door_cell:
            ax.plot(
                try_diag.door_cell[0] + 0.5,
                try_diag.door_cell[1] + 0.5,
                "D",
                color="#ffeb3b",
                markersize=8,
                markeredgecolor="black",
            )
        if try_diag.path_cell:
            ax.plot(
                try_diag.path_cell[0] + 0.5,
                try_diag.path_cell[1] + 0.5,
                "s",
                color="white",
                markersize=6,
                markeredgecolor="black",
            )
        for i, c in enumerate(try_diag.route):
            ax.add_patch(
                plt.Rectangle(
                    (c[0] - 0.05, c[1] - 0.05),
                    0.9,
                    0.9,
                    facecolor=COLOR_ROUTE,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.85,
                )
            )
            ax.text(
                c[0] + 0.5,
                c[1] + 0.5,
                str(i + 1),
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )
        ax.text(
            1.01,
            0.98,
            _format_diagnosis(try_diag),
            transform=ax.transAxes,
            color="#ccccdd",
            fontsize=8,
            verticalalignment="top",
            family="monospace",
        )
    # Top candidates legend
    elif step.top_candidates:
        lines = ["Top scores:"]
        for i, cand in enumerate(step.top_candidates[:5]):
            mark = "→" if step.selected and cand.key() == step.selected.key() else " "
            lines.append(
                f"{mark} {cand.score:4d}  {cand.label}@{cand.origin} qt{cand.quarter_turns}"
            )
        ax.text(
            1.01,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            color="#ccccdd",
            fontsize=8,
            verticalalignment="top",
            family="monospace",
        )

    # Legend
    patches = [
        mpatches.Patch(color=COLOR_ZONE, label="green zone"),
        mpatches.Patch(color=LINE_COLORS[1], label="small"),
        mpatches.Patch(color=LINE_COLORS[2], label="big"),
        mpatches.Patch(color=LINE_COLORS[3], label="L"),
        mpatches.Patch(color=COLOR_PLANNED, label="planned path"),
        mpatches.Patch(color=COLOR_HIGHLIGHT, label="candidate"),
    ]
    ax.legend(
        handles=patches,
        loc="lower left",
        fontsize=7,
        facecolor="#222244",
        labelcolor="white",
    )
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#444466")


class StepPlayer:
    def __init__(
        self,
        grid: WorldGrid,
        steps: List[SolveStep],
        *,
        start_try: bool = False,
        start_step: int = 0,
    ) -> None:
        self.grid = grid
        self.steps = steps
        self.idx = max(0, min(len(steps) - 1, start_step))
        self.try_mode = start_try
        self.try_origin: Optional[Tuple[int, int]] = None
        self.try_line = HousingLine.SMALL
        self.try_qt = 0
        self.try_flipped = False
        self.try_diag: Optional[PlacementDiagnosis] = None
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.patch.set_facecolor(COLOR_BG)
        plt.subplots_adjust(bottom=0.12, right=0.82)
        ax_prev = self.fig.add_axes([0.15, 0.02, 0.12, 0.05])
        ax_next = self.fig.add_axes([0.30, 0.02, 0.12, 0.05])
        ax_play = self.fig.add_axes([0.45, 0.02, 0.12, 0.05])
        ax_try = self.fig.add_axes([0.60, 0.02, 0.12, 0.05])
        self.btn_prev = Button(ax_prev, "◀ Prev")
        self.btn_next = Button(ax_next, "Next ▶")
        self.btn_play = Button(ax_play, "Auto ▶")
        self.btn_try = Button(ax_try, "Try")
        self.btn_prev.on_clicked(lambda _: self._move(-1))
        self.btn_next.on_clicked(lambda _: self._move(1))
        self.btn_play.on_clicked(lambda _: self._auto())
        self.btn_try.on_clicked(lambda _: self._toggle_try())
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._timer = None
        self._draw()

    def _current_step(self) -> SolveStep:
        return self.steps[self.idx]

    def _th_paths(self) -> Set[Tuple[int, int]]:
        return set(self.grid.th_paths)

    def _evaluate_try(self) -> None:
        if self.try_origin is None:
            self.try_diag = None
            return
        step = self._current_step()
        _apply_step_state(self.grid, step)
        self.try_diag = diagnose_placement(
            self.try_origin,
            self.try_line,
            self.try_qt,
            self.try_flipped,
            self.grid,
            step.houses,
            set(step.planned_paths),
            self._th_paths(),
        )

    def _toggle_try(self) -> None:
        self.try_mode = not self.try_mode
        if self.try_mode:
            self._evaluate_try()
        else:
            self.try_diag = None
        self._draw()

    def _on_click(self, event) -> None:
        if not self.try_mode or event.inaxes != self.ax or event.xdata is None:
            return
        self.try_origin = (int(event.xdata), int(event.ydata))
        self._evaluate_try()
        self._draw()

    def _draw(self) -> None:
        draw_step(
            self.ax,
            self.grid,
            self._current_step(),
            self.idx,
            len(self.steps),
            try_diag=self.try_diag if self.try_mode else None,
            try_mode=self.try_mode,
        )
        self.fig.canvas.draw_idle()

    def _move(self, delta: int) -> None:
        self.idx = max(0, min(len(self.steps) - 1, self.idx + delta))
        if self.try_mode:
            self._evaluate_try()
        self._draw()

    def _on_key(self, event) -> None:
        if self.try_mode:
            if event.key == "escape":
                self.try_mode = False
                self.try_diag = None
                self._draw()
                return
            if event.key == "t":
                self._toggle_try()
                return
            if event.key == "r":
                self.try_qt = (self.try_qt + 1) % 4
                self._evaluate_try()
                self._draw()
                return
            if event.key == "x":
                self.try_flipped = not self.try_flipped
                self._evaluate_try()
                self._draw()
                return
            if event.key == "1":
                self.try_line = HousingLine.SMALL
                self._evaluate_try()
                self._draw()
                return
            if event.key == "2":
                self.try_line = HousingLine.BIG
                self._evaluate_try()
                self._draw()
                return
            if event.key == "3":
                self.try_line = HousingLine.L
                self._evaluate_try()
                self._draw()
                return
        if event.key == "t":
            self._toggle_try()
            return
        if event.key in ("right", "down", " "):
            self._move(1)
        elif event.key in ("left", "up"):
            self._move(-1)
        elif event.key == "home":
            self.idx = 0
            if self.try_mode:
                self._evaluate_try()
            self._draw()
        elif event.key == "end":
            self.idx = len(self.steps) - 1
            if self.try_mode:
                self._evaluate_try()
            self._draw()

    def _auto(self) -> None:
        if self.idx < len(self.steps) - 1:
            self._move(1)
            self.fig.canvas.new_timer(interval=600).single_shot(
                0, self._auto
            )

    def run(self) -> None:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate housing cluster solver")
    parser.add_argument(
        "--zone",
        default="deep",
        choices=["deep", "pocket", "shallow"],
        help="Demo scenario",
    )
    parser.add_argument(
        "--zone-size",
        type=_parse_size,
        metavar="WxH",
        help="Green zone size in cells, e.g. 28x16 (uses --zone-min as top-left)",
    )
    parser.add_argument(
        "--zone-min",
        type=_parse_cell,
        metavar="X,Y",
        help="Green zone top-left corner (default: 8,14 for deep, 8,18 for shallow)",
    )
    parser.add_argument(
        "--zone-max",
        type=_parse_cell,
        metavar="X,Y",
        help="Green zone bottom-right corner (overrides --zone-size)",
    )
    parser.add_argument(
        "--path-y",
        type=int,
        metavar="Y",
        help="Road row below zone (default: preset value, or zone_max_y+1 when size is custom)",
    )
    parser.add_argument("--save-gif", metavar="PATH", help="Export animation as GIF")
    parser.add_argument(
        "--try-mode",
        action="store_true",
        dest="try_mode",
        help="Start in interactive try-placement mode",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        metavar="N",
        help="Start at solver step N (1-based, for use with --try)",
    )
    args = parser.parse_args()

    if args.zone == "pocket" and (
        args.zone_size is not None or args.zone_min is not None or args.zone_max is not None
    ):
        parser.error("--zone-size, --zone-min, and --zone-max apply to deep/shallow only")

    grid = _build_grid(args)

    print(
        f"Solving {args.zone} zone ({_zone_label(grid)}, {len(grid.zone)} cells)..."
    )
    result = solve(grid, record_steps=True)
    print(f"Done: {len(result.houses)} houses, {len(result.steps)} animation steps")

    if args.save_gif:
        try:
            import matplotlib.animation as animation

            fig, ax = plt.subplots(figsize=(12, 7))
            fig.patch.set_facecolor(COLOR_BG)

            def update(i: int):
                draw_step(ax, grid, result.steps[i], i, len(result.steps))

            anim = animation.FuncAnimation(
                fig,
                update,
                frames=len(result.steps),
                interval=800,
                repeat=True,
            )
            anim.save(args.save_gif, writer="pillow", fps=1.2)
            print(f"Saved {args.save_gif}")
        except ImportError:
            print("Install pillow for GIF export: pip install pillow")
    else:
        start = max(0, min(len(result.steps) - 1, args.step - 1))
        player = StepPlayer(
            grid,
            result.steps,
            start_try=args.try_mode,
            start_step=start,
        )
        if args.try_mode:
            print(
                f"Try mode at step {start + 1}. Click the grid to test placements.\n"
                "Keys: 1/2/3 line, r rotate, x flip, t toggle try, Esc exit try."
            )
        player.run()


if __name__ == "__main__":
    main()
