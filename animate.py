#!/usr/bin/env python3
"""Step-by-step matplotlib animation of the housing cluster solver."""

from __future__ import annotations

import argparse
from typing import List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
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

_DOOR_RGB = (255, 235, 59)
_PATH_RGB = (196, 165, 116)
_PLANNED_RGB = (232, 200, 122)
_ZONE_RGB = (61, 139, 64)
_BG_RGB = (22, 33, 62)
_HIGHLIGHT_RGB = (255, 235, 59)
_ROUTE_RGB = (255, 87, 34)
_FRONTIER_RGB = (255, 255, 255)
_FRONTIER_ALPHA = 0x44 / 255.0  # matches COLOR_FRONTIER #ffffff44
_TRY_OK_RGB = (102, 255, 204)
_TRY_BAD_RGB = (255, 102, 102)
_LINE_RGB = {
    HousingLine.SMALL: (74, 144, 217),
    HousingLine.BIG: (230, 126, 34),
    HousingLine.L: (155, 89, 182),
}


def _cells_xy(cells, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    if not cells:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    xs = np.fromiter((c[0] for c in cells), dtype=np.int32, count=len(cells))
    ys = np.fromiter((c[1] for c in cells), dtype=np.int32, count=len(cells))
    ok = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    return xs[ok], ys[ok]


def _paint(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, rgb: Tuple[int, int, int]) -> None:
    if xs.size == 0:
        return
    img[ys, xs, 0] = rgb[0]
    img[ys, xs, 1] = rgb[1]
    img[ys, xs, 2] = rgb[2]
    img[ys, xs, 3] = 255


def _blend(
    img: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    rgb: Tuple[int, int, int],
    alpha: float,
) -> None:
    if xs.size == 0:
        return
    a = int(alpha * 255)
    inv = 255 - a
    fg = np.array(rgb, dtype=np.uint16)
    bg = img[ys, xs, :3].astype(np.uint16)
    img[ys, xs, :3] = ((fg * a + bg * inv) // 255).astype(np.uint8)
    img[ys, xs, 3] = 255


def build_step_rgba(
    grid: WorldGrid,
    step: SolveStep,
    *,
    try_diag: Optional[PlacementDiagnosis] = None,
) -> np.ndarray:
    h, w = grid.height, grid.width
    img = np.empty((h, w, 4), dtype=np.uint8)
    img[..., 0] = _BG_RGB[0]
    img[..., 1] = _BG_RGB[1]
    img[..., 2] = _BG_RGB[2]
    img[..., 3] = 255

    xs, ys = _cells_xy(grid.paths, w, h)
    _paint(img, xs, ys, _PATH_RGB)

    free_zone = grid.zone - step.reserved - step.planned_paths
    xs, ys = _cells_xy(free_zone, w, h)
    _paint(img, xs, ys, _ZONE_RGB)

    xs, ys = _cells_xy(step.planned_paths, w, h)
    _paint(img, xs, ys, _PLANNED_RGB)

    for hse in step.houses:
        fp = footprint_set(hse.origin, hse.line, hse.quarter_turns, hse.flipped)
        xs, ys = _cells_xy(fp, w, h)
        _paint(img, xs, ys, _LINE_RGB[hse.line])
        door = entrance_door_cell(hse.origin, hse.line, hse.quarter_turns, hse.flipped)
        if 0 <= door[0] < w and 0 <= door[1] < h:
            _paint(img, np.array([door[0]], np.int32), np.array([door[1]], np.int32), _DOOR_RGB)

    xs, ys = _cells_xy(step.highlight_footprint, w, h)
    _paint(img, xs, ys, _HIGHLIGHT_RGB)

    xs, ys = _cells_xy(step.highlight_path_route, w, h)
    _paint(img, xs, ys, _ROUTE_RGB)

    # Semi-transparent frontier markers (opaque white looked like broken holes).
    highlight = step.highlight_footprint
    frontier_cells = [c for c in step.frontier if c not in highlight]
    xs, ys = _cells_xy(frontier_cells, w, h)
    _blend(img, xs, ys, _FRONTIER_RGB, _FRONTIER_ALPHA)

    if try_diag is not None:
        rgb = _TRY_OK_RGB if try_diag.ok else _TRY_BAD_RGB
        xs, ys = _cells_xy(try_diag.footprint, w, h)
        _paint(img, xs, ys, rgb)
        if try_diag.door_cell:
            d = try_diag.door_cell
            if 0 <= d[0] < w and 0 <= d[1] < h:
                _paint(img, np.array([d[0]], np.int32), np.array([d[1]], np.int32), _DOOR_RGB)
        xs, ys = _cells_xy(try_diag.route, w, h)
        _paint(img, xs, ys, _ROUTE_RGB)

    return img


def _side_panel_text(
    step: SolveStep,
    *,
    try_diag: Optional[PlacementDiagnosis] = None,
) -> Optional[str]:
    if try_diag is not None:
        return _format_diagnosis(try_diag)
    if not step.top_candidates:
        return None
    lines = ["Top scores:"]
    for cand in step.top_candidates[:5]:
        mark = "→" if step.selected and cand.key() == step.selected.key() else " "
        lines.append(
            f"{mark} {cand.score:4d}  {cand.label}@{cand.origin} qt{cand.quarter_turns}"
        )
    return "\n".join(lines)


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


def _setup_axes(ax, grid: WorldGrid) -> None:
    ax.set_facecolor(COLOR_BG)
    ax.set_aspect("equal")
    margin = 2
    ax.set_xlim(-margin, grid.width + margin)
    ax.set_ylim(grid.height + margin, -margin)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#444466")
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
    """Draw one solver frame (numpy raster, used for GIF export)."""
    ax.clear()
    _setup_axes(ax, grid)
    rgba = build_step_rgba(grid, step, try_diag=try_diag)
    ax.imshow(
        rgba,
        origin="upper",
        extent=(0, grid.width, grid.height, 0),
        interpolation="nearest",
        zorder=0,
    )
    title = f"Step {step_idx + 1}/{total}: [{step.kind}] {step.title}"
    if try_mode:
        title += "  [TRY MODE — click grid]"
    ax.set_title(title, color="white", fontsize=11, pad=8)
    panel = _side_panel_text(step, try_diag=try_diag)
    show_detail = step.detail and not step.top_candidates and (
        try_mode or not panel
    )
    if show_detail:
        ax.text(
            0.01,
            0.14,
            step.detail,
            transform=ax.transAxes,
            color="#aaaacc",
            fontsize=9,
            verticalalignment="bottom",
        )
    if panel:
        ax.text(
            1.01,
            0.98,
            panel,
            transform=ax.transAxes,
            color="#ccccdd",
            fontsize=8,
            verticalalignment="top",
            family="monospace",
        )


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
        self._im = None
        self._detail_text = None
        self._side_text = None
        self._axes_ready = False
        self._auto_ms = 250
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
        step = self._current_step()
        if not self._axes_ready:
            self.ax.clear()
            _setup_axes(self.ax, self.grid)
            blank = np.zeros((self.grid.height, self.grid.width, 4), dtype=np.uint8)
            blank[..., :3] = _BG_RGB
            blank[..., 3] = 255
            self._im = self.ax.imshow(
                blank,
                origin="upper",
                extent=(0, self.grid.width, self.grid.height, 0),
                interpolation="nearest",
                zorder=0,
            )
            self._axes_ready = True

        rgba = build_step_rgba(
            self.grid,
            step,
            try_diag=self.try_diag if self.try_mode else None,
        )
        self._im.set_data(rgba)

        title = f"Step {self.idx + 1}/{len(self.steps)}: [{step.kind}] {step.title}"
        if self.try_mode:
            title += "  [TRY MODE — click grid]"
        self.ax.set_title(title, color="white", fontsize=11, pad=8)

        if self._detail_text is not None:
            self._detail_text.remove()
            self._detail_text = None
        if self._side_text is not None:
            self._side_text.remove()
            self._side_text = None
        panel = _side_panel_text(
            step, try_diag=self.try_diag if self.try_mode else None
        )
        show_detail = step.detail and not step.top_candidates and (
            self.try_mode or not panel
        )
        if show_detail:
            self._detail_text = self.ax.text(
                0.01,
                0.14,
                step.detail,
                transform=self.ax.transAxes,
                color="#aaaacc",
                fontsize=9,
                verticalalignment="bottom",
            )
        if panel:
            self._side_text = self.ax.text(
                1.01,
                0.98,
                panel,
                transform=self.ax.transAxes,
                color="#ccccdd",
                fontsize=8,
                verticalalignment="top",
                family="monospace",
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
            self.fig.canvas.new_timer(interval=self._auto_ms).single_shot(
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
    solve(_build_grid(args), record_steps=False)
    result = solve(grid, record_steps=True, animation=True)
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
