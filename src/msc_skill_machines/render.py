"""Reusable pyray drawing helpers for gridworld tensors.

Every function only *draws*; the caller owns ``begin_drawing`` / ``end_drawing``. Because
they just paint onto the current frame in order, overlays compose: draw a heatmap, then a
mask, then labels, and each one sits on top of the previous.

Cell ``(row, col)`` maps to pixels via :class:`GridLayout`, row 0 at the top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pyray as rl

if TYPE_CHECKING:
    from msc_skill_machines.primitives import GridworldTaskPrimitive

Colour = rl.Color
ColourRange = tuple[Colour, Colour]


@dataclass(frozen=True)
class GridLayout:
    origin_x: int
    origin_y: int
    cell_px: int
    rows: int
    cols: int

    @property
    def width(self) -> int:
        return self.cols * self.cell_px

    @property
    def height(self) -> int:
        return self.rows * self.cell_px

    def cell_rect(self, row: int, col: int) -> tuple[int, int, int, int]:
        """``(x, y, w, h)`` in pixels for a cell."""
        return (
            self.origin_x + col * self.cell_px,
            self.origin_y + row * self.cell_px,
            self.cell_px,
            self.cell_px,
        )

    def cell_at(self, px: int, py: int) -> tuple[int, int] | None:
        """Inverse mapping: pixel -> ``(row, col)`` or ``None`` if outside the grid."""
        col = (px - self.origin_x) // self.cell_px
        row = (py - self.origin_y) // self.cell_px
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return int(row), int(col)
        return None


def _rgba(colour) -> tuple[int, int, int, int]:
    """pyray colour constants are plain tuples; ``rl.Color`` structs have .r/.g/.b/.a."""
    if hasattr(colour, "r"):
        return int(colour.r), int(colour.g), int(colour.b), int(colour.a)
    r, g, b, a = (tuple(colour) + (255,))[:4]
    return int(r), int(g), int(b), int(a)


def lerp_colour(low: Colour, high: Colour, t: float) -> Colour:
    t = float(min(1.0, max(0.0, t)))
    lo, hi = _rgba(low), _rgba(high)
    return rl.Color(*(int(round(a + (b - a) * t)) for a, b in zip(lo, hi)))


def draw_grid(layout: GridLayout, line_colour: Colour = rl.BLACK, fill: Colour = rl.WHITE) -> None:
    """Base black-and-white grid."""
    rl.draw_rectangle(layout.origin_x, layout.origin_y, layout.width, layout.height, fill)
    for row in range(layout.rows + 1):
        y = layout.origin_y + row * layout.cell_px
        rl.draw_line(layout.origin_x, y, layout.origin_x + layout.width, y, line_colour)
    for col in range(layout.cols + 1):
        x = layout.origin_x + col * layout.cell_px
        rl.draw_line(x, layout.origin_y, x, layout.origin_y + layout.height, line_colour)


def draw_mask(layout: GridLayout, mask: npt.NDArray[np.bool_], colour: Colour, opacity: float) -> None:
    """Fill every ``True`` cell of a 2D mask with ``colour`` at ``opacity`` (0..1)."""
    mask = np.asarray(mask, dtype=np.bool_)
    _check_shape(layout, mask, "mask")
    tinted = rl.fade(colour, float(opacity))
    for row, col in np.argwhere(mask):
        x, y, w, h = layout.cell_rect(int(row), int(col))
        rl.draw_rectangle(x, y, w, h, tinted)


def draw_heatmap(
    layout: GridLayout,
    values: npt.NDArray[np.floating],
    colour_range: ColourRange,
    vmin: float | None = None,
    vmax: float | None = None,
    opacity: float = 1.0,
) -> None:
    """Colour every cell of a 2D float tensor by linearly interpolating ``colour_range``.

    ``vmin``/``vmax`` default to the tensor's min/max; a constant tensor renders as the low colour.
    """
    values = np.asarray(values, dtype=np.float64)
    _check_shape(layout, values, "values")
    low, high = colour_range
    lo = float(np.nanmin(values)) if vmin is None else float(vmin)
    hi = float(np.nanmax(values)) if vmax is None else float(vmax)
    span = hi - lo
    for row in range(layout.rows):
        for col in range(layout.cols):
            v = values[row, col]
            t = 0.0 if span <= 0 or np.isnan(v) else (v - lo) / span
            x, y, w, h = layout.cell_rect(row, col)
            rl.draw_rectangle(x, y, w, h, rl.fade(lerp_colour(low, high, t), float(opacity)))


def draw_heatmap_stack(
    layout: GridLayout,
    values: npt.NDArray[np.floating],
    colour_range: ColourRange,
    gap_px: int = 16,
    titles: list[str] | None = None,
    max_width: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    draw_base_grid: bool = True,
    title_colour: Colour = rl.DARKGRAY,
) -> list[GridLayout]:
    """Render each layer ``values[:, :, k]`` of a 3D tensor as its own 2D heatmap, side by side.

    Layers start at ``layout.origin`` and use ``layout.cell_px`` unless the whole row would exceed
    ``max_width``, in which case the cell size is shrunk to fit. All layers share one colour scale
    (``vmin``/``vmax`` default to the global min/max) so they are comparable.

    Returns one :class:`GridLayout` per layer so callers can draw further overlays on them.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected a 3D tensor, got shape {values.shape}")
    rows, cols, depth = values.shape
    if (rows, cols) != (layout.rows, layout.cols):
        raise ValueError(f"tensor layers are {rows}x{cols} but layout is {layout.rows}x{layout.cols}")

    cell_px = layout.cell_px
    if max_width is not None and depth > 0:
        # depth grids + (depth-1) gaps must fit
        cell_px = max(4, min(cell_px, (max_width - gap_px * (depth - 1)) // (depth * cols)))

    lo = float(np.nanmin(values)) if vmin is None else float(vmin)
    hi = float(np.nanmax(values)) if vmax is None else float(vmax)
    title_h = 14 if titles else 0

    layouts: list[GridLayout] = []
    x = layout.origin_x
    for k in range(depth):
        sub = GridLayout(x, layout.origin_y + title_h, cell_px, rows, cols)
        if draw_base_grid:
            draw_grid(sub)
        draw_heatmap(sub, values[:, :, k], colour_range, vmin=lo, vmax=hi)
        if draw_base_grid:
            draw_grid_lines(sub)
        if titles:
            rl.draw_text(titles[k] if k < len(titles) else str(k), x, layout.origin_y, 10, title_colour)
        layouts.append(sub)
        x += sub.width + gap_px
    return layouts


# --------------------------------------------------------------------------- #
# Primitive-specific rendering
# --------------------------------------------------------------------------- #

PRIMITIVE_TITLE_H = 14
PRIMITIVE_ROW_TITLES = ("no-term", "term")


@dataclass(frozen=True)
class PrimitiveMapGeometry:
    """Pixel geometry of a primitive reward map: ``n_actions`` columns x 2 rows of grids."""
    cell_px: int
    n_actions: int
    rows: int
    cols: int
    gap_px: int

    @property
    def grid_w(self) -> int:
        return self.cols * self.cell_px

    @property
    def grid_h(self) -> int:
        return self.rows * self.cell_px

    @property
    def column_pitch(self) -> int:
        return self.grid_w + self.gap_px

    @property
    def row_pitch(self) -> int:
        return PRIMITIVE_TITLE_H + self.grid_h + self.gap_px

    @property
    def width(self) -> int:
        return self.n_actions * self.column_pitch - self.gap_px

    @property
    def height(self) -> int:
        return len(PRIMITIVE_ROW_TITLES) * self.row_pitch - self.gap_px


def primitive_reward_map_geometry(
    layout: GridLayout, n_actions: int, gap_px: int = 12, max_width: int | None = None,
) -> PrimitiveMapGeometry:
    """Size of a primitive reward map drawn from ``layout``; cells shrink to fit ``max_width``."""
    cell_px = layout.cell_px
    if max_width is not None and n_actions > 0:
        cell_px = max(4, min(cell_px, (max_width - gap_px * (n_actions - 1)) // (n_actions * layout.cols)))
    return PrimitiveMapGeometry(cell_px, n_actions, layout.rows, layout.cols, gap_px)


def draw_primitive_reward_map(
    layout: GridLayout,
    primitive: GridworldTaskPrimitive,
    colour_range: ColourRange = (rl.WHITE, rl.ORANGE),
    gap_px: int = 12,
    max_width: int | None = None,
    reward_map: npt.NDArray[np.floating] | None = None,
    title_colour: Colour = rl.DARKGRAY,
) -> list[list[GridLayout]]:
    """Render a primitive's reward function as a grid of heatmaps.

    One column per env action, two rows: no-terminate on top, terminate below::

        a0 no-term   a1 no-term   a2 no-term   a3 no-term
        a0 term      a1 term      a2 term      a3 term

    Every primitive's action is ``(env_action, terminate_action)``, and
    :meth:`GridworldTaskPrimitive.reward_map` stores layer ``2 * a + t`` for action ``a`` and
    terminate flag ``t``, which is what this reads. Pass ``reward_map`` to reuse a precomputed
    tensor. All grids share the colour scale ``[0, 1]``.

    Returns ``layouts[row][column]`` so callers can overlay more on any grid.
    """
    values = np.asarray(primitive.reward_map() if reward_map is None else reward_map, dtype=np.float64)
    n_actions = len(primitive.env.transitions)
    if values.shape != (layout.rows, layout.cols, 2 * n_actions):
        raise ValueError(
            f"reward map has shape {values.shape}, expected ({layout.rows}, {layout.cols}, {2 * n_actions})"
        )
    titles = primitive.action_labels()
    geom = primitive_reward_map_geometry(layout, n_actions, gap_px, max_width)

    layouts: list[list[GridLayout]] = []
    for t, _row_name in enumerate(PRIMITIVE_ROW_TITLES):
        row_layouts: list[GridLayout] = []
        y = layout.origin_y + t * geom.row_pitch
        for a in range(n_actions):
            x = layout.origin_x + a * geom.column_pitch
            sub = GridLayout(x, y + PRIMITIVE_TITLE_H, geom.cell_px, layout.rows, layout.cols)
            draw_grid(sub)
            draw_heatmap(sub, values[:, :, 2 * a + t], colour_range, vmin=0.0, vmax=1.0)
            draw_grid_lines(sub)
            rl.draw_text(titles[2 * a + t], x, y, 10, title_colour)
            row_layouts.append(sub)
        layouts.append(row_layouts)
    return layouts


def draw_grid_lines(layout: GridLayout, line_colour: Colour = rl.BLACK) -> None:
    """Just the grid lines (no fill), for re-drawing borders on top of filled cells."""
    for row in range(layout.rows + 1):
        y = layout.origin_y + row * layout.cell_px
        rl.draw_line(layout.origin_x, y, layout.origin_x + layout.width, y, line_colour)
    for col in range(layout.cols + 1):
        x = layout.origin_x + col * layout.cell_px
        rl.draw_line(x, layout.origin_y, x, layout.origin_y + layout.height, line_colour)


def draw_labels(
    layout: GridLayout,
    label_mask: npt.NDArray[np.bool_],
    labels: list[str],
    colour: Colour = rl.BLACK,
) -> None:
    """Write the symbol of every ``True`` bit of the assignment vector into its cell.

    ``label_mask`` is ``(rows, cols, len(labels))``; a cell with several true bits shows the
    symbols joined (``"AB"``), with the font shrunk so the text fits inside the cell.
    """
    label_mask = np.asarray(label_mask, dtype=np.bool_)
    if label_mask.ndim != 3 or label_mask.shape[:2] != (layout.rows, layout.cols):
        raise ValueError(f"label_mask must be ({layout.rows}, {layout.cols}, n), got {label_mask.shape}")
    if label_mask.shape[2] > len(labels):
        raise ValueError(f"label_mask has {label_mask.shape[2]} bits but only {len(labels)} labels were given")

    padding = max(2, layout.cell_px // 8)
    inner = layout.cell_px - 2 * padding
    for row, col in np.argwhere(label_mask.any(axis=2)):
        bits = label_mask[row, col]
        text = "".join(labels[i] for i in np.flatnonzero(bits))
        font_px = _fit_font(text, inner)
        text_w = rl.measure_text(text, font_px)
        x, y, w, h = layout.cell_rect(int(row), int(col))
        rl.draw_text(text, x + (w - text_w) // 2, y + (h - font_px) // 2, font_px, colour)


def _fit_font(text: str, max_px: int) -> int:
    font_px = max(6, max_px)
    while font_px > 6 and rl.measure_text(text, font_px) > max_px:
        font_px -= 1
    return font_px


def _check_shape(layout: GridLayout, arr: np.ndarray, name: str) -> None:
    if arr.shape != (layout.rows, layout.cols):
        raise ValueError(f"{name} has shape {arr.shape} but layout is ({layout.rows}, {layout.cols})")
