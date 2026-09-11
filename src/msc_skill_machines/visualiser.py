"""RayLib visualiser for a TOML-described gridworld and its label primitives.

Usage::

    python -m msc_skill_machines.visualiser environments/office.toml [--cell-size 48]
    python -m msc_skill_machines.visualiser environments/office.toml --screenshot out.png \
        --show all --primitive A --reward-map --target-goals
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pyray as rl

from msc_skill_machines.gridworld_builder import build_primitives, load_gridworld
from msc_skill_machines.render import (
    GridLayout,
    draw_grid,
    draw_grid_lines,
    draw_heatmap,
    draw_labels,
    draw_mask,
    draw_primitive_reward_map,
    primitive_reward_map_geometry,
)

PANEL_W = 260
MARGIN = 20
ROW_H = 26
STACK_GAP = 12
STACK_TITLE_H = 14
MAX_STACK_W = 1400
WINDOW_INSET = 40   # minimum gap between the window and the monitor edges

MASK_OPACITY = 0.7
INITIAL_RANGE = (rl.WHITE, rl.BLUE)
REWARD_RANGE = (rl.WHITE, rl.ORANGE)

OVERLAYS = ("barrier", "absorbing", "initial", "goal", "labels")
DEFAULT_OVERLAYS = ("barrier", "absorbing", "labels")


@dataclass
class Toggles:
    barrier: bool = True
    absorbing: bool = True
    initial: bool = False
    goal: bool = False
    labels: bool = True
    primitive: str | None = None
    reward_map: bool = False
    target_goals: bool = False


@dataclass
class Checklist:
    """Tiny helper that lays raygui checkboxes out top-to-bottom in the side panel."""
    x: int
    y: int
    cursor: int = field(init=False)

    def __post_init__(self) -> None:
        self.cursor = self.y

    def heading(self, text: str) -> None:
        rl.draw_text(text, self.x, self.cursor + 4, 12, rl.DARKGRAY)
        self.cursor += ROW_H

    def checkbox(self, text: str, value: bool) -> bool:
        box = rl.Rectangle(self.x, self.cursor, 18, 18)
        # raygui's check box takes a pointer to the bool and returns whether it changed
        ptr = rl.ffi.new("bool *", value)
        rl.gui_check_box(box, text, ptr)
        self.cursor += ROW_H
        return bool(ptr[0])

    def gap(self, px: int = 8) -> None:
        self.cursor += px


@dataclass
class Scene:
    """Everything needed to draw one frame; shared by the window loop and screenshot mode."""
    spec: object
    env: object
    primitives: dict
    reward_maps: dict
    grid: GridLayout
    cell_size: int

    @property
    def rows(self) -> int:
        return self.grid.rows

    @property
    def cols(self) -> int:
        return self.grid.cols

    @property
    def n_actions(self) -> int:
        return len(self.env.transitions)

    def stack_origin(self) -> GridLayout:
        return GridLayout(MARGIN, self.grid.origin_y + self.grid.height + MARGIN + STACK_TITLE_H, self.cell_size, self.rows, self.cols)

    def stack_width(self) -> int:
        """Unshrunk width of the reward map (n_actions columns of full-size grids)."""
        return primitive_reward_map_geometry(self.stack_origin(), self.n_actions, STACK_GAP).width

    def stack_height(self, content_width: int, content_height: int | None = None) -> int:
        """Height of the reward map (2 rows) once shrunk to fit the content box, plus its heading."""
        max_h = None if content_height is None else content_height - STACK_TITLE_H
        geom = primitive_reward_map_geometry(self.stack_origin(), self.n_actions, STACK_GAP, content_width, max_h)
        return STACK_TITLE_H + geom.height


def load_scene(toml_path: str, cell_size: int) -> Scene:
    spec, env = load_gridworld(toml_path)
    primitives = build_primitives(env, seed=spec.seed)
    reward_maps = {name: prim.reward_map() for name, prim in primitives.items()}
    rows, cols = env.barrier_mask.shape
    grid = GridLayout(MARGIN, MARGIN, cell_size, rows, cols)
    return Scene(spec, env, primitives, reward_maps, grid, cell_size)


def draw_scene(
    scene: Scene, toggles: Toggles, content_width: int, content_height: int | None = None, draw_hover: bool = True,
) -> None:
    """Draw the grid, its enabled overlays and (if selected) the reward-map stack.

    ``content_width`` / ``content_height`` bound the space below the grid for the reward map;
    its grids shrink to fit.
    """
    env, grid = scene.env, scene.grid
    draw_grid(grid)
    if toggles.initial:
        draw_heatmap(grid, env.initial_state_distribution, INITIAL_RANGE, vmin=0.0)
    if toggles.barrier:
        draw_mask(grid, env.barrier_mask, rl.BLACK, MASK_OPACITY)
    if toggles.absorbing:
        draw_mask(grid, env.absorbing_mask, rl.RED, MASK_OPACITY)
    if toggles.goal:
        draw_mask(grid, env.goal_mask, rl.GREEN, MASK_OPACITY)
    prim = scene.primitives.get(toggles.primitive) if toggles.primitive else None
    if prim is not None and toggles.target_goals:
        draw_mask(grid, prim.target_goal_mask(), rl.GREEN, MASK_OPACITY)
    draw_grid_lines(grid)
    if toggles.labels:
        # only proposition symbols; con_X columns duplicate X and would just add noise
        draw_labels(grid, env.label_mask[:, :, :len(env.prop_labels)], env.prop_labels)

    if draw_hover:
        mouse = rl.get_mouse_position()
        cell = grid.cell_at(int(mouse.x), int(mouse.y))
        if cell is not None:
            r, c = cell
            labels = ",".join(sorted(env.label_assignment_to_set(env.label_mask[r, c]))) or "-"
            rl.draw_text(
                f"({r},{c}) p0={env.initial_state_distribution[r, c]:.3f} labels={labels}",
                MARGIN, grid.origin_y + grid.height + 4, 10, rl.DARKGRAY,
            )

    if prim is not None and toggles.reward_map:
        stack = scene.stack_origin()
        rl.draw_text(f"reward map: {prim.target_label}", MARGIN, stack.origin_y - STACK_TITLE_H, 12, rl.DARKGRAY)
        max_h = None if content_height is None else content_height - STACK_TITLE_H
        draw_primitive_reward_map(
            stack, prim, REWARD_RANGE, gap_px=STACK_GAP, max_width=content_width, max_height=max_h,
            reward_map=scene.reward_maps[prim.target_label],
        )


def draw_panel(scene: Scene, toggles: Toggles) -> None:
    """Right-hand raygui checklist; mutates ``toggles`` in place."""
    panel_x = rl.get_screen_width() - PANEL_W - MARGIN
    rl.draw_rectangle(panel_x - 10, 0, PANEL_W + 10 + MARGIN, rl.get_screen_height(), rl.Color(240, 240, 240, 255))
    ui = Checklist(panel_x, MARGIN)
    ui.heading(f"{scene.spec.name}  ({scene.rows}x{scene.cols})")
    ui.heading("environment")
    toggles.barrier = ui.checkbox("barrier_mask (black)", toggles.barrier)
    toggles.absorbing = ui.checkbox("absorbing_mask (red)", toggles.absorbing)
    toggles.initial = ui.checkbox("initial_state_distribution", toggles.initial)
    toggles.goal = ui.checkbox("goal_mask (green)", toggles.goal)
    toggles.labels = ui.checkbox("label_mask (symbols)", toggles.labels)

    ui.gap()
    ui.heading("primitive (select one)")
    for name in scene.primitives:
        selected = ui.checkbox(name, toggles.primitive == name)
        if selected and toggles.primitive != name:
            toggles.primitive = name
        elif not selected and toggles.primitive == name:
            toggles.primitive = None
    ui.gap()
    toggles.reward_map = ui.checkbox("reward map", toggles.reward_map)
    toggles.target_goals = ui.checkbox("target goals (green)", toggles.target_goals)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualise a TOML gridworld and its primitives")
    parser.add_argument("toml_path", help="path to the gridworld .toml file")
    parser.add_argument("--cell-size", type=int, default=48, help="pixel size of one grid cell")
    parser.add_argument(
        "--screenshot", metavar="PATH", default=None,
        help="render one frame to this PNG with a hidden window and exit, instead of opening the window",
    )
    render = parser.add_argument_group("renderer options", "which overlays are drawn (initial state in window mode)")
    render.add_argument(
        "--show", nargs="+", choices=(*OVERLAYS, "all"), default=list(DEFAULT_OVERLAYS), metavar="OVERLAY",
        help=f"overlays to enable: {', '.join(OVERLAYS)} or all (default: {' '.join(DEFAULT_OVERLAYS)})",
    )
    render.add_argument("--primitive", metavar="LABEL", default=None, help="select this primitive (a prop or con_* label)")
    render.add_argument("--reward-map", action="store_true", help="show the selected primitive's reward map")
    render.add_argument("--target-goals", action="store_true", help="show the selected primitive's target goals")
    return parser


def toggles_from_args(args: argparse.Namespace, scene: Scene, parser: argparse.ArgumentParser) -> Toggles:
    show = set(OVERLAYS) if "all" in args.show else set(args.show)
    if args.primitive is not None and args.primitive not in scene.primitives:
        parser.error(f"--primitive {args.primitive!r} is not one of {list(scene.primitives)}")
    if (args.reward_map or args.target_goals) and args.primitive is None:
        parser.error("--reward-map / --target-goals require --primitive")
    return Toggles(
        barrier="barrier" in show, absorbing="absorbing" in show, initial="initial" in show,
        goal="goal" in show, labels="labels" in show,
        primitive=args.primitive, reward_map=args.reward_map, target_goals=args.target_goals,
    )


def run_window(scene: Scene, toggles: Toggles) -> None:
    content_w = max(scene.grid.width, min(scene.stack_width(), MAX_STACK_W))
    win_w = content_w + PANEL_W + 3 * MARGIN
    win_h = scene.grid.height + 3 * MARGIN + scene.stack_height(content_w)
    win_h = max(win_h, MARGIN * 2 + ROW_H * (10 + len(scene.primitives)))

    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    rl.init_window(win_w, win_h, f"gridworld: {scene.spec.name}")
    rl.set_target_fps(60)
    try:
        # Keep the window on screen: a window taller/wider than the monitor is placed with its
        # title bar off-screen on Windows and looks like it never opened.
        monitor = rl.get_current_monitor()
        max_w, max_h = rl.get_monitor_width(monitor) - 2 * WINDOW_INSET, rl.get_monitor_height(monitor) - 2 * WINDOW_INSET
        if win_w > max_w or win_h > max_h:
            rl.set_window_size(min(win_w, max_w), min(win_h, max_h))
            rl.set_window_position(WINDOW_INSET, WINDOW_INSET)

        while not rl.window_should_close():
            rl.begin_drawing()
            rl.clear_background(rl.RAYWHITE)
            draw_scene(
                scene, toggles,
                content_width=rl.get_screen_width() - PANEL_W - 3 * MARGIN,
                content_height=rl.get_screen_height() - scene.grid.height - 3 * MARGIN,
            )
            draw_panel(scene, toggles)
            rl.end_drawing()
    finally:
        rl.close_window()


def save_screenshot(scene: Scene, toggles: Toggles, path: Path) -> Path:
    """Render one frame in a hidden window sized to the content and write it to ``path``."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    show_stack = bool(toggles.primitive and toggles.reward_map)
    content_w = max(scene.grid.width, min(scene.stack_width(), MAX_STACK_W) if show_stack else 0)
    win_w = content_w + 2 * MARGIN
    win_h = scene.grid.height + 2 * MARGIN
    if show_stack:
        win_h += MARGIN + scene.stack_height(content_w) + MARGIN

    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)
    rl.init_window(win_w, win_h, f"gridworld: {scene.spec.name}")
    try:
        # A hidden window's framebuffer reads back black, so draw into an offscreen render
        # texture instead. (take_screenshot() also prepends the cwd to its path, so we export
        # the image ourselves.)
        target = rl.load_render_texture(win_w, win_h)
        try:
            rl.begin_texture_mode(target)
            rl.clear_background(rl.RAYWHITE)
            draw_scene(scene, toggles, content_width=content_w, draw_hover=False)
            rl.end_texture_mode()
            image = rl.load_image_from_texture(target.texture)
            try:
                rl.image_flip_vertical(image)  # render textures are stored bottom-up
                rl.export_image(image, str(path))
            finally:
                rl.unload_image(image)
        finally:
            rl.unload_render_texture(target)
    finally:
        rl.close_window()

    if not path.exists():
        raise RuntimeError(f"raylib did not write {path}")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    scene = load_scene(args.toml_path, args.cell_size)
    toggles = toggles_from_args(args, scene, parser)

    if args.screenshot:
        out = save_screenshot(scene, toggles, Path(args.screenshot))
        print(f"wrote {out}")
    else:
        run_window(scene, toggles)


if __name__ == "__main__":
    main()
