"""RayLib visualiser for a TOML-described gridworld and its label primitives.

Usage::

    python -m msc_skill_machines.visualiser environments/office.toml [--cell-size 48]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import pyray as rl

from msc_skill_machines.gridworld_builder import build_primitives, load_gridworld
from msc_skill_machines.render import (
    GridLayout,
    draw_grid,
    draw_grid_lines,
    draw_heatmap,
    draw_heatmap_stack,
    draw_labels,
    draw_mask,
)

PANEL_W = 260
MARGIN = 20
ROW_H = 26
STACK_GAP = 12
STACK_MIN_H = 140

MASK_OPACITY = 0.7
INITIAL_RANGE = (rl.WHITE, rl.BLUE)
REWARD_RANGE = (rl.WHITE, rl.ORANGE)


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Visualise a TOML gridworld and its primitives")
    parser.add_argument("toml_path", help="path to the gridworld .toml file")
    parser.add_argument("--cell-size", type=int, default=48, help="pixel size of one grid cell")
    parser.add_argument("--frames", type=int, default=0, help="render this many frames then exit (0 = run until closed)")
    parser.add_argument("--screenshot", default=None, help="save a screenshot to this path before exiting (with --frames)")
    parser.add_argument("--all-on", action="store_true", help="start with every overlay enabled and the first primitive selected")
    args = parser.parse_args(argv)

    spec, env = load_gridworld(args.toml_path)
    primitives = build_primitives(env, seed=spec.seed)
    reward_maps = {name: prim.reward_map() for name, prim in primitives.items()}
    action_titles = next(iter(primitives.values())).action_labels() if primitives else []

    rows, cols = env.barrier_mask.shape
    grid = GridLayout(MARGIN, MARGIN, args.cell_size, rows, cols)
    n_layers = len(action_titles)
    stack_w = n_layers * cols * args.cell_size + (n_layers - 1) * STACK_GAP if n_layers else 0
    win_w = max(grid.width + PANEL_W + 3 * MARGIN, min(stack_w, 1400) + PANEL_W + 3 * MARGIN)
    win_h = grid.height + 3 * MARGIN + STACK_MIN_H
    win_h = max(win_h, MARGIN * 2 + ROW_H * (10 + len(primitives)))

    toggles = Toggles()
    if args.all_on:
        toggles = Toggles(
            barrier=True, absorbing=True, initial=True, goal=True, labels=True,
            primitive=next(iter(primitives), None), reward_map=True, target_goals=True,
        )

    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    rl.init_window(win_w, win_h, f"gridworld: {spec.name}")
    rl.set_target_fps(60)

    frame = 0
    try:
        while not rl.window_should_close():
            if args.frames and frame >= args.frames:
                if args.screenshot:
                    rl.take_screenshot(args.screenshot)
                break
            frame += 1
            rl.begin_drawing()
            rl.clear_background(rl.RAYWHITE)

            # ---- main grid + composable overlays -------------------------------
            draw_grid(grid)
            if toggles.initial:
                draw_heatmap(grid, env.initial_state_distribution, INITIAL_RANGE, vmin=0.0)
            if toggles.barrier:
                draw_mask(grid, env.barrier_mask, rl.BLACK, MASK_OPACITY)
            if toggles.absorbing:
                draw_mask(grid, env.absorbing_mask, rl.RED, MASK_OPACITY)
            if toggles.goal:
                draw_mask(grid, env.goal_mask, rl.GREEN, MASK_OPACITY)
            prim = primitives.get(toggles.primitive) if toggles.primitive else None
            if prim is not None and toggles.target_goals:
                draw_mask(grid, prim.target_goal_mask(), rl.GREEN, MASK_OPACITY)
            draw_grid_lines(grid)
            if toggles.labels:
                # only proposition symbols; con_X columns duplicate X and would just add noise
                draw_labels(grid, env.label_mask[:, :, :len(env.prop_labels)], env.prop_labels)

            # hover readout
            mouse = rl.get_mouse_position()
            cell = grid.cell_at(int(mouse.x), int(mouse.y))
            if cell is not None:
                r, c = cell
                labels = ",".join(sorted(env.label_assignment_to_set(env.label_mask[r, c]))) or "-"
                rl.draw_text(
                    f"({r},{c}) p0={env.initial_state_distribution[r, c]:.3f} labels={labels}",
                    MARGIN, grid.origin_y + grid.height + 4, 10, rl.DARKGRAY,
                )

            # ---- reward-map stack below the main grid ----------------------------
            if prim is not None and toggles.reward_map:
                stack_origin = GridLayout(MARGIN, grid.origin_y + grid.height + MARGIN, args.cell_size, rows, cols)
                avail_w = rl.get_screen_width() - PANEL_W - 3 * MARGIN
                rl.draw_text(f"reward map: {prim.target_label}", MARGIN, stack_origin.origin_y - 14, 12, rl.DARKGRAY)
                draw_heatmap_stack(
                    stack_origin, reward_maps[prim.target_label], REWARD_RANGE,
                    gap_px=STACK_GAP, titles=action_titles, max_width=avail_w, vmin=0.0, vmax=1.0,
                )

            # ---- side panel ---------------------------------------------------------
            panel_x = rl.get_screen_width() - PANEL_W - MARGIN
            rl.draw_rectangle(panel_x - 10, 0, PANEL_W + 10 + MARGIN, rl.get_screen_height(), rl.Color(240, 240, 240, 255))
            ui = Checklist(panel_x, MARGIN)
            ui.heading(f"{spec.name}  ({rows}x{cols})")
            ui.heading("environment")
            toggles.barrier = ui.checkbox("barrier_mask (black)", toggles.barrier)
            toggles.absorbing = ui.checkbox("absorbing_mask (red)", toggles.absorbing)
            toggles.initial = ui.checkbox("initial_state_distribution", toggles.initial)
            toggles.goal = ui.checkbox("goal_mask (green)", toggles.goal)
            toggles.labels = ui.checkbox("label_mask (symbols)", toggles.labels)

            ui.gap()
            ui.heading("primitive (select one)")
            for name in primitives:
                selected = ui.checkbox(name, toggles.primitive == name)
                if selected and toggles.primitive != name:
                    toggles.primitive = name
                elif not selected and toggles.primitive == name:
                    toggles.primitive = None
            ui.gap()
            toggles.reward_map = ui.checkbox("reward map", toggles.reward_map)
            toggles.target_goals = ui.checkbox("target goals (green)", toggles.target_goals)

            rl.end_drawing()
    finally:
        rl.close_window()


if __name__ == "__main__":
    main()
