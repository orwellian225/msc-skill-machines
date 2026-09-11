# Skill Machines

An attempt to recreate [Skill Machines](10.48550/arXiv.2205.12532) without the LTL component.

This is my trying to understand it.

## Visualiser

Gridworlds are described in TOML (see `environments/office.toml` and
`environments/corridor.toml`). Run the RayLib visualiser on one with:

```
uv run msc-visualise environments/office.toml
uv run msc-visualise environments/corridor.toml --cell-size 40
```

The checklist on the right toggles each overlay; they compose, so a heatmap and a mask can be
shown at once:

- `barrier_mask` (black, 70%), `absorbing_mask` (red, 70%), `goal_mask` (green, 70%)
- `initial_state_distribution` as a white-to-blue heatmap
- `label_mask` as the proposition symbols written into each cell
- a primitive picker, then its `reward map` (one heatmap per move x stay/terminate action,
  side by side) and `target goals` (green mask of the target label's cells)

Hovering a cell prints its coordinates, initial probability and labels under the grid.

### Screenshots instead of a window

Pass `--screenshot PATH` to render a single frame to a PNG (hidden window, no side panel) and
exit. The renderer options choose what is drawn; in window mode they set the initial toggles:

```
uv run msc-visualise environments/corridor.toml --screenshot out.png \
    --show all --primitive A --reward-map --target-goals
```

- `--show OVERLAY...` any of `barrier absorbing initial goal labels`, or `all`
  (default: `barrier absorbing labels`)
- `--primitive LABEL` selects a primitive; `--reward-map` and `--target-goals` draw its views

### TOML schema

```toml
name = "office"
seed = 1                # seeds random label placement and the env reset
step_limit = 100        # optional

layout = """
#######
#A....#
#.(S,X,A).#     # a bracket group puts several symbols in one cell
#S...B#
#######
"""

[[labels]]
identifier = "A"
constraint = true       # also creates a con_A label and primitive
random = true           # add num_states A cells on random non-barrier cells
num_states = 4

[[labels]]
identifier = "B"
constraint = false
random = false          # only where the map says
```

Reserved symbols: `#` barrier, `X` absorbing, `S` initial state (uniform over all `S` cells,
or over all free cells if there is none), `.` free. Every other symbol is a label identifier.
Random labels are sampled independently, so they may overlap each other or fixed cells.

Full details are in the docstring of `src/msc_skill_machines/gridworld_builder.py`.
