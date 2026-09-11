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

### TOML schema

```toml
name = "office"
props = ["A", "B"]      # proposition labels, in order
cons  = ["B"]           # subset of props that are also constraints (-> con_B primitive)
step_limit = 100        # optional

layout = """
#######
#A...B#
#..X..#
#######
"""

[symbols]               # "." is always a free cell
"#" = { barrier = true }
"X" = { absorbing = true }
"S" = { initial = 1.0 }      # initial-state weight; omit everywhere for uniform over free cells
"A" = { labels = ["A"] }
"*" = { labels = ["A", "B"] } # a cell may carry several labels
```

Full details are in the docstring of `src/msc_skill_machines/gridworld_builder.py`.
