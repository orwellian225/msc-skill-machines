"""Build a :class:`GridworldEnv` (and its per-label primitives) from a TOML description.

TOML schema::

    name = "office"
    step_limit = 100          # optional, default 1000
    seed = 1                  # optional
    props = ["A", "B"]        # proposition labels, in order
    cons  = ["B"]             # optional subset of props that are also constraints
    transitions = [[0, 1], [0, -1], [1, 0], [-1, 0]]   # optional

    layout = \"\"\"
    #######
    #A...B#
    #..X..#
    #S....#
    #######
    \"\"\"

    [symbols]
    "#" = { barrier = true }
    "X" = { absorbing = true }
    "S" = { initial = 1.0 }          # initial-state weight, normalised across all cells
    "A" = { labels = ["A"] }
    "*" = { labels = ["A", "B"] }    # a cell may carry several labels

``"."`` is always a plain free cell. If no symbol declares ``initial`` the initial state
distribution is uniform over free (non-barrier, non-absorbing) cells.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from msc_skill_machines.environments import GridworldEnv
from msc_skill_machines.primitives import GridworldTaskPrimitive

FREE_SYMBOL = "."
_SYMBOL_KEYS = {"barrier", "absorbing", "initial", "labels"}


@dataclass
class GridworldSpec:
    name: str
    props: list[str]
    cons: list[str]
    layout: list[str]
    symbols: dict[str, dict]
    step_limit: int = 1000
    seed: int | None = None
    transitions: npt.NDArray[np.int32] | None = None
    source: Path | None = field(default=None, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.layout), len(self.layout[0])

    @property
    def con_mask(self) -> list[bool]:
        return [p in self.cons for p in self.props]

    @property
    def all_labels(self) -> list[str]:
        return list(self.props) + [f"con_{c}" for c in self.props if c in self.cons]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def load_spec(path: str | Path) -> GridworldSpec:
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    for required in ("props", "layout"):
        if required not in data:
            raise ValueError(f"{path}: missing required key '{required}'")

    props = list(data["props"])
    if len(set(props)) != len(props):
        raise ValueError(f"{path}: duplicate entries in props {props}")

    cons = list(data.get("cons", []))
    unknown_cons = [c for c in cons if c not in props]
    if unknown_cons:
        raise ValueError(f"{path}: cons {unknown_cons} are not in props {props}")

    layout = [line for line in data["layout"].strip("\n").splitlines()]
    if not layout:
        raise ValueError(f"{path}: layout is empty")
    width = len(layout[0])
    ragged = [i for i, line in enumerate(layout) if len(line) != width]
    if ragged:
        raise ValueError(f"{path}: layout rows {ragged} do not match width {width} of row 0")

    symbols: dict[str, dict] = dict(data.get("symbols", {}))
    for sym, desc in symbols.items():
        if len(sym) != 1:
            raise ValueError(f"{path}: symbol keys must be single characters, got {sym!r}")
        bad_keys = set(desc) - _SYMBOL_KEYS
        if bad_keys:
            raise ValueError(f"{path}: symbol {sym!r} has unknown keys {sorted(bad_keys)}")
        bad_labels = [l for l in desc.get("labels", []) if l not in props]
        if bad_labels:
            raise ValueError(f"{path}: symbol {sym!r} uses labels {bad_labels} not in props {props}")
    if FREE_SYMBOL in symbols:
        raise ValueError(f"{path}: {FREE_SYMBOL!r} is reserved for free cells")

    used = {ch for line in layout for ch in line} - {FREE_SYMBOL}
    undefined = sorted(used - set(symbols))
    if undefined:
        raise ValueError(f"{path}: layout uses undefined symbols {undefined}")

    transitions = None
    if "transitions" in data:
        transitions = np.asarray(data["transitions"], dtype=np.int32)
        if transitions.ndim != 2 or transitions.shape[1] != 2:
            raise ValueError(f"{path}: transitions must be a list of [drow, dcol] pairs")

    return GridworldSpec(
        name=str(data.get("name", path.stem)),
        props=props,
        cons=cons,
        layout=layout,
        symbols=symbols,
        step_limit=int(data.get("step_limit", 1000)),
        seed=data.get("seed"),
        transitions=transitions,
        source=path,
    )


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

def build_env(spec: GridworldSpec) -> GridworldEnv:
    rows, cols = spec.shape
    barrier = np.zeros((rows, cols), dtype=np.bool_)
    absorbing = np.zeros((rows, cols), dtype=np.bool_)
    initial_weights = np.zeros((rows, cols), dtype=np.float64)
    prop_mask = np.zeros((rows, cols, len(spec.props)), dtype=np.bool_)
    prop_idx = {p: i for i, p in enumerate(spec.props)}

    any_initial = any("initial" in d for d in spec.symbols.values())

    for r, line in enumerate(spec.layout):
        for c, ch in enumerate(line):
            if ch == FREE_SYMBOL:
                continue
            desc = spec.symbols[ch]
            barrier[r, c] = bool(desc.get("barrier", False))
            absorbing[r, c] = bool(desc.get("absorbing", False))
            initial_weights[r, c] = float(desc.get("initial", 0.0))
            for label in desc.get("labels", []):
                prop_mask[r, c, prop_idx[label]] = True

    if any_initial:
        total = initial_weights.sum()
        if total <= 0:
            raise ValueError(f"{spec.name}: initial weights must sum to a positive value")
        initial = initial_weights / total
    else:
        free = ~barrier & ~absorbing
        if not free.any():
            raise ValueError(f"{spec.name}: no free cells for the initial state distribution")
        initial = free.astype(np.float64) / free.sum()

    con_columns = [prop_mask[:, :, prop_idx[p]] for p in spec.props if p in spec.cons]
    if con_columns:
        label_mask = np.concatenate([prop_mask, np.stack(con_columns, axis=2)], axis=2)
    else:
        label_mask = prop_mask

    return GridworldEnv(
        barrier_mask=barrier,
        absorbing_mask=absorbing,
        initial_state_distribution=initial,
        label_mask=label_mask,
        prop_labels=spec.props,
        con_mask=spec.con_mask,
        transitions=spec.transitions,
        step_limit=spec.step_limit,
        seed=spec.seed,
    )


def load_gridworld(path: str | Path) -> tuple[GridworldSpec, GridworldEnv]:
    spec = load_spec(path)
    return spec, build_env(spec)


def build_primitives(env: GridworldEnv, seed: int | None = None) -> dict[str, GridworldTaskPrimitive]:
    """One :class:`GridworldTaskPrimitive` per label (propositions and ``con_*`` constraints)."""
    return {label: GridworldTaskPrimitive(env, label, seed=seed) for label in env.all_labels}
