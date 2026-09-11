"""Build a :class:`GridworldEnv` (and its per-label primitives) from a TOML description.

TOML schema::

    name = "office"
    seed = 1                  # optional; drives random label placement and the env reset
    step_limit = 100          # optional, default 1000
    transitions = [[0, 1], [0, -1], [1, 0], [-1, 0]]   # optional

    layout = \"\"\"
    #######
    #A....#
    #.(S,X,A).#     # a bracket group puts several symbols in one cell; spaces are ignored
    #S...B#
    #######
    \"\"\"

    [[labels]]
    identifier = "A"
    constraint = true         # also creates a con_A label / primitive
    random = true             # place num_states extra A cells on random non-barrier cells
    num_states = 4

    [[labels]]
    identifier = "B"
    constraint = false
    random = false            # only where the map says; num_states is ignored

Reserved map symbols: ``#`` barrier, ``X`` absorbing, ``S`` initial state, ``.`` free cell.
Every other symbol must be a label identifier. Multi-character identifiers are allowed but can
only appear inside a bracket group. All ``S`` cells share the initial distribution uniformly; if
there is no ``S`` at all it is uniform over free (non-barrier, non-absorbing) cells.

Random labels are sampled independently (with the same seeded generator, in file order), so two
random labels may share a cell, and may also land on ``S``/``X`` cells or fixed labels.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from msc_skill_machines.environments import GridworldEnv
from msc_skill_machines.primitives import GridworldTaskPrimitive

BARRIER, ABSORBING, INITIAL, FREE = "#", "X", "S", "."
RESERVED = {BARRIER, ABSORBING, INITIAL, FREE}

type Cell = frozenset[str]
type Layout = list[list[Cell]]


@dataclass
class LabelSpec:
    identifier: str
    constraint: bool = False
    random: bool = False
    num_states: int = 0


@dataclass
class GridworldSpec:
    name: str
    labels: list[LabelSpec]
    layout: Layout
    step_limit: int = 1000
    seed: int | None = None
    transitions: npt.NDArray[np.int32] | None = None
    source: Path | None = field(default=None, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.layout), len(self.layout[0])

    @property
    def props(self) -> list[str]:
        return [l.identifier for l in self.labels]

    @property
    def cons(self) -> list[str]:
        return [l.identifier for l in self.labels if l.constraint]

    @property
    def con_mask(self) -> list[bool]:
        return [l.constraint for l in self.labels]

    @property
    def all_labels(self) -> list[str]:
        return self.props + [f"con_{c}" for c in self.cons]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def tokenise_layout(text: str) -> Layout:
    """Split a layout string into rows of cells, each cell a set of symbols.

    Outside brackets every non-space character is one cell. ``(a, b, c)`` is one cell holding
    all listed symbols. Rows must end up with the same number of cells.
    """
    rows: Layout = []
    for line_no, line in enumerate(text.strip("\n").splitlines()):
        row: list[Cell] = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch.isspace():
                i += 1
            elif ch == "(":
                end = line.find(")", i)
                if end == -1:
                    raise ValueError(f"layout row {line_no}: unclosed '(' at column {i}")
                parts = [p.strip() for p in line[i + 1:end].split(",")]
                symbols = frozenset(p for p in parts if p)
                if not symbols:
                    raise ValueError(f"layout row {line_no}: empty group at column {i}")
                row.append(symbols)
                i = end + 1
            elif ch == ")":
                raise ValueError(f"layout row {line_no}: stray ')' at column {i}")
            else:
                row.append(frozenset({ch}))
                i += 1
        if row:
            rows.append(row)

    if not rows:
        raise ValueError("layout is empty")
    width = len(rows[0])
    ragged = [i for i, row in enumerate(rows) if len(row) != width]
    if ragged:
        raise ValueError(f"layout rows {ragged} do not have {width} cells like row 0")
    return rows


def load_spec(path: str | Path) -> GridworldSpec:
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    for required in ("labels", "layout"):
        if required not in data:
            raise ValueError(f"{path}: missing required key '{required}'")

    labels: list[LabelSpec] = []
    for i, entry in enumerate(data["labels"]):
        if "identifier" not in entry:
            raise ValueError(f"{path}: labels[{i}] has no identifier")
        label = LabelSpec(
            identifier=str(entry["identifier"]),
            constraint=bool(entry.get("constraint", False)),
            random=bool(entry.get("random", False)),
            num_states=int(entry.get("num_states", 0)),
        )
        if not label.identifier or any(c.isspace() or c in "(),." for c in label.identifier):
            raise ValueError(f"{path}: label identifier {label.identifier!r} is invalid")
        if label.identifier in RESERVED:
            raise ValueError(f"{path}: label identifier {label.identifier!r} is a reserved symbol")
        if label.random and label.num_states < 1:
            raise ValueError(f"{path}: label {label.identifier!r} is random but num_states is {label.num_states}")
        labels.append(label)

    identifiers = [l.identifier for l in labels]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path}: duplicate label identifiers in {identifiers}")

    try:
        layout = tokenise_layout(data["layout"])
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None

    known = RESERVED | set(identifiers)
    unknown = sorted({sym for row in layout for cell in row for sym in cell} - known)
    if unknown:
        raise ValueError(f"{path}: layout uses unknown symbols {unknown} (labels: {identifiers})")

    transitions = None
    if "transitions" in data:
        transitions = np.asarray(data["transitions"], dtype=np.int32)
        if transitions.ndim != 2 or transitions.shape[1] != 2:
            raise ValueError(f"{path}: transitions must be a list of [drow, dcol] pairs")

    return GridworldSpec(
        name=str(data.get("name", path.stem)),
        labels=labels,
        layout=layout,
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
    initial_cells = np.zeros((rows, cols), dtype=np.bool_)
    prop_mask = np.zeros((rows, cols, len(spec.labels)), dtype=np.bool_)
    prop_idx = {l.identifier: i for i, l in enumerate(spec.labels)}

    for r, row in enumerate(spec.layout):
        for c, cell in enumerate(row):
            barrier[r, c] = BARRIER in cell
            absorbing[r, c] = ABSORBING in cell
            initial_cells[r, c] = INITIAL in cell
            for sym in cell:
                if sym in prop_idx:
                    prop_mask[r, c, prop_idx[sym]] = True

    # Random label placement: seeded, in file order, independent per label
    rng = np.random.default_rng(spec.seed)
    eligible = np.argwhere(~barrier)
    for label in spec.labels:
        if not label.random:
            continue
        if label.num_states > len(eligible):
            raise ValueError(
                f"{spec.name}: label {label.identifier!r} wants {label.num_states} random cells "
                f"but only {len(eligible)} non-barrier cells exist"
            )
        picks = eligible[rng.choice(len(eligible), size=label.num_states, replace=False)]
        prop_mask[picks[:, 0], picks[:, 1], prop_idx[label.identifier]] = True

    if initial_cells.any():
        initial = initial_cells.astype(np.float64) / initial_cells.sum()
    else:
        free = ~barrier & ~absorbing
        if not free.any():
            raise ValueError(f"{spec.name}: no free cells for the initial state distribution")
        initial = free.astype(np.float64) / free.sum()

    con_columns = [prop_mask[:, :, prop_idx[l.identifier]] for l in spec.labels if l.constraint]
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
