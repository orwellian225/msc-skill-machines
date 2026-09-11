import logging
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

import typing as ty
import numpy.typing as npt

from msc_skill_machines.environments import GridworldEnv
from msc_skill_machines.tasks import GridworldTask

class GridworldTaskPrimitive(gym.Env):
    type State = ty.Tuple[GridworldEnv.State, npt.NDArray[np.bool_]]
    type Action = ty.Tuple[int, bool]
    type Info = dict[str, ty.Any]

    env: GridworldEnv
    target_label_idx: int
    target_label: str
    rfunc: ty.Callable[[GridworldEnv.State, int, GridworldEnv.State], float]

    state: GridworldTaskPrimitive.State

    def __init__(self,
        env: GridworldEnv,
        target_label: str,
        seed: int | None = None
    ):

        self.env = env
        self.target_label = target_label
        self.target_label_idx = env.label_str_to_idx(target_label)

        self.state, _info = self.reset(seed)

    def reset(self, seed = None, **kwargs ) -> ty.Tuple[GridworldTaskPrimitive.State, dict[str, ty.Any]]:
        env_state, info = self.env.reset(seed=seed, **kwargs)
        return (env_state, np.zeros(len(self.env.con_labels), dtype=np.bool_)), info

    def step(self, action: GridworldTaskPrimitive.Action) -> ty.Tuple[GridworldTaskPrimitive.State, float, bool, bool, dict[str, ty.Any]]:
        env_curr_state, curr_violated_constraints = self.state
        env_action, terminate_action = action

        env_next_state, env_reward, env_is_terminated, env_is_truncated, env_info = self.env.step(env_action)
        env_next_state, next_violated_constraints = self.next_state(
            env_curr_state, env_next_state,
            curr_violated_constraints, env_action, terminate_action
        )

        primitive_next_state = (env_next_state, next_violated_constraints)
        primitive_reward = self.reward( env_curr_state, curr_violated_constraints, env_action, terminate_action)
        primitive_is_terminated = env_is_terminated or terminate_action
        primitive_is_truncated = env_is_truncated
        primitive_info = env_info | {
            "curr_violated_constraints": curr_violated_constraints,
            "next_violated_constraints": next_violated_constraints,
        }

        return primitive_next_state, primitive_reward, primitive_is_terminated, primitive_is_truncated, primitive_info

    def next_state(self,
        env_curr_state: GridworldEnv.State,
        env_next_state: GridworldEnv.State,
        violated_constraints: npt.NDArray[np.bool_],
        env_action: int,
        terminate_action: bool
    ) -> GridworldTaskPrimitive.State:
        curr_state_labels = self.env.state_to_label_assignment(env_curr_state)
        next_state_labels = self.env.state_to_label_assignment(env_next_state)

        if terminate_action:
            violated_constraints = self.env.pull_cons_from_assignment(next_state_labels) | violated_constraints
        else:
            env_next_state = env_curr_state
            violated_constraints = violated_constraints | self.env.pull_cons_from_assignment(curr_state_labels ^ next_state_labels)

        return ( env_next_state, violated_constraints )

    def reward(self,
        env_curr_state: GridworldEnv.State,
        violated_constraints: npt.NDArray[np.bool_],
        _env_action: int,
        terminate_action: bool
    ) -> float:
        # Full-width label assignment: state labels OR-ed with the constraints violated so far
        labels = self.env.state_to_label_assignment(env_curr_state).copy()
        labels[len(self.env.prop_labels):] |= violated_constraints

        return 1. if (terminate_action and bool(labels[self.target_label_idx])) else 0.

    # ------------------------------------------------------------------ #
    # Visualisation helpers
    # ------------------------------------------------------------------ #

    def action_labels(self) -> list[str]:
        """Human readable name for every primitive action, in the same order as the
        third axis of :meth:`reward_map`: ``(move0, stay), (move0, term), (move1, stay), ...``"""
        names = []
        for delta in self.env.transitions:
            delta_str = f"({int(delta[0])},{int(delta[1])})"
            names.append(f"{delta_str} stay")
            names.append(f"{delta_str} term")
        return names

    def reward_map(self) -> npt.NDArray[np.float32]:
        """Evaluate the reward function over every state and every primitive action.

        Returns a tensor of shape ``(H, W, 2 * len(transitions))`` where layer
        ``2 * a + t`` holds ``reward(state, no_violations, a, bool(t))``. Barrier cells are 0.
        """
        rows, cols = self.env.barrier_mask.shape
        n_moves = len(self.env.transitions)
        no_violations = np.zeros(len(self.env.con_labels), dtype=np.bool_)
        out = np.zeros((rows, cols, 2 * n_moves), dtype=np.float32)

        for row in range(rows):
            for col in range(cols):
                if self.env.barrier_mask[row, col]:
                    continue
                state = np.array([row, col], dtype=np.int32)
                for move in range(n_moves):
                    for terminate in (False, True):
                        out[row, col, 2 * move + int(terminate)] = self.reward(state, no_violations, move, terminate)
        return out

    def target_goal_mask(self) -> npt.NDArray[np.bool_]:
        """2D mask of the cells labelled with this primitive's target label."""
        return self.env.label_mask[:, :, self.target_label_idx]
