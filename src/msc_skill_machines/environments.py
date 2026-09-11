import logging
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

import typing as ty
import numpy.typing as npt

logger = logging.getLogger(__name__)

class GridworldEnv(gym.Env):
    type State = npt.NDArray[np.int32]
    type Info = dict[str, ty.Any]

    # Environment Description
    observation_space: gym.spaces.MultiDiscrete
    action_space: gym.spaces.Discrete
    goal_space: gym.spaces.MultiDiscrete

    prop_labels: list[str]
    con_labels: list[str]
    all_labels: list[str]

    # Environment Dynamics
    barrier_mask: npt.NDArray[np.bool_]
    absorbing_mask: npt.NDArray[np.bool_]
    initial_state_distribution: npt.NDArray[np.floating]
    label_mask: npt.NDArray[np.bool_] # A 3D tensor where the 3rd dimension is an assignment vector
    goal_mask: npt.NDArray[np.bool_] # A 2D tensor of goals - goals are any state with a non-zero assignment vector in the labelling mask
    transitions: npt.NDArray[np.int32]

    step_limit: int = 1000
    step_counter: int = 0

    hash: int
    short_hash: str

    npr: np.random.Generator
    seed: int | None

    def _env_hash(self) -> int:
        return hash(
            (self.barrier_mask.tobytes(), self.absorbing_mask.tobytes(), self.initial_state_distribution.tobytes(), self.label_mask.tobytes(), self.goal_mask.tobytes(), self.transitions.tobytes())
        )

    def __init__(self,
        barrier_mask: npt.NDArray[np.bool_],
        absorbing_mask: npt.NDArray[np.bool_],
        initial_state_distribution: npt.NDArray[np.floating],
        label_mask: npt.NDArray[np.bool_],
        prop_labels: list[str],
        con_mask: list[bool],
        transitions: ty.Optional[npt.NDArray[np.int32]],
        step_limit: int = 1000,
        seed: int | None = None,
    ):

        if transitions is None:
            transitions = np.array([
                [0, 1], [0, -1],
                [1, 0], [-1, 0],
            ])
        self.transitions = transitions

        self.barrier_mask = barrier_mask
        self.absorbing_mask = absorbing_mask
        self.initial_state_distribution = initial_state_distribution
        self.label_mask = label_mask

        self.prop_labels = list(prop_labels)
        self.con_labels = [ f"con_{prop}" for prop_idx, prop in enumerate(prop_labels) if con_mask[prop_idx] ]
        self.all_labels = self.prop_labels + self.con_labels

        if self.label_mask.ndim != 3 or self.label_mask.shape[2] != len(self.all_labels):
            raise ValueError(
                f"label_mask must have shape (H, W, {len(self.all_labels)}) for labels {self.all_labels}, got {self.label_mask.shape}"
            )

        self.goal_mask = self.label_mask[:, :, :len(self.prop_labels)].any(axis=2)

        self.observation_space = gym.spaces.MultiDiscrete( np.asarray( self.barrier_mask.shape ) )
        self.action_space = gym.spaces.Discrete( len(transitions) )
        self.goal_space = gym.spaces.MultiDiscrete( np.asarray( self.goal_mask.shape ) )

        self.hash = self._env_hash()
        self.short_hash = str(self.hash)[:8]

        self.step_limit = step_limit
        self.reset( seed )

    def reset(self, seed = None, **kwargs ) -> ty.Tuple[GridworldEnv.State, GridworldEnv.Info]:
        self.seed = seed
        self.npr = np.random.default_rng(seed)

        if "start_position" in kwargs:
            self.pos = kwargs["start_position"]
        else:
            self.pos = self._select_random_state(use_initial_state=True)

        logger.debug(f"gridworld-{self.short_hash}, reset: seed={seed}, start_position={self.pos}")

        self.step_counter = 0
        return self.pos, {}

    def step(self, action: int) -> ty.Tuple[ GridworldEnv.State, float, bool, bool, Info ]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        curr_state = self.pos
        next_state = self._next_state(curr_state, action)
        self.pos = next_state
        logger.debug(f"gridworld-{self.short_hash}, step {self.step_counter}: action={action}, curr_state={curr_state}, next_state={next_state}")
        logger.debug(f"gridworld-{self.short_hash}, step {self.step_counter}: curr_state_labels_set={self.label_assignment_to_set(self.label_mask[*curr_state])}, next_state_labels_set={self.label_assignment_to_set(self.label_mask[*next_state])}")

        self.step_counter += 1
        environment_reward = 0.
        is_terminated = self.state_is_absorbing( curr_state )
        is_truncated = self.step_counter > self.step_limit

        logger.debug(f"gridworld-{self.short_hash}, step {self.step_counter}: environment_reward={environment_reward}, is_terminated={is_terminated}, is_truncated={is_truncated}")

        return self.pos, environment_reward, is_terminated, is_truncated, {
            "curr_state": curr_state,
            "next_state": next_state,
            "curr_state_labels_assignment": self.label_mask[*curr_state],
            "next_state_labels_assignment": self.label_mask[*next_state],
            "curr_state_labels_set": self.label_assignment_to_set(self.label_mask[*curr_state]),
            "next_state_labels_set": self.label_assignment_to_set(self.label_mask[*next_state])
        }

    def _select_random_state(self, use_initial_state: bool = True) -> GridworldEnv.State:
        if use_initial_state:
            flat_probs = self.initial_state_distribution.ravel()
            flat_idx = self.npr.choice(flat_probs.size, p=flat_probs)
            row, col = np.unravel_index(flat_idx, self.barrier_mask.shape)
        else:
            valid_mask = ~self.barrier_mask
            valid_positions = np.argwhere(valid_mask)          # shape (N, 2)
            choice_idx = self.npr.integers(len(valid_positions))
            row, col = valid_positions[choice_idx]

        return np.array([row, col], dtype=np.int32)

    def _next_state(self, state: GridworldEnv.State, action: int) -> GridworldEnv.State:
        candidate_state = state + self.transitions[action]
        validate_candidate = all( 0 <= candidate_state[dim] < dim_max for dim, dim_max in enumerate(self.barrier_mask.shape)  )
        validate_candidate &= ~self.barrier_mask[*candidate_state] if validate_candidate else False
        validate_candidate &= ~self.absorbing_mask[*state] if validate_candidate else False
        return candidate_state if validate_candidate else state

    def state_is_goal(self, state: GridworldEnv.State) -> bool:
        return self.goal_mask[*state]

    def sample_goal(self) -> GridworldEnv.State:
        goal_indices = np.where(self.goal_mask)
        return np.array([goal_indices[0][0], goal_indices[1][0]], dtype=np.int32)

    def state_is_absorbing(self, state: GridworldEnv.State) -> bool:
        return self.absorbing_mask[*state]

    def state_is_barrier(self, state: GridworldEnv.State) -> bool:
        return self.barrier_mask[*state]

    def state_is_initial(self, state: GridworldEnv.State) -> bool:
        return self.initial_state_distribution[*state] > 0.

    def state_to_label_assignment(self, state: GridworldEnv.State) -> npt.NDArray[np.bool_]:
        return self.label_mask[*state]

    def label_assignment_to_set(self, label_assignment: npt.NDArray[np.bool_]) -> set[str]:
        return { label for label_idx, label in enumerate(self.all_labels) if label_assignment[label_idx] }

    def label_str_to_idx(self, label: str) -> int:
        if label in self.all_labels:
            return self.all_labels.index(label)
        raise ValueError(f"Label {label} not found in {self.all_labels}")

    def pull_props_from_set(self, label_set: set[str]) -> set[str]: return label_set & set(self.prop_labels)
    def pull_props_from_assignment(self, label_assignment: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        return label_assignment[:len(self.prop_labels)]
    def pull_cons_from_set(self, label_set: set[str]) -> set[str]: return label_set & set(self.con_labels)
    def pull_cons_from_assignment(self, label_assignment: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
        return label_assignment[len(self.prop_labels):]
