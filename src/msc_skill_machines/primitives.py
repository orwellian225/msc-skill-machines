from abc import abstractmethod
import logging
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

import typing as ty
import numpy.typing as npt

from msc_skill_machines.value_functions import State, Action, StateValueFunction, ActionValueFunction
from msc_skill_machines.environments import GridworldEnv
from msc_skill_machines.tasks import GridworldTask

class TaskPrimitive(ty.Protocol):

    type ConstraintAssignment = int
    type LabelAssignment = int

    type State = ty.Tuple[State, ConstraintAssignment] | LabelAssignment
    type Action = ty.Tuple[Action, bool]

    @abstractmethod
    def identifier(self) -> str:
        pass

    @abstractmethod
    def reward(self,
        curr_env_state: State, curr_cons: ConstraintAssignment,
        env_action: Action, term_action: bool,
        next_labels: LabelAssignment
    ) -> float:
        pass

    @abstractmethod
    def reset(self, seed: int | None = None, **kwargs) -> ty.Tuple[TaskPrimitive.State, dict[str, ty.Any]]:
        pass

    @abstractmethod
    def step(self, action: TaskPrimitive.Action) -> ty.Tuple[TaskPrimitive.State, float, bool, bool, dict[str, ty.Any]]:
        pass

    @abstractmethod
    def sample_state(self, randgen: np.random.Generator) -> TaskPrimitive.State:
        pass

    @abstractmethod
    def sample_action(self, randgen: np.random.Generator) -> TaskPrimitive.Action:
        pass

    @abstractmethod
    def env_state_vfunc(self, vfunc: StateValueFunction) -> ty.Any:
        pass

    @abstractmethod
    def env_action_vfunc(self, qfunc: ActionValueFunction) -> ty.Any:
        pass

    @staticmethod
    def bits_to_int(arr: np.ndarray) -> int:
        packed = np.packbits(arr, bitorder='little')
        return int.from_bytes(packed.tobytes(), byteorder='little')

    @staticmethod
    def int_to_bits(n: int, length: int) -> np.ndarray:
        nbytes = (length + 7) // 8
        packed = np.frombuffer(n.to_bytes(nbytes, byteorder='little'), dtype=np.uint8)
        return np.unpackbits(packed, count=length, bitorder='little').astype(np.bool_)

class GridworldTaskPrimitive(TaskPrimitive):
    env: GridworldEnv
    target_label: str
    target_label_idx: int

    def __init__(self, env: GridworldEnv, target_label: str, seed: int | None = None):
        self.env = env
        self.target_label = target_label
        self.target_label_idx = env.label_str_to_idx(target_label)

        self.state, _ = self.reset(seed)

    def identifier(self) -> str:
        return f"primitive-{self.target_label}"

    def reward(self,
        curr_env_state: State, curr_cons: TaskPrimitive.ConstraintAssignment,
        env_action: Action, term_action: bool,
        next_labels: TaskPrimitive.LabelAssignment
    ) -> float:
        return 1. if term_action and self.int_to_bits(next_labels | curr_cons, len(self.env.all_labels))[self.target_label_idx] else 0.

    def reset(self, seed: int | None = None, **kwargs) -> ty.Tuple[TaskPrimitive.State, dict[str, ty.Any]]:
        env_state, env_info = self.env.reset(seed)
        self.state = (env_state, 0)
        return self.state, { **env_info }

    def step(self, action: TaskPrimitive.Action) -> ty.Tuple[TaskPrimitive.State, float, bool, bool, dict[str, ty.Any]]:
        if not isinstance(self.state, tuple):
            raise ValueError(f"Cannot call step with Goal state {self.state}")

        curr_env_state, curr_cons = self.state
        env_action, terminate_action = action
        next_env_state, env_reward, env_terminal, env_truncated, env_info = self.env.step(env_action)

        curr_labels = self.env.state_to_label_assignment(curr_env_state)
        next_labels = self.env.state_to_label_assignment(next_env_state)
        next_cons = curr_cons | self.bits_to_int( self.env.pull_cons_from_assignment(curr_labels) ^ self.env.pull_cons_from_assignment(next_labels) )

        next_state = (next_env_state, next_cons) if not terminate_action else self.bits_to_int(next_labels) | curr_cons
        self.state = next_state

        reward = self.reward(curr_env_state, curr_cons, env_action, terminate_action, self.bits_to_int(next_labels))
        return next_state, env_reward + reward, env_terminal or terminate_action, env_truncated, { **env_info }

    def sample_state(self, randgen: np.random.Generator) -> TaskPrimitive.State:
        return (tuple(randgen.integers(0, self.env.observation_space.shape)), randgen.integers(0, 2**len(self.env.con_labels)))

    def sample_action(self, randgen: np.random.Generator) -> TaskPrimitive.Action:
        return (int(randgen.integers(0, self.env.action_space.n)), randgen.random() > 0.5 )

    def sample_goal(self, randgen: np.random.Generator) -> TaskPrimitive.Goal:
        return int(randgen.integers(0, 2**len(self.env.all_labels)))

    def env_state_vfunc(self, vfunc: StateValueFunction) -> ty.Any:
        pass

    def env_action_vfunc(self, qfunc: ActionValueFunction) -> npt.NDArray[np.float32]:
        rows, cols = self.env.barrier_mask.shape
        n_actions = len(self.env.transitions)
        data = np.zeros((rows, cols, 2 * n_actions), dtype=np.float32)
        for row in range(rows):
            for col in range(cols):
                if self.env.barrier_mask[row, col]:
                    continue
                state = ((row, col), 0)
                for action in range(n_actions):
                    data[row, col, action] = qfunc(state, (action, False))
                    data[row, col, n_actions + action] = qfunc(state, (action, True))
        return data
