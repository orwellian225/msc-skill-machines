from abc import abstractmethod
import logging
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

import typing as ty
import numpy.typing as npt

from msc_skill_machines.environments import GridworldEnv
from msc_skill_machines.value_functions import State, Action, StateValueFunction, ActionValueFunction

logger = logging.getLogger(__name__)

class Task(ty.Protocol):
    @abstractmethod
    def identifier(self) -> str:
        pass

    @abstractmethod
    def reset(self, seed = None, **kwargs) -> ty.Tuple[State, dict[str, ty.Any]]:
        pass

    @abstractmethod
    def step(self, action: Action) -> ty.Tuple[State, float, bool, bool, dict]:
        pass

    @abstractmethod
    def sample_action(self, random_generator: np.random.Generator) -> Action:
        pass

    @abstractmethod
    def sample_state(self, random_generator: np.random.Generator) -> State:
        pass

    @abstractmethod
    def state_value_array(self, vfunc: StateValueFunction) -> npt.NDArray[np.float32]:
        pass

    @abstractmethod
    def action_value_array(self, qfunc: ActionValueFunction) -> npt.NDArray[np.float32]:
        pass

class GridworldTask(gym.Env):
    env: GridworldEnv
    target_label_idx: int
    target_label: str

    def __init__(self, env: GridworldEnv, target_label: str):
        self.env = env
        self.target_label = target_label
        self.target_label_idx = env.label_str_to_idx(target_label)

    def reset(self, seed = None, **kwargs  ) -> ty.Tuple[GridworldEnv.State, dict[str, ty.Any]]:
        return self.env.reset(seed=seed, **kwargs)

    def step(self, action: int) -> tuple[GridworldEnv.State, float, bool, bool, dict]:
        next_state, env_reward, is_terminated, is_truncated, info = self.env.step(action)
        curr_state = info["curr_state"]
        return next_state, env_reward + self.reward(curr_state, action, next_state), is_terminated, is_truncated, info

    def reward(self,
        curr_state, action, next_state
    ) -> float:
        labels = self.env.state_to_label_assignment(curr_state).copy()
        return 1. if bool(labels[self.target_label_idx]) else 0.

    def identifier(self) -> str:
        return f"task-{self.target_label}"

    def sample_action(self, random_generator: np.random.Generator) -> int:
        return int(random_generator.integers(0, self.env.action_space.n))

    def sample_state(self, random_generator: np.random.Generator) -> GridworldEnv.State:
        return tuple(random_generator.integers(0, self.env.observation_space.shape))

    def state_value_array(self, vfunc: StateValueFunction) -> npt.NDArray[np.float32]:
        rows, cols = self.env.barrier_mask.shape
        data = np.zeros((rows, cols), dtype=np.float32)
        for row in range(rows):
            for col in range(cols):
                if self.env.barrier_mask[row, col]:
                    continue
                data[row, col] = vfunc((row, col))
        return data

    def action_value_array(self, qfunc: ActionValueFunction) -> npt.NDArray[np.float32]:
        rows, cols = self.env.barrier_mask.shape
        n_actions = self.env.action_space.n
        data = np.zeros((rows, cols, n_actions), dtype=np.float32)
        for row in range(rows):
            for col in range(cols):
                if self.env.barrier_mask[row, col]:
                    continue
                for action in range(n_actions):
                    data[row, col, action] = qfunc((row, col), action)
        return data
