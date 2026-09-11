import logging
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

import typing as ty
import numpy.typing as npt

from msc_skill_machines.environments import GridworldEnv

logger = logging.getLogger(__name__)

class GridworldTask(gym.Env):
    env: GridworldEnv
    rfunc: ty.Callable[[GridworldEnv.State, int, GridworldEnv.State], float]

    def __init__(self, env: GridworldEnv, reward_mask: npt.NDArray[np.float32]):
        self.env = env
        self.reward_mask = reward_mask

    def reset(self, seed = None, **kwargs  ) -> ty.Tuple[GridworldEnv.State, dict[str, ty.Any]]:
        return self.env.reset(seed=seed, **kwargs)

    def step(self, action: int) -> tuple[GridworldEnv.State, float, bool, bool, dict]:
        next_state, env_reward, is_terminated, is_truncated, info = self.env.step(action)
        curr_state = info["curr_state"]
        return next_state, env_reward + self.rfunc(curr_state, action, next_state), is_terminated, is_truncated, info
